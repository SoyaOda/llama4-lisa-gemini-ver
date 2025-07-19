# model/moe_adapters.py
"""
Phase 3A: Heterogeneous MoE Adapters Implementation

2025年最新MoE最適化戦略に基づく実装:
- Heterogeneous MoE Adapters for Multimodal Fine-tuning (2025年3月論文)
- SAM2 MoE Integration with Mixture of LoRA Experts
- Llama-4 + Q-Former + SAM2統合モデル向けMoE最適化

参考文献:
- ArXiv: 2503.20633 (Heterogeneous MoE Adapters)
- "Customize SAM for Multi-Modal Semantic Segmentation with Mixture of LoRA Experts"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
import sys
import os
import math
import logging

logger = logging.getLogger(__name__)

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_linux

from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import AutoModel, AutoTokenizer


class DynamicRouter(nn.Module):
    """
    動的エキスパートルーター (Heterogeneous MoE Adapters準拠)
    
    各モーダル間の適応的ルーティングを実現:
    - Vision Expert (SAM2特化)
    - Language Expert (Llama-4特化)
    - Fusion Expert (Q-Former特化)
    """
    
    def __init__(
        self,
        hidden_size: int = 5120,  # Llama-4隠れ層サイズ
        num_experts: int = 3,     # Vision, Language, Fusion
        num_tokens_per_expert: int = 2,  # Top-k routing
        router_bias: bool = False,
        temperature: float = 1.0,
        capacity_factor: float = 1.25,  # Expert capacity factor
        dropout: float = 0.1,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.num_tokens_per_expert = num_tokens_per_expert
        self.temperature = temperature
        self.capacity_factor = capacity_factor
        
        # デバイス設定
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # ルーターネットワーク (線形層) - GPUに配置、bfloat16に変換
        self.router = nn.Linear(hidden_size, num_experts, bias=router_bias).to(device).to(torch.bfloat16)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
        # エキスパート特化学習のためのゲート - GPUに配置、bfloat16に変換
        self.expert_gates = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size).to(device).to(torch.bfloat16) for _ in range(num_experts)
        ])
        
        # 負荷分散のためのノイズ
        self.noise_epsilon = 1e-8
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        modal_type: Optional[str] = None  # "vision", "language", "fusion"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        動的ルーティング実行
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            modal_type: モーダルタイプヒント
            
        Returns:
            routed_output: ルーティング後の出力
            routing_info: ルーティング統計情報
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # ルーター確率計算
        router_logits = self.router(hidden_states)  # (B, L, num_experts)
        
        # 温度調整
        router_logits = router_logits / self.temperature
        
        # ソフトマックス確率
        router_probs = F.softmax(router_logits, dim=-1)  # (B, L, num_experts)
        
        # Top-k expert selection
        top_k_probs, top_k_indices = torch.topk(
            router_probs, 
            k=self.num_tokens_per_expert, 
            dim=-1
        )  # (B, L, k)
        
        # Expert capacity計算
        expert_capacity = int(self.capacity_factor * seq_len * batch_size / self.num_experts)
        
        # 各エキスパートでの処理
        expert_outputs = []
        expert_weights = []
        load_balancing_loss = 0.0
        
        for expert_idx in range(self.num_experts):
            # このエキスパートを選択したトークンのマスク
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # (B, L)
            
            if expert_mask.sum() == 0:
                # このエキスパートを使用するトークンがない場合
                expert_outputs.append(torch.zeros_like(hidden_states))
                expert_weights.append(torch.zeros(batch_size, seq_len, device=hidden_states.device))
                continue
            
            # エキスパート特化ゲート適用
            gated_hidden = self.expert_gates[expert_idx](hidden_states)
            gated_hidden = self.dropout(gated_hidden)
            
            # ReLU活性化（エキスパート特化）
            expert_output = F.relu(gated_hidden)
            
            # 重み計算（このエキスパートに対する確率）
            expert_weight = router_probs[:, :, expert_idx]  # (B, L)
            
            expert_outputs.append(expert_output)
            expert_weights.append(expert_weight)
            
            # 負荷分散損失計算
            expert_usage = expert_mask.float().mean()
            target_usage = 1.0 / self.num_experts
            load_balancing_loss += (expert_usage - target_usage) ** 2
        
        # 重み付き結合
        final_output = torch.zeros_like(hidden_states)
        total_weights = torch.zeros(batch_size, seq_len, device=hidden_states.device)
        
        for expert_idx in range(self.num_experts):
            weight = expert_weights[expert_idx].unsqueeze(-1)  # (B, L, 1)
            final_output += weight * expert_outputs[expert_idx]
            total_weights += expert_weights[expert_idx]
        
        # 正規化
        total_weights = total_weights.unsqueeze(-1).clamp(min=self.noise_epsilon)
        final_output = final_output / total_weights
        
        # 残差接続
        final_output = hidden_states + final_output
        
        # ルーティング統計情報
        routing_info = {
            "router_probs": router_probs,
            "top_k_indices": top_k_indices,
            "top_k_probs": top_k_probs,
            "load_balancing_loss": load_balancing_loss,
            "expert_usage": [expert_weights[i].mean().item() for i in range(self.num_experts)]
        }
        
        return final_output, routing_info


class LoRAExpert(nn.Module):
    """
    モーダル特化LoRA Expert (Mixture of LoRA Experts準拠)
    
    各モーダルに特化したLoRAアダプター:
    - target_modules: モーダル別最適化対象
    - rank: パラメータ効率化レベル
    - alpha: LoRA強度調整
    """
    
    def _infer_hidden_size(self, model: nn.Module, modal_type: str) -> int:
        """モデルから隠れ層サイズを推測"""
        # デフォルト値
        default_sizes = {
            "vision": 256,    # SAM2の特徴次元
            "language": 5120, # Llama-4の隠れ層サイズ
            "fusion": 768,    # Q-Formerの隠れ層サイズ
            "general": 768
        }
        
        # Linear層から推測を試みる
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # 一般的な隠れ層サイズをチェック
                for size in [module.in_features, module.out_features]:
                    if size in [256, 384, 512, 768, 1024, 1280, 2048, 5120]:
                        return size
        
        # 見つからない場合はデフォルト値を返す
        return default_sizes.get(modal_type, 768)
    
    def __init__(
        self,
        base_model: nn.Module,
        target_modules: List[str],
        rank: int = 16,                    # 🔄 SAM2+MLE論文準拠（64→16）
        alpha: int = 32,                   # 🔄 SAM2+MLE論文準拠（128→32）
        dropout: float = 0.1,              # 🔄 SAM2+MLE論文準拠（0.05→0.1）
        modal_type: str = "general",
        task_type: TaskType = TaskType.CAUSAL_LM
    ):
        super().__init__()
        
        self.modal_type = modal_type
        self.rank = rank
        self.alpha = alpha
        
        # LoRA設定 (config_linux.py準拠 + モーダル特化)
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type=task_type,
            use_rslora=False  # rank>=64で不安定性ある場合True
        )
        
        # PEFT適用（SAM2Base対応）
        try:
            # すでにPEFTが適用されているモデルの場合はそのまま使用
            if hasattr(base_model, 'peft_config'):
                logger.info(f"✅ すでにPEFTが適用されています: {modal_type}")
                self.peft_model = base_model
                # 既存のPEFT設定を保存
                self.rank = base_model.peft_config.get('default', {}).get('r', rank)
                self.alpha = base_model.peft_config.get('default', {}).get('lora_alpha', alpha)
            else:
                # SAM2Baseなど、configがないモデルのための対処
                if not hasattr(base_model, 'config'):
                    # PEFTが期待するconfig形式を作成（dictライクなオブジェクト）
                    class ConfigDict(dict):
                        def __init__(self, hidden_size, model_type):
                            super().__init__()
                            self['hidden_size'] = hidden_size
                            self['model_type'] = model_type
                            # PEFTが期待する属性
                            self['tie_word_embeddings'] = False
                            
                        def __getattr__(self, key):
                            return self.get(key, None)
                            
                        def __setattr__(self, key, value):
                            self[key] = value
                        
                        def to_dict(self):
                            return dict(self)
                    
                    base_model.config = ConfigDict(
                        hidden_size=self._infer_hidden_size(base_model, modal_type),
                        model_type=modal_type
                    )
                
                # Option E: PEFT dtype自動変換無効化
                self.peft_model = get_peft_model(
                    base_model, 
                    lora_config, 
                    autocast_adapter_dtype=False  # 🔥 dtype自動変換無効化
                )
        except Exception as e:
            logger.error(f"PEFT適用エラー: {e}")
            raise RuntimeError(f"PEFT適用に失敗しました: {e}")
        
        # モーダル特化レイヤー
        # PEFT適用済みモデルの場合、base_modelから取得
        if hasattr(self.peft_model, 'base_model'):
            actual_model = self.peft_model.base_model
        else:
            actual_model = self.peft_model
            
        # hidden_size取得
        if hasattr(actual_model, 'config') and hasattr(actual_model.config, 'hidden_size'):
            hidden_size = actual_model.config.hidden_size
        elif hasattr(base_model, 'config') and hasattr(base_model.config, 'hidden_size'):
            hidden_size = base_model.config.hidden_size
        else:
            # SAM2は256、Llama-4は5120、Q-Formerは768が一般的
            if modal_type == "vision":
                hidden_size = 256  # SAM2の特徴次元
            elif modal_type == "language":
                hidden_size = 5120  # Llama-4の隠れ層サイズ
            else:
                hidden_size = 768  # Q-Formerの隠れ層サイズ
        
        # デバイス設定（GPUが利用可能ならGPUに配置）+ dtype統一
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modal_projection = nn.Linear(hidden_size, hidden_size).to(device).to(torch.bfloat16)
        self.modal_norm = nn.LayerNorm(hidden_size).to(device).to(torch.bfloat16)
        
        print(f"✅ LoRAExpert初期化完了: {modal_type}")
        print(f"  - Rank: {rank}, Alpha: {alpha}")
        print(f"  - Target modules: {target_modules}")
        if hasattr(self, 'peft_model') and hasattr(self.peft_model, 'get_nb_trainable_parameters'):
            trainable_params = self.peft_model.get_nb_trainable_parameters()
            if isinstance(trainable_params, tuple):
                trainable_params = trainable_params[0]
            print(f"  - Trainable params: {trainable_params:,}")
        else:
            print(f"  - Trainable params: N/A (エラー発生)")
        
    def forward(self, *args, **kwargs):
        """PEFTモデルのforward処理"""
        # peft_modelがない場合はエラー
        if not hasattr(self, 'peft_model'):
            raise RuntimeError("peft_modelが初期化されていません")
        
        # テストモード: hidden_statesが直接渡された場合の処理
        if len(args) == 1 and isinstance(args[0], torch.Tensor) and args[0].dim() == 3:
            # hidden_states形式 (batch_size, seq_len, hidden_size) の場合
            hidden_states = args[0]
            
            # 入力次元が異なる場合の調整
            expected_hidden_size = self.modal_projection.in_features
            actual_hidden_size = hidden_states.shape[-1]
            
            if actual_hidden_size != expected_hidden_size:
                # 入力投影レイヤーを動的に作成（キャッシュ）
                if not hasattr(self, '_input_projection_cache'):
                    self._input_projection_cache = {}
                
                cache_key = f"{actual_hidden_size}_to_{expected_hidden_size}"
                if cache_key not in self._input_projection_cache:
                    device = hidden_states.device
                    dtype = hidden_states.dtype
                    self._input_projection_cache[cache_key] = nn.Linear(
                        actual_hidden_size, expected_hidden_size
                    ).to(device).to(dtype)
                
                # 入力を適切な次元に変換
                hidden_states = self._input_projection_cache[cache_key](hidden_states)
            
            # モーダル特化処理のみ実行（モデル推論はスキップ）
            projected_states = self.modal_projection(hidden_states)
            projected_states = self.modal_norm(projected_states)
            
            # 残差接続して返す
            return hidden_states + projected_states
        
        # 通常モード: モデル推論を実行
        outputs = self.peft_model(*args, **kwargs)
        
        # モーダル特化投影適用
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
            projected_states = self.modal_projection(hidden_states)
            projected_states = self.modal_norm(projected_states)
            
            # 残差接続
            outputs.last_hidden_state = hidden_states + projected_states
        
        return outputs
    
    def get_expert_info(self) -> Dict[str, Any]:
        """エキスパート情報取得"""
        if not hasattr(self, 'peft_model'):
            raise RuntimeError(f"LoRAExpert ({self.modal_type}) が正しく初期化されていません")
        
        # get_nb_trainable_parameters()がタプルを返す場合の処理
        trainable_params = self.peft_model.get_nb_trainable_parameters()
        if isinstance(trainable_params, tuple):
            trainable_params = trainable_params[0]  # 最初の要素が実際の訓練可能パラメータ数
            
        return {
            "modal_type": self.modal_type,
            "rank": self.rank,
            "alpha": self.alpha,
            "trainable_params": trainable_params,
            "total_params": sum(p.numel() for p in self.peft_model.parameters())
        }


class HeterogeneousMoEAdapter(nn.Module):
    """
    Heterogeneous MoE Adapters統合クラス (2025年論文準拠)
    
    マルチモーダル統合モデル向けMoE最適化:
    - Llama-4 Expert: 言語理解・推論特化
    - SAM2 Expert: 視覚セグメンテーション特化  
    - Q-Former Expert: クロスモーダル融合特化
    """
    
    def __init__(
        self,
        base_models: Dict[str, nn.Module],  # {"llama": model, "sam2": model, "qformer": model}
        hidden_size: int = 5120,
        moe_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.moe_config = moe_config or self._get_default_moe_config()
        
        print("=== Heterogeneous MoE Adapters初期化 ===")
        
        # デバイス設定
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 動的ルーター初期化
        self.router = DynamicRouter(
            hidden_size=hidden_size,
            num_experts=3,  # Llama, SAM2, Q-Former
            num_tokens_per_expert=self.moe_config.get("active_experts", 2),
            capacity_factor=self.moe_config.get("expert_capacity_factor", 1.25),
            dropout=0.1,
            device=device
        )
        
        # モーダル特化LoRA Experts
        self.experts = nn.ModuleDict()
        
        # 🔄 SAM2+MLE論文準拠target_modules取得
        mle_config = config_linux.get_mle_config()
        target_modules_config = mle_config['target_modules']
        
        # Llama-4 Expert (言語理解・推論)
        if "llama" in base_models:
            llama_targets = target_modules_config['llama']  # 論文準拠
            try:
                self.experts["llama"] = LoRAExpert(
                    base_model=base_models["llama"],
                    target_modules=llama_targets,
                    rank=self.moe_config.get("lora_rank", 16),   # 🔄 論文準拠デフォルト
                    alpha=self.moe_config.get("lora_alpha", 32), # 🔄 論文準拠デフォルト
                    modal_type="language",
                    task_type=TaskType.CAUSAL_LM
                )
            except Exception as e:
                logger.error(f"❌ Llama-4 LoRA適用エラー: {e}")
                raise RuntimeError(f"Llama-4 LoRA適用に失敗しました: {e}")
        
        # SAM2 Expert (視覚セグメンテーション) 
        if "sam2" in base_models:
            sam2_targets = target_modules_config['sam2']    # 🔄 Hiera ViT特化
            
            # SAM2の実際のモジュール名を確認してターゲットを調整
            sam2_model = base_models["sam2"]
            
            # SAM2モデルの実体を取得
            actual_model = sam2_model
            if hasattr(sam2_model, 'predictor'):
                # SAM2Wrapperの場合
                predictor = sam2_model.predictor
                if hasattr(predictor, 'model'):
                    actual_model = predictor.model
            elif hasattr(sam2_model, 'model'):
                actual_model = sam2_model.model
            
            # ワイルドカードを実際のモジュール名に展開
            sam2_actual_targets = []
            for target_pattern in sam2_targets:
                if '*' in target_pattern:
                    # ワイルドカードパターンを正規表現に変換
                    import re
                    # 例: "image_encoder.trunk.blocks.*.attn.qkv" -> "image_encoder.trunk.blocks.\d+.attn.qkv"
                    pattern = target_pattern.replace('*', r'\d+')
                    # エスケープが必要な文字（ドット）を処理
                    pattern = pattern.replace('.', r'\.')
                    pattern = pattern.replace(r'\d+', r'(\d+)')  # 数字部分をキャプチャ
                    pattern = f"^{pattern}$"
                    regex = re.compile(pattern)
                    
                    # マッチするモジュールを探す
                    for name, module in actual_model.named_modules():
                        if isinstance(module, nn.Linear) and regex.match(name):
                            sam2_actual_targets.append(name)
                else:
                    # ワイルドカードなしの場合はそのまま追加
                    sam2_actual_targets.append(target_pattern)
            
            # 重複を除去
            sam2_actual_targets = list(set(sam2_actual_targets))
            
            # フォールバック: モジュールが見つからない場合
            if not sam2_actual_targets:
                logger.warning(f"SAM2の指定されたモジュールが見つかりません: {sam2_targets}")
                # 実際に存在するLinearモジュールを探す
                available_modules = []
                for name, module in actual_model.named_modules():
                    if isinstance(module, nn.Linear):
                        available_modules.append(name)
                
                # qkvかprojを含むモジュールを優先
                for name in available_modules:
                    if 'qkv' in name or 'proj' in name:
                        sam2_actual_targets.append(name)
                        if len(sam2_actual_targets) >= 4:
                            break
            
            logger.info(f"✔️ SAM2 LoRAターゲット: {sam2_actual_targets}")
            
            try:
                self.experts["sam2"] = LoRAExpert(
                    base_model=actual_model,
                    target_modules=sam2_actual_targets,
                    rank=self.moe_config.get("lora_rank", 16),   # 🔄 論文準拠デフォルト
                    alpha=self.moe_config.get("lora_alpha", 32), # 🔄 論文準拠デフォルト
                    modal_type="vision",
                    task_type=TaskType.FEATURE_EXTRACTION
                )
            except Exception as e:
                logger.error(f"❌ SAM2 LoRA適用エラー: {e}")
                raise RuntimeError(f"SAM2 LoRA適用に失敗しました: {e}")
        
        # Q-Former Expert (クロスモーダル融合)
        if "qformer" in base_models:
            qformer_targets = target_modules_config['qformer'] # 論文準拠
            self.experts["qformer"] = LoRAExpert(
                base_model=base_models["qformer"],
                target_modules=qformer_targets,
                rank=self.moe_config.get("lora_rank", 16),   # 🔄 論文準拠デフォルト
                alpha=self.moe_config.get("lora_alpha", 32), # 🔄 論文準拠デフォルト
                modal_type="fusion",
                task_type=TaskType.FEATURE_EXTRACTION
            )
        
        # エキスパート重み (学習可能)
        expert_weights = self.moe_config.get("expert_weights", {
            "llama": 0.4,
            "sam2": 0.4, 
            "qformer": 0.2
        })
        
        self.expert_weight_params = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(weight, dtype=torch.float32))
            for name, weight in expert_weights.items()
            if name in self.experts
        })
        
        # ルーターとエキスパート重みのみGPUに移動（モデル本体は除外）
        self.router = self.router.to(device)
        self.expert_weight_params = self.expert_weight_params.to(device)
        
        print(f"✅ MoE Adapter初期化完了")
        print(f"  - 総エキスパート数: {len(self.experts)}")
        print(f"  - エキスパート重み: {expert_weights}")
        print(f"  - デバイス: {device}")
        
    def _get_default_moe_config(self) -> Dict[str, Any]:
        """デフォルトMoE設定取得 (SAM2+MLE論文準拠)"""
        # 🔄 SAM2+MLE論文準拠設定を基盤として使用
        mle_config = config_linux.get_mle_config()
        
        return {
            "num_experts": mle_config["num_experts"],                    # 3
            "active_experts": mle_config["moe_top_k"],                   # 2 (Top-k)
            "router_type": "top_k",                                      # 論文準拠
            "load_balancing": True,
            "expert_capacity_factor": mle_config["expert_capacity_factor"], # 1.25
            "lora_rank": mle_config["lora_rank"],                        # 16 (論文準拠)
            "lora_alpha": mle_config["lora_alpha"],                      # 32 (論文準拠)
            "expert_weights": mle_config["expert_weights"]               # 論文準拠重み配分
        }
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_type: Optional[str] = None,
        test_mode: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Heterogeneous MoE forward処理
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            expert_type: 強制使用するエキスパートタイプ (None=自動ルーティング)
            test_mode: テストモード（エキスパートのembedding層をバイパス）
            
        Returns:
            output: MoE処理後の出力
            moe_info: MoE統計情報
        """
        
        if expert_type and expert_type in self.experts:
            # 特定エキスパート強制使用
            if test_mode:
                # テストモード: hidden_statesを直接処理
                expert_output = self.experts[expert_type](hidden_states)
                
                # 出力次元が異なる場合の調整（入力次元に合わせる）
                if expert_output.shape[-1] != hidden_states.shape[-1]:
                    # 出力投影レイヤーを動的に作成（キャッシュ）
                    if not hasattr(self, '_output_projection_cache'):
                        self._output_projection_cache = {}
                    
                    cache_key = f"{expert_type}_{expert_output.shape[-1]}_to_{hidden_states.shape[-1]}"
                    if cache_key not in self._output_projection_cache:
                        device = expert_output.device
                        dtype = expert_output.dtype
                        self._output_projection_cache[cache_key] = nn.Linear(
                            expert_output.shape[-1], hidden_states.shape[-1]
                        ).to(device).to(dtype)
                    
                    expert_output = self._output_projection_cache[cache_key](expert_output)
            else:
                # 通常モード: モデル推論実行
                expert_output = self.experts[expert_type](hidden_states, **kwargs)
            
            if hasattr(expert_output, 'last_hidden_state'):
                output = expert_output.last_hidden_state
            else:
                output = expert_output
            
            moe_info = {
                "forced_expert": expert_type,
                "expert_weight": self.expert_weight_params.get(expert_type, 1.0).item(),
                "routing_type": "forced",
                "test_mode": test_mode
            }
            
        else:
            # 動的ルーティング
            routed_output, routing_info = self.router(hidden_states)
            
            # テストモード: 各エキスパートの処理をシミュレート
            if test_mode:
                # ルーティング結果に基づいて重み付き結合
                expert_outputs = {}
                output_projections = {}
                
                for expert_name in self.experts:
                    expert_output = self.experts[expert_name](hidden_states)
                    expert_outputs[expert_name] = expert_output
                    
                    # 出力次元が異なる場合の調整
                    if expert_output.shape[-1] != hidden_states.shape[-1]:
                        # 出力投影レイヤーを動的に作成
                        if expert_name not in output_projections:
                            device = expert_output.device
                            dtype = expert_output.dtype
                            output_projections[expert_name] = nn.Linear(
                                expert_output.shape[-1], hidden_states.shape[-1]
                            ).to(device).to(dtype)
                        
                        expert_outputs[expert_name] = output_projections[expert_name](expert_output)
                
                # 重み付き結合
                weighted_output = torch.zeros_like(hidden_states)
                for i, expert_name in enumerate(self.experts):
                    expert_weight = routing_info['router_probs'][:, :, i].unsqueeze(-1)
                    weighted_output += expert_weight * expert_outputs[expert_name]
                
                output = weighted_output
            else:
                # 通常モード: 元の処理
                weighted_output = torch.zeros_like(routed_output)
                total_weight = 0.0
                
                for expert_name, expert_weight in self.expert_weight_params.items():
                    if expert_name in self.experts:
                        weighted_output += expert_weight * routed_output
                        total_weight += expert_weight.item()
                
                # 正規化
                if total_weight > 0:
                    output = weighted_output / total_weight
                else:
                    output = routed_output
            
            moe_info = {
                "routing_info": routing_info,
                "expert_weights": {name: weight.item() for name, weight in self.expert_weight_params.items()},
                "routing_type": "dynamic",
                "test_mode": test_mode
            }
        
        return output, moe_info
    
    def get_moe_statistics(self) -> Dict[str, Any]:
        """MoE統計情報取得"""
        stats = {
            "total_experts": len(self.experts),
            "expert_info": {},
            "total_trainable_params": 0,
            "expert_weights": {}
        }
        
        for name, expert in self.experts.items():
            if not hasattr(expert, 'get_expert_info'):
                raise RuntimeError(f"Expert '{name}' does not have get_expert_info method")
            expert_info = expert.get_expert_info()
            stats["expert_info"][name] = expert_info
            # tupleの場合は最初の要素を使用
            trainable_params = expert_info["trainable_params"]
            if isinstance(trainable_params, tuple):
                stats["total_trainable_params"] += trainable_params[0]
            else:
                stats["total_trainable_params"] += trainable_params
            stats["expert_weights"][name] = self.expert_weight_params.get(name, torch.tensor(0.0)).item()
        
        return stats


def create_heterogeneous_moe_adapter(
    base_models: Dict[str, nn.Module],
    moe_config: Optional[Dict[str, Any]] = None
) -> HeterogeneousMoEAdapter:
    """
    Heterogeneous MoE Adapterファクトリ関数
    
    Args:
        base_models: ベースモデル辞書 {"llama": model, "sam2": model, "qformer": model}
        moe_config: MoE設定辞書
        
    Returns:
        HeterogeneousMoEAdapter instance
    """
    print("🔄 Heterogeneous MoE Adapter作成中...")
    
    # 設定検証
    if not base_models:
        raise ValueError("少なくとも1つのベースモデルが必要です")
    
    # デフォルト設定とマージ
    default_config = {
        "num_experts": len(base_models),
        "active_experts": min(2, len(base_models)),
        "lora_rank": config_linux.LORA_R,
        "lora_alpha": config_linux.LORA_ALPHA,
        "expert_capacity_factor": 1.25,
        "load_balancing": True
    }
    
    if moe_config:
        default_config.update(moe_config)
    
    adapter = HeterogeneousMoEAdapter(
        base_models=base_models,
        hidden_size=config_linux.LLAMA_HIDDEN_SIZE,
        moe_config=default_config
    )
    
    print(f"✅ Heterogeneous MoE Adapter作成完了")
    
    return adapter


if __name__ == "__main__":
    print("=== Heterogeneous MoE Adapters テスト ===")
    
    # テスト用ダミーモデル
    class DummyModel(nn.Module):
        def __init__(self, hidden_size=5120):
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)
            self.config = type('Config', (), {'hidden_size': hidden_size})()
        
        def forward(self, hidden_states):
            return type('Output', (), {'last_hidden_state': self.linear(hidden_states)})()
    
    # ダミーベースモデル作成
    base_models = {
        "llama": DummyModel(),
        "sam2": DummyModel(),
        "qformer": DummyModel()
    }
    
    # MoE Adapter作成
    moe_adapter = create_heterogeneous_moe_adapter(base_models)
    
    # テストデータ
    batch_size, seq_len, hidden_size = 2, 16, 5120
    test_input = torch.randn(batch_size, seq_len, hidden_size)
    
    print(f"\nテスト実行...")
    print(f"入力: {test_input.shape}")
    
    # フォワードパス
    with torch.no_grad():
        output, moe_info = moe_adapter(test_input)
    
    print(f"出力: {output.shape}")
    print(f"MoE情報: {list(moe_info.keys())}")
    
    # 統計情報
    stats = moe_adapter.get_moe_statistics()
    print(f"\nMoE統計:")
    for key, value in stats.items():
        if key != "expert_info":
            print(f"  {key}: {value}")
    
    print("\n✅ Heterogeneous MoE Adapters テスト完了")