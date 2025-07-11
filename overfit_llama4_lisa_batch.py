#!/usr/bin/env python3
"""
LISA-Llama4統合モデル 学習能力検証スクリプト
overfit_llama4_single_batch.pyを参考に、LISA統合モデル特有の機能もテスト

実行方法:
Lambda Cloud (129.213.148.184):
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='verification_output' --exclude='vis_output' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@129.213.148.184:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

ssh -i ~/.ssh/lambda_cloud_key ubuntu@129.213.148.184 "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 python -u overfit_llama4_lisa_batch.py 2>&1"
"""

import sys
import os
import gc
import json
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # バックエンドを非対話モードに設定
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# PEFT関連
from peft import LoraConfig, get_peft_model, TaskType

# プロジェクト固有のインポート
sys.path.append('.')
from model.llama4_lisa import LisaLlama4ForCausalLM, LisaLlama4Config
import config_linux

# ロギング設定
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LisaOverfitConfig:
    """LISA統合モデル過学習テスト用設定（config_linux統一設定使用）"""
    # 学習固有設定
    max_length: int = 256
    num_epochs: int = 10
    
    # 過学習判定設定
    target_loss: float = 0.1
    
    # 出力設定
    output_dir: str = "./llama4_lisa_overfit_results"
    
    def __post_init__(self):
        """config_linux統一設定を適用（デバッグ情報付き）"""
        # LISA統合モデル設定
        lisa_config = config_linux.get_lisa_model_config()
        print(f"📋 取得したlisa_config: {lisa_config}")
        
        self.llama_model_id = lisa_config["llama_model_id"]
        self.sam_checkpoint_path = lisa_config["sam_checkpoint_path"]
        self.attn_implementation = lisa_config["attn_implementation"]
        self.torch_dtype = lisa_config["torch_dtype"]
        print(f"🔧 適用されたtorch_dtype: {self.torch_dtype}")
        
        # LoRA設定
        lora_config = config_linux.get_lora_config()
        self.lora_r = lora_config["r"]
        self.lora_alpha = lora_config["lora_alpha"]
        self.lora_dropout = lora_config["lora_dropout"]
        self.lora_target_modules = lora_config["target_modules"]
        
        # 学習設定
        training_config = config_linux.get_training_config()
        self.learning_rate = training_config["learning_rate"]

class TensorJSONEncoder(json.JSONEncoder):
    """Tensorオブジェクト用のJSONエンコーダー"""
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return {
                "__tensor__": True,
                "data": obj.detach().cpu().tolist() if obj.numel() <= 100 else f"<Tensor shape={obj.shape}>",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device)
            }
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return super().default(obj)

class LisaOverfitTest:
    """LISA統合モデル過学習テストクラス"""
    
    def __init__(self, config: LisaOverfitConfig):
        self.config = config
        self.results = {
            "training_logs": [],
            "loss_history": [],
            "success_metrics": {},
            "config": config.__dict__
        }
        
        # 出力ディレクトリ作成
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 固定データ
        self.fixed_data = None
        
    def setup_model(self) -> Any:
        """LISA統合モデル初期化"""
        logger.info("=== LISA-Llama4統合モデル初期化 ===")
        
        try:
            # 動的コンパイルを無効化してGPU分散エラーを回避
            torch.compiler.disable()
            logger.info("動的コンパイル無効化: GPU分散エラー回避のため")
            
            # LISA統合モデル設定（config_linux統一設定を使用）
            lisa_config = LisaLlama4Config(**config_linux.get_lisa_model_config())
            
            # LISA統合モデル初期化
            model = LisaLlama4ForCausalLM(lisa_config)
            
            logger.info(f"✓ LISA統合モデル初期化完了")
            logger.info(f"  - 総パラメータ: {sum(p.numel() for p in model.parameters()):,}")
            
            return model
            
        except Exception as e:
            logger.error(f"LISA統合モデル初期化エラー: {e}")
            raise
    
    def apply_lora_config(self, model) -> Any:
        """Web調査結果に基づくLoRA設定適用（device_map preservation対応）"""
        logger.info("=== LoRA設定適用 ===")
        
        try:
            # PEFT適用前にdevice_mapを保存（Web調査：既知の問題対策）
            original_device_map = None
            original_device_map_location = None
            
            # device_mapの場所を特定して保存
            if hasattr(model, 'hf_device_map') and model.hf_device_map:
                original_device_map = model.hf_device_map.copy()
                original_device_map_location = "direct"
                logger.info(f"✓ 元のdevice_map保存（直接アクセス）: {len(original_device_map)} エントリ")
            elif hasattr(model, 'llama_model') and hasattr(model.llama_model, 'hf_device_map') and model.llama_model.hf_device_map:
                original_device_map = model.llama_model.hf_device_map.copy()
                original_device_map_location = "llama_model"
                logger.info(f"✓ 元のdevice_map保存（llama_model経由）: {len(original_device_map)} エントリ")
            else:
                logger.warning("⚠️ device_mapが見つかりません。Model Parallelismが未設定の可能性があります。")
            
            # LoRA設定作成（config_linux統一設定を使用）
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                **config_linux.get_lora_config()
            )
            
            logger.info(f"LoRA設定:")
            logger.info(f"  - rank: {self.config.lora_r}")
            logger.info(f"  - alpha: {self.config.lora_alpha}")
            logger.info(f"  - dropout: {self.config.lora_dropout}")
            logger.info(f"  - target_modules: {self.config.lora_target_modules}")
            
            # LoRA適用
            model = get_peft_model(model, lora_config)
            
            # device_mapの復元試行（Web調査：PEFT既知問題の対策）
            if original_device_map and original_device_map_location:
                # 複数のアクセス方法を試行
                restoration_success = False
                
                # 方法1: 直接アクセス確認
                if hasattr(model, 'hf_device_map') and model.hf_device_map:
                    logger.info("✓ 直接device_mapアクセス確認済み")
                    restoration_success = True
                
                # 方法2: base_model経由のアクセス
                elif hasattr(model, 'base_model') and hasattr(model.base_model, 'hf_device_map') and model.base_model.hf_device_map:
                    logger.info("✓ base_model経由device_mapアクセス確認済み")
                    restoration_success = True
                
                # 方法3: llama_model経由のアクセス（LISA統合モデル特有）
                elif hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model') and hasattr(model.base_model.llama_model, 'hf_device_map') and model.base_model.llama_model.hf_device_map:
                    logger.info("✓ base_model.llama_model経由device_mapアクセス確認済み")
                    restoration_success = True
                
                # 方法4: 手動復元
                if not restoration_success:
                    logger.warning("⚠️ device_mapが失われました。手動復元を試行...")
                    
                    # 元の場所に基づいて復元
                    if original_device_map_location == "llama_model":
                        if hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model'):
                            model.base_model.llama_model.hf_device_map = original_device_map
                            logger.info("✓ base_model.llama_model.hf_device_mapを手動復元")
                            restoration_success = True
                    elif original_device_map_location == "direct":
                        if hasattr(model, 'base_model'):
                            model.base_model.hf_device_map = original_device_map
                            logger.info("✓ base_model.hf_device_mapを手動復元")
                            restoration_success = True
                
                if not restoration_success:
                    logger.error("❌ device_mapの復元に失敗しました")
            
            # 学習可能パラメータ統計
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            
            logger.info(f"✓ LoRA適用完了")
            logger.info(f"  - 学習可能パラメータ: {trainable_params:,}")
            logger.info(f"  - 全パラメータ: {total_params:,}")
            logger.info(f"  - 学習可能割合: {100 * trainable_params / total_params:.3f}%")
            
            # 最終device_map確認
            self.verify_device_map_after_lora(model)
            
            return model
            
        except Exception as e:
            logger.error(f"LoRA適用エラー: {e}")
            raise
    
    def verify_device_map_after_lora(self, model):
        """LoRA適用後のdevice_map確認（LISA統合モデル対応）"""
        logger.info("=== LoRA適用後device_map確認 ===")
        
        device_map = None
        access_path = None
        
        # 複数のアクセス方法を試行
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            device_map = model.hf_device_map
            access_path = "直接アクセス"
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'hf_device_map') and model.base_model.hf_device_map:
            device_map = model.base_model.hf_device_map
            access_path = "base_model経由"
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model') and hasattr(model.base_model.llama_model, 'hf_device_map') and model.base_model.llama_model.hf_device_map:
            device_map = model.base_model.llama_model.hf_device_map
            access_path = "base_model.llama_model経由"
        
        if device_map:
            logger.info(f"✓ {access_path}でhf_device_mapアクセス成功")
            logger.info(f"  - デバイスマップ: {dict(list(device_map.items())[:5])}...")
            logger.info(f"  - 使用GPU数: {len(set(device_map.values()))}")
            return True
        else:
            logger.error("❌ device_mapが見つかりません")
            logger.error("Model Parallelismが設定されていません。103Bモデルには必須です。")
            return False
    
    def prepare_fixed_data(self, model) -> Dict[str, Any]:
        """固定データ準備（公式推奨方法）"""
        logger.info("=== 固定データ準備 ===")
        
        try:
            # テスト画像作成
            test_image = Image.new('RGB', (224, 224), color='red')
            test_prompt = "この画像で赤い領域を[SEG]してください。"
            
            # 公式推奨方法：メッセージ形式で準備（勾配フロー検証で成功した方法）
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": test_image},
                        {"type": "text", "text": test_prompt}
                    ]
                }
            ]
            
            # Llama4 Processorで直接処理（成功した方法）
            inputs = model.llama_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # BatchFeatureを辞書に変換（勾配フロー検証で成功した方法）
            if hasattr(inputs, 'keys') and hasattr(inputs, '__getitem__'):
                result_dict = {}
                for key in inputs.keys():
                    result_dict[key] = inputs[key]
                inputs = result_dict
                logger.info(f"✅ BatchFeature→dict変換完了: {list(inputs.keys())}")
            
            self.fixed_data = {
                "inputs": inputs,
                "test_image": test_image,
                "test_prompt": test_prompt
            }
            
            logger.info(f"✓ 固定データ準備完了")
            logger.info(f"  - 入力形状: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in inputs.items()]}")
            
            return self.fixed_data
            
        except Exception as e:
            logger.error(f"固定データ準備エラー: {e}")
            raise
    
    def setup_optimizer(self, model) -> torch.optim.Optimizer:
        """オプティマイザー設定（成功した単独モデルと同じ）"""
        logger.info("=== オプティマイザー設定 ===")
        
        # 学習可能パラメータのみ対象
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        logger.info(f"✓ AdamWオプティマイザー設定完了")
        logger.info(f"  - 学習率: {self.config.learning_rate}")
        logger.info(f"  - 学習可能パラメータ数: {len(trainable_params)}")
        
        return optimizer
    
    def run_training_epoch(self, model, optimizer, epoch: int) -> float:
        """単一エポックの学習実行（勾配フロー検証で成功した方法）"""
        model.train()
        optimizer.zero_grad()
        
        # 固定データを適切なデバイスに移動（Web調査：フォールバック処理を削除）
        inputs = self.fixed_data["inputs"]
        
        # Model Parallelismのdevice_mapを厳密にチェック（LISA統合モデル対応）
        device_map = None
        access_path = None
        
        # 複数のアクセス方法を試行
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            device_map = model.hf_device_map
            access_path = "直接アクセス"
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'hf_device_map') and model.base_model.hf_device_map:
            device_map = model.base_model.hf_device_map
            access_path = "base_model経由"
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model') and hasattr(model.base_model.llama_model, 'hf_device_map') and model.base_model.llama_model.hf_device_map:
            device_map = model.base_model.llama_model.hf_device_map
            access_path = "base_model.llama_model経由"
        
        if not device_map:
            raise RuntimeError("Model Parallelismが設定されていません。103Bモデルには必須です。")
        
        logger.info(f"Model Parallelismデバイスマップ確認: {access_path}")
        first_device = next(iter(device_map.values()))
        inputs = {k: v.to(first_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # 実際のLISA統合モデル使用：完全なフォワードパス（SAM機能付き）
        # verify_llama4_lisa_gradients.pyで成功した実装と同じ方法
        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            pixel_values=inputs.get('pixel_values'),
            labels=inputs['input_ids'],  # 言語モデリング用
            generate_mask=True  # SAM機能を有効化
        )
        
        # CompositeLoss統合による損失取得
        if isinstance(model_outputs, dict):
            # CompositeLossからの統一損失
            if 'text_loss' in model_outputs:
                loss = model_outputs['text_loss']
            # 予備処理：lossキーも確認
            elif 'loss' in model_outputs:
                loss = model_outputs['loss']
            # フォールバック：手動計算
            else:
                logits = model_outputs.get('logits')
                if logits is None:
                    raise ValueError("logitsが見つかりません")
                
                # 言語モデリング損失を手動計算
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )
        else:
            # 非辞書型出力の場合
            if hasattr(model_outputs, 'loss'):
                loss = model_outputs.loss
            else:
                raise ValueError("損失が見つかりません")
        
        # 逆伝播
        loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # パラメータ更新
        optimizer.step()
        
        # 損失値取得
        loss_value = loss.item()
        
        # ログ記録
        log_entry = {
            "epoch": epoch,
            "loss": loss_value,
            "learning_rate": self.config.learning_rate
        }
        
        self.results["training_logs"].append(log_entry)
        self.results["loss_history"].append(loss_value)
        
        return loss_value
    
    def run_overfit_test(self, model) -> Dict[str, Any]:
        """過学習テスト実行（成功した単独モデルと同じロジック）"""
        logger.info("=== LISA統合モデル過学習テスト開始 ===")
        
        # オプティマイザー設定
        optimizer = self.setup_optimizer(model)
        
        # 学習ループ
        initial_loss = None
        final_loss = None
        
        for epoch in range(self.config.num_epochs):
            loss = self.run_training_epoch(model, optimizer, epoch + 1)
            
            if epoch == 0:
                initial_loss = loss
            final_loss = loss
            
            logger.info(f"エポック {epoch + 1}/{self.config.num_epochs}: 損失 = {loss:.6f}")
            
            # 早期停止判定
            if loss < self.config.target_loss:
                logger.info(f"✓ 目標損失{self.config.target_loss}を達成！エポック{epoch + 1}で早期停止")
                break
            
            # メモリクリーンアップ
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        # 過学習成功判定
        loss_reduction = initial_loss - final_loss if initial_loss else 0
        loss_reduction_ratio = loss_reduction / initial_loss if initial_loss else 0
        
        overfit_success = (
            final_loss < self.config.target_loss or
            loss_reduction_ratio > 0.5  # 50%以上損失が減少
        )
        
        success_metrics = {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction": loss_reduction,
            "loss_reduction_ratio": loss_reduction_ratio,
            "target_loss_achieved": final_loss < self.config.target_loss,
            "significant_improvement": loss_reduction_ratio > 0.5,
            "overfit_success": overfit_success
        }
        
        self.results["success_metrics"] = success_metrics
        
        logger.info("=== LISA統合モデル過学習テスト完了 ===")
        logger.info(f"  - 初期損失: {initial_loss:.6f}")
        logger.info(f"  - 最終損失: {final_loss:.6f}")
        logger.info(f"  - 損失減少: {loss_reduction:.6f} ({loss_reduction_ratio:.1%})")
        logger.info(f"  - 過学習成功: {'✓' if overfit_success else '✗'}")
        
        return success_metrics
    
    def create_loss_plot(self) -> str:
        """損失曲線プロット作成"""
        if not self.results["loss_history"]:
            return ""
        
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.results["loss_history"]) + 1), 
                    self.results["loss_history"], 
                    'b-', linewidth=2, label='Training Loss')
            
            # 目標損失線
            plt.axhline(y=self.config.target_loss, color='r', linestyle='--', 
                       label=f'Target Loss ({self.config.target_loss})')
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('LISA-Llama4統合モデル Overfit Test: Loss Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 最小値にマーク
            min_loss_epoch = np.argmin(self.results["loss_history"]) + 1
            min_loss_value = min(self.results["loss_history"])
            plt.plot(min_loss_epoch, min_loss_value, 'ro', markersize=8, 
                    label=f'Min Loss: {min_loss_value:.4f}')
            plt.legend()
            
            plot_file = Path(self.config.output_dir) / "lisa_loss_curve.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ 損失曲線プロット保存: {plot_file}")
            return str(plot_file)
            
        except Exception as e:
            logger.error(f"プロット作成エラー: {e}")
            return ""
    
    def save_results(self) -> str:
        """結果保存"""
        output_file = Path(self.config.output_dir) / "llama4_lisa_overfit_test_results.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, cls=TensorJSONEncoder)
            
            logger.info(f"✓ テスト結果を保存: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"結果保存エラー: {e}")
            return ""
    
    def run_test(self) -> Dict[str, Any]:
        """テスト実行"""
        logger.info("🚀 LISA-Llama4統合モデル 学習能力検証開始")
        logger.info(f"設定: {self.config}")
        
        try:
            # 1. LISA統合モデル初期化
            model = self.setup_model()
            
            # 2. LoRA適用
            model = self.apply_lora_config(model)
            
            # 3. 固定データ準備
            self.prepare_fixed_data(model)
            
            # 4. 過学習テスト実行
            success_metrics = self.run_overfit_test(model)
            
            # 5. 損失曲線プロット作成
            plot_file = self.create_loss_plot()
            if plot_file:
                self.results["plot_file"] = plot_file
            
            # 6. 結果保存
            results_file = self.save_results()
            if results_file:
                self.results["results_file"] = results_file
            
            # 最終評価
            if success_metrics["overfit_success"]:
                logger.info("🎉 LISA統合モデル学習能力検証成功！")
            else:
                logger.warning("⚠️ LISA統合モデル学習能力検証で問題が発生")
            
            return self.results
            
        except Exception as e:
            logger.error(f"❌ テスト実行エラー: {e}")
            raise

def main():
    """メイン関数"""
    try:
        # 設定
        config = LisaOverfitConfig()
        
        # テスト実行
        test = LisaOverfitTest(config)
        results = test.run_test()
        
        # 結果表示
        logger.info("=== 最終結果 ===")
        success_metrics = results.get("success_metrics", {})
        logger.info(f"過学習成功: {success_metrics.get('overfit_success', False)}")
        logger.info(f"初期損失: {success_metrics.get('initial_loss', 'N/A')}")
        logger.info(f"最終損失: {success_metrics.get('final_loss', 'N/A')}")
        logger.info(f"損失減少率: {success_metrics.get('loss_reduction_ratio', 0):.1%}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ メイン実行エラー: {e}")
        raise

if __name__ == "__main__":
    main() 