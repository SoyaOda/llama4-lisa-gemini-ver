# model/metap_optimizer.py
"""
MetaP (Meta Parameter tuning) Optimizer for Llama-4 + SAM2 + Q-Former

Phase 3C実装: MetaOptimize 2024論文準拠のハイパーパラメータ最適化
- Gradient-based hyperparameter optimization with meta-learning
- 長期的効果考慮（割引和による将来損失への影響評価）
- Phase 3B実証済みベースライン設定活用

目標:
- 学習効率: 5倍高速化
- LoRA rank/alpha、学習率、fusion weightsの同時最適化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MetaPConfig:
    """MetaP最適化設定"""
    # Phase 3B実証済みベースライン
    baseline_lora_rank: int = 16      # SAM2+MLE論文準拠
    baseline_lora_alpha: int = 32     # SAM2+MLE論文準拠
    baseline_lr: float = 1e-4         # config_linux実証値
    
    # MetaP最適化パラメータ範囲
    lora_rank_range: Tuple[int, int] = (8, 32)
    lora_alpha_range: Tuple[int, int] = (16, 64)
    lr_multiplier_range: Tuple[float, float] = (0.1, 10.0)
    
    # 融合重み初期値（Phase 3B実証値）
    initial_fusion_weights: Dict[str, float] = None
    
    # MetaP学習設定
    meta_lr: float = 0.01             # メタパラメータ学習率
    discount_factor: float = 0.95     # 将来損失割引率
    meta_batch_size: int = 4          # メタ最適化バッチサイズ
    
    def __post_init__(self):
        if self.initial_fusion_weights is None:
            self.initial_fusion_weights = {
                'llama_weight': 0.4,      # 言語理解・推論
                'sam2_weight': 0.4,       # 視覚セグメンテーション
                'qformer_weight': 0.2     # クロスモーダル融合
            }


class LlamaMetaPOptimizer(nn.Module):
    """
    Llama-4+SAM2+Q-Former特化MetaPハイパーパラメータ最適化器
    
    MetaOptimize 2024論文準拠：
    - 即時損失ではなく長期的効果を考慮
    - 勾配ベースのメタ学習
    - 制約付き最適化（実用的範囲内）
    """
    
    def __init__(self, base_model: nn.Module, config: MetaPConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Meta-parameters（学習可能パラメータ）
        self.meta_lora_rank = nn.Parameter(
            torch.tensor(float(config.baseline_lora_rank))
        )
        self.meta_lora_alpha = nn.Parameter(
            torch.tensor(float(config.baseline_lora_alpha))
        )
        self.meta_lr_multiplier = nn.Parameter(
            torch.tensor(1.0)
        )
        
        # 融合重みメタパラメータ（3次元：llama, sam2, qformer）
        fusion_weights_tensor = torch.tensor([
            config.initial_fusion_weights['llama_weight'],
            config.initial_fusion_weights['sam2_weight'],
            config.initial_fusion_weights['qformer_weight']
        ])
        self.meta_fusion_weights = nn.Parameter(fusion_weights_tensor)
        
        # Meta-optimizer（メタパラメータ用Adam）
        self.meta_parameters = [
            self.meta_lora_rank,
            self.meta_lora_alpha,
            self.meta_lr_multiplier,
            self.meta_fusion_weights
        ]
        
        self.meta_optimizer = torch.optim.Adam(
            self.meta_parameters,
            lr=config.meta_lr
        )
        
        # 履歴追跡（長期的効果評価用）
        self.loss_history = []
        self.param_history = []
        
        logger.info("LlamaMetaPOptimizer初期化完了")
        logger.info(f"  - ベースラインLoRA: r={config.baseline_lora_rank}, α={config.baseline_lora_alpha}")
        logger.info(f"  - メタ学習率: {config.meta_lr}")
        logger.info(f"  - 割引率: {config.discount_factor}")
    
    def compute_meta_gradient(
        self, 
        train_loss: torch.Tensor, 
        val_loss: torch.Tensor,
        future_losses: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        """
        メタ勾配計算（MetaOptimize 2024論文準拠）
        
        Args:
            train_loss: 訓練損失
            val_loss: 検証損失
            future_losses: 将来損失リスト（長期的効果評価用）
        
        Returns:
            メタパラメータの勾配リスト
        """
        # 基本的なメタ勾配（検証損失に対する勾配）
        if val_loss.requires_grad:
            base_meta_grads = torch.autograd.grad(
                val_loss,
                self.meta_parameters,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )
        else:
            # 検証損失が勾配を持たない場合、訓練損失を使用
            base_meta_grads = torch.autograd.grad(
                train_loss,
                self.meta_parameters,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )
        
        # 長期的効果を考慮（MetaOptimize特徴）
        if future_losses:
            discount = 1.0
            total_meta_grads = [g.clone() if g is not None else torch.zeros_like(p) 
                               for g, p in zip(base_meta_grads, self.meta_parameters)]
            
            for future_loss in future_losses:
                discount *= self.config.discount_factor
                if future_loss.requires_grad:
                    future_grads = torch.autograd.grad(
                        future_loss * discount,
                        self.meta_parameters,
                        create_graph=False,
                        retain_graph=True,
                        allow_unused=True
                    )
                    
                    for i, (g, fg) in enumerate(zip(total_meta_grads, future_grads)):
                        if fg is not None:
                            total_meta_grads[i] = g + fg
            
            return total_meta_grads
        else:
            return [g if g is not None else torch.zeros_like(p) 
                   for g, p in zip(base_meta_grads, self.meta_parameters)]
    
    def optimize_hyperparameters(
        self, 
        train_loss: torch.Tensor, 
        val_loss: torch.Tensor,
        future_losses: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        メタパラメータ最適化ステップ
        
        Args:
            train_loss: 訓練損失
            val_loss: 検証損失
            future_losses: 将来損失リスト
        
        Returns:
            最適化済み設定辞書
        """
        # メタ勾配計算
        meta_grads = self.compute_meta_gradient(train_loss, val_loss, future_losses)
        
        # 勾配を手動で設定
        for param, grad in zip(self.meta_parameters, meta_grads):
            if grad is not None:
                param.grad = grad
        
        # メタ最適化ステップ
        self.meta_optimizer.step()
        
        # 制約付き更新（実用的な範囲内に制限）
        with torch.no_grad():
            # LoRA rank制約
            self.meta_lora_rank.data = torch.clamp(
                self.meta_lora_rank.data,
                float(self.config.lora_rank_range[0]),
                float(self.config.lora_rank_range[1])
            )
            
            # LoRA alpha制約
            self.meta_lora_alpha.data = torch.clamp(
                self.meta_lora_alpha.data,
                float(self.config.lora_alpha_range[0]),
                float(self.config.lora_alpha_range[1])
            )
            
            # 学習率倍率制約
            self.meta_lr_multiplier.data = torch.clamp(
                self.meta_lr_multiplier.data,
                self.config.lr_multiplier_range[0],
                self.config.lr_multiplier_range[1]
            )
            
            # 融合重みソフトマックス正規化
            self.meta_fusion_weights.data = F.softmax(self.meta_fusion_weights.data, dim=0)
        
        # 最適化済み設定を返す
        optimized_config = self.get_optimized_config()
        
        # 履歴更新
        self.loss_history.append({
            'train_loss': train_loss.item(),
            'val_loss': val_loss.item()
        })
        self.param_history.append(optimized_config.copy())
        
        # ログ出力
        logger.debug(f"MetaP最適化ステップ完了:")
        logger.debug(f"  - LoRA rank: {optimized_config['lora_rank']}")
        logger.debug(f"  - LoRA alpha: {optimized_config['lora_alpha']}")
        logger.debug(f"  - 学習率: {optimized_config['learning_rate']:.6f}")
        
        return optimized_config
    
    def get_optimized_config(self) -> Dict[str, Any]:
        """最適化済み設定取得"""
        return {
            'lora_rank': int(torch.round(self.meta_lora_rank).item()),
            'lora_alpha': int(torch.round(self.meta_lora_alpha).item()),
            'learning_rate': self.config.baseline_lr * self.meta_lr_multiplier.item(),
            'lr_multiplier': self.meta_lr_multiplier.item(),
            'fusion_weights': {
                'llama_weight': self.meta_fusion_weights[0].item(),
                'sam2_weight': self.meta_fusion_weights[1].item(),
                'qformer_weight': self.meta_fusion_weights[2].item()
            }
        }
    
    def estimate_future_performance(self, current_loss: float) -> float:
        """
        将来性能推定（MetaOptimize特徴）
        過去の履歴から将来の損失減少を予測
        """
        if len(self.loss_history) < 2:
            return current_loss
        
        # 簡易的な線形外挿
        recent_losses = [h['val_loss'] for h in self.loss_history[-5:]]
        if len(recent_losses) >= 2:
            trend = (recent_losses[-1] - recent_losses[0]) / len(recent_losses)
            future_estimate = current_loss + trend * self.config.meta_batch_size
            return max(0.0, future_estimate)  # 負の損失を防ぐ
        
        return current_loss
    
    def reset_history(self):
        """履歴リセット（新しい学習フェーズ開始時）"""
        self.loss_history.clear()
        self.param_history.clear()
        logger.info("MetaP履歴リセット完了")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """性能サマリ取得"""
        if not self.loss_history:
            return {}
        
        initial_loss = self.loss_history[0]['val_loss']
        final_loss = self.loss_history[-1]['val_loss']
        improvement = (initial_loss - final_loss) / initial_loss * 100
        
        return {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'improvement_percentage': improvement,
            'total_steps': len(self.loss_history),
            'current_config': self.get_optimized_config()
        }


class DynamicConfigAdapter:
    """動的設定適用器（MetaP最適化設定をモデルに適用）"""
    
    def __init__(self):
        self.applied_configs = []
        logger.info("DynamicConfigAdapter初期化完了")
    
    def apply_config(self, model: nn.Module, config: Dict[str, Any]) -> bool:
        """
        MetaP最適化設定をモデルに適用
        
        Args:
            model: 対象モデル
            config: MetaP最適化設定
        
        Returns:
            適用成功フラグ
        """
        try:
            # LoRA設定の動的適用
            if hasattr(model, 'peft_config'):
                # 既存のLoRA設定を更新
                model.peft_config.r = config['lora_rank']
                model.peft_config.lora_alpha = config['lora_alpha']
                logger.info(f"LoRA設定更新: r={config['lora_rank']}, α={config['lora_alpha']}")
            
            # 学習率の動的適用
            if hasattr(model, 'optimizer') and model.optimizer is not None:
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] = config['learning_rate']
                logger.info(f"学習率更新: {config['learning_rate']:.6f}")
            
            # 融合重みの動的適用
            if 'fusion_weights' in config:
                if hasattr(model, 'fusion_weights'):
                    for key, value in config['fusion_weights'].items():
                        if hasattr(model.fusion_weights, key):
                            setattr(model.fusion_weights, key, value)
                elif hasattr(model, 'dual_decoder') and hasattr(model.dual_decoder, 'fusion_weight'):
                    # Phase 3Bデュアルデコーダーの融合重み更新
                    model.dual_decoder.fusion_weight.data = torch.tensor([
                        config['fusion_weights']['llama_weight'],
                        config['fusion_weights']['sam2_weight']
                    ])
                
                logger.info(f"融合重み更新: {config['fusion_weights']}")
            
            # 履歴保存
            self.applied_configs.append(config.copy())
            
            return True
            
        except Exception as e:
            logger.error(f"設定適用エラー: {e}")
            return False
    
    def revert_to_baseline(self, model: nn.Module, baseline_config: Dict[str, Any]):
        """ベースライン設定に戻す"""
        return self.apply_config(model, baseline_config)


def create_metap_optimizer(
    base_model: nn.Module,
    config_override: Optional[Dict[str, Any]] = None
) -> LlamaMetaPOptimizer:
    """
    MetaP最適化器作成ヘルパー関数
    
    Args:
        base_model: ベースモデル
        config_override: 設定オーバーライド
    
    Returns:
        LlamaMetaPOptimizer インスタンス
    """
    config = MetaPConfig()
    
    if config_override:
        for key, value in config_override.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return LlamaMetaPOptimizer(base_model, config)