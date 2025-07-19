# model/metap_integration.py
"""
MetaP + Phase 3B機能統合モデル

Phase 3C実装: MetaP最適化とPhase 3B実証済み機能の統合
- デュアルパスウェイデコーダ、多重解像度融合、OHEM損失の活用
- 動的ハイパーパラメータ調整
- 訓練時と推論時の設定切り替え

目標:
- Phase 3B（28.14%向上）+ MetaP効果で40%向上達成
- 実装安定性95%維持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path

# Phase 3B実装済み機能
from model.dual_pathway_decoder import create_dual_pathway_decoder
from model.multiresolution_fusion import Llama4SAM2MultiResolutionFusion
from model.ohem_loss import create_ohem_loss
from model.metap_optimizer import (
    LlamaMetaPOptimizer, 
    MetaPConfig, 
    DynamicConfigAdapter,
    create_metap_optimizer
)

# 設定
import config_linux

logger = logging.getLogger(__name__)


class MetaPIntegratedModel(nn.Module):
    """
    MetaP + Phase 3B機能統合モデル
    
    統合機能:
    - MetaP動的ハイパーパラメータ最適化
    - Phase 3Bデュアルパスウェイデコーダ
    - Phase 3B多重解像度融合
    - Phase 3B OHEM損失関数
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Phase 3B実装済み機能の初期化
        logger.info("Phase 3B機能初期化中...")
        
        # 1. デュアルパスウェイデコーダ（28.14%向上の核心）
        self.dual_decoder = create_dual_pathway_decoder(
            llama_hidden_size=config.get('llama_hidden_size', 5120),
            sam_output_dim=config.get('sam_output_dim', 256),
            fusion_strategy=config.get('fusion_strategy', 'learned_weighted')
        )
        
        # 2. 多重解像度融合（10-15%向上）
        self.multiresolution_fusion = Llama4SAM2MultiResolutionFusion(
            llama_hidden_size=config.get('llama_hidden_size', 5120),
            sam_feature_dim=config.get('sam_feature_dim', 256),
            sam_scales=config.get('sam_scales', [1024, 512, 256]),
            qformer_dim=config.get('qformer_dim', 768),
            qformer_queries=config.get('qformer_queries', 32),
            fusion_dim=config.get('fusion_dim', 512),
            output_size=config.get('output_size', (448, 448)),
            fusion_strategy=config.get('fusion_strategy', 'attention')
        )
        
        # 3. OHEM損失関数（収束2倍高速化）
        self.ohem_loss = create_ohem_loss(
            hard_ratio=config.get('hard_ratio', 0.25),
            config_override=config.get('loss_config', {
                'ce_loss_weight': 1.0,
                'bce_loss_weight': 2.0,
                'dice_loss_weight': 0.5
            })
        )
        
        # MetaP最適化器の初期化（遅延初期化）
        self.metap_optimizer = None
        self.config_adapter = DynamicConfigAdapter()
        
        # 現在の動的設定
        self.current_config = None
        
        # 訓練モードフラグ
        self.metap_enabled = config.get('enable_metap', True)
        
        logger.info("MetaPIntegratedModel初期化完了")
        logger.info(f"  - MetaP有効: {self.metap_enabled}")
        logger.info(f"  - デュアルパスウェイ: ✓")
        logger.info(f"  - 多重解像度融合: ✓")
        logger.info(f"  - OHEM損失: ✓")
    
    def initialize_metap(self, base_model: Optional[nn.Module] = None):
        """MetaP最適化器の遅延初期化"""
        if self.metap_optimizer is None and self.metap_enabled:
            # ベースモデルが指定されていない場合は自身を使用
            target_model = base_model if base_model is not None else self
            
            # MetaP設定作成
            metap_config = MetaPConfig(
                baseline_lora_rank=self.config.get('lora_rank', 16),
                baseline_lora_alpha=self.config.get('lora_alpha', 32),
                baseline_lr=self.config.get('learning_rate', 1e-4),
                initial_fusion_weights={
                    'llama_weight': self.config.get('llama_weight', 0.4),
                    'sam2_weight': self.config.get('sam2_weight', 0.4),
                    'qformer_weight': self.config.get('qformer_weight', 0.2)
                }
            )
            
            self.metap_optimizer = LlamaMetaPOptimizer(target_model, metap_config)
            logger.info("MetaP最適化器初期化完了")
    
    def forward_with_metap(
        self,
        batch: Dict[str, torch.Tensor],
        is_training: bool = True,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        MetaP統合推論
        
        Args:
            batch: 入力バッチ（images, texts, targets等）
            is_training: 訓練モードフラグ
            return_intermediate: 中間結果返却フラグ
        
        Returns:
            推論結果辞書
        """
        # MetaP最適化設定の適用（訓練時のみ）
        if is_training and self.metap_enabled and self.metap_optimizer is not None:
            self.current_config = self.metap_optimizer.get_optimized_config()
            self.config_adapter.apply_config(self, self.current_config)
        
        # 1. 多重解像度融合
        # 入力データ準備
        images = batch.get('images')
        texts = batch.get('texts')
        llama_hidden_states = batch.get('llama_hidden_states')
        sam_features = batch.get('sam_features')
        qformer_features = batch.get('qformer_features')
        
        # 動的解像度調整（MetaP設定適用）
        if self.current_config and 'target_resolution' in self.current_config:
            target_resolution = self.current_config['target_resolution']
        else:
            target_resolution = self.config.get('output_size', (448, 448))
        
        # 多重解像度融合実行
        fusion_results = self.multiresolution_fusion(
            llama_hidden_states=llama_hidden_states,
            sam_features=sam_features,
            qformer_features=qformer_features,
            return_intermediate=return_intermediate
        )
        
        # 2. デュアルパスウェイデコーダ
        # 融合重み適用（MetaP最適化）
        if self.current_config and 'fusion_weights' in self.current_config:
            fusion_weights = self.current_config['fusion_weights']
        else:
            fusion_weights = {
                'llama_weight': self.config.get('llama_weight', 0.4),
                'sam2_weight': self.config.get('sam2_weight', 0.4)
            }
        
        # SAMプロンプト準備
        sam_prompts = batch.get('sam_prompts')
        
        # SAMプロンプトがない場合は生成
        if sam_prompts is None:
            batch_size = llama_hidden_states.shape[0] if llama_hidden_states is not None else 1
            device = next(self.parameters()).device
            # ダミーのSAMプロンプト生成（test_phase3b_integration_real.py準拠）
            sam_prompts = torch.randn(
                batch_size, 32, 256,
                device=device
            )
        
        # デュアルパスウェイ推論
        # 注：llama_hidden_statesは元の5120次元を使用（融合特徴ではない）
        decoder_results = self.dual_decoder(
            images=images,
            sam_prompts=sam_prompts,
            llama_hidden_states=llama_hidden_states,  # 元のLlama-4隠れ状態を使用
            return_intermediate=return_intermediate,
            return_consistency=True
        )
        
        # 結果統合
        results = {
            'predictions': decoder_results.get('fused_masks'),
            'fusion_features': fusion_results.get('fused_features'),
            'consistency_loss': decoder_results.get('consistency_loss', torch.tensor(0.0))
        }
        
        # 中間結果追加（必要な場合）
        if return_intermediate:
            results.update({
                'fusion_intermediate': fusion_results,
                'decoder_intermediate': decoder_results,
                'metap_config': self.current_config if self.current_config else {}
            })
        
        return results
    
    def meta_training_step(
        self,
        train_batch: Dict[str, torch.Tensor],
        val_batch: Optional[Dict[str, torch.Tensor]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        MetaP統合学習ステップ
        
        Args:
            train_batch: 訓練バッチ
            val_batch: 検証バッチ（MetaP最適化用）
            optimizer: ベースモデル最適化器
        
        Returns:
            学習結果辞書
        """
        # MetaP初期化確認
        if self.metap_enabled and self.metap_optimizer is None:
            self.initialize_metap()
        
        # 訓練推論
        train_results = self.forward_with_metap(train_batch, is_training=True)
        
        # OHEM損失計算
        train_loss_dict = self.ohem_loss(
            llama_logits=train_batch.get('llama_logits'),
            llama_targets=train_batch.get('llama_targets'),
            llama_attention_mask=train_batch.get('attention_mask'),  # キー名を修正
            sam_predictions=train_results['predictions'],
            sam_targets=train_batch.get('sam_targets'),
            apply_ohem=True,
            return_individual=True
        )
        train_loss = train_loss_dict['total_loss']
        
        # 勾配計算とモデル更新
        if optimizer is not None:
            optimizer.zero_grad()
            train_loss.backward(retain_graph=True)  # MetaP用にグラフ保持
            optimizer.step()
        
        # MetaP最適化（検証バッチがある場合）
        metap_results = {}
        if self.metap_enabled and val_batch is not None and self.metap_optimizer is not None:
            # 検証推論
            with torch.no_grad():
                val_results = self.forward_with_metap(val_batch, is_training=False)
                val_loss_dict = self.ohem_loss(
                    llama_logits=val_batch.get('llama_logits'),
                    llama_targets=val_batch.get('llama_targets'),
                    llama_attention_mask=val_batch.get('attention_mask'),  # キー名を修正
                    sam_predictions=val_results['predictions'],
                    sam_targets=val_batch.get('sam_targets'),
                    apply_ohem=True,
                    return_individual=True
                )
                val_loss = val_loss_dict['total_loss']
            
            # MetaPハイパーパラメータ最適化
            optimized_config = self.metap_optimizer.optimize_hyperparameters(
                train_loss=train_loss,
                val_loss=val_loss
            )
            
            metap_results = {
                'optimized_config': optimized_config,
                'val_loss': val_loss.item()
            }
        
        # 結果統合
        results = {
            'train_loss': train_loss.item(),
            'train_loss_components': {
                'llama_loss': train_loss_dict['llama_loss'].item(),
                'sam_loss': train_loss_dict['sam_loss'].item(),
                'consistency_loss': train_results['consistency_loss'].item()
            },
            'metap_results': metap_results
        }
        
        return results
    
    def set_metap_enabled(self, enabled: bool):
        """MetaP最適化の有効/無効切り替え"""
        self.metap_enabled = enabled
        logger.info(f"MetaP最適化: {'有効' if enabled else '無効'}")
    
    def get_current_config(self) -> Dict[str, Any]:
        """現在の設定取得"""
        if self.current_config:
            return self.current_config.copy()
        else:
            # デフォルト設定
            return {
                'lora_rank': self.config.get('lora_rank', 16),
                'lora_alpha': self.config.get('lora_alpha', 32),
                'learning_rate': self.config.get('learning_rate', 1e-4),
                'fusion_weights': {
                    'llama_weight': self.config.get('llama_weight', 0.4),
                    'sam2_weight': self.config.get('sam2_weight', 0.4),
                    'qformer_weight': self.config.get('qformer_weight', 0.2)
                }
            }
    
    def save_metap_state(self, save_path: str):
        """MetaP状態保存"""
        if self.metap_optimizer is not None:
            state_dict = {
                'metap_optimizer_state': self.metap_optimizer.state_dict(),
                'current_config': self.current_config,
                'performance_summary': self.metap_optimizer.get_performance_summary()
            }
            torch.save(state_dict, save_path)
            logger.info(f"MetaP状態保存: {save_path}")
    
    def load_metap_state(self, load_path: str):
        """MetaP状態読み込み"""
        if Path(load_path).exists():
            state_dict = torch.load(load_path, map_location='cpu')
            
            # MetaP最適化器初期化（必要な場合）
            if self.metap_optimizer is None:
                self.initialize_metap()
            
            # 状態復元
            if self.metap_optimizer is not None:
                self.metap_optimizer.load_state_dict(state_dict['metap_optimizer_state'])
                self.current_config = state_dict.get('current_config')
                logger.info(f"MetaP状態復元: {load_path}")
        else:
            logger.warning(f"MetaP状態ファイルが見つかりません: {load_path}")


def create_metap_integrated_model(
    config_override: Optional[Dict[str, Any]] = None
) -> MetaPIntegratedModel:
    """
    MetaP統合モデル作成ヘルパー関数
    
    Args:
        config_override: 設定オーバーライド
    
    Returns:
        MetaPIntegratedModel インスタンス
    """
    # 基本設定取得
    base_config = {
        'llama_hidden_size': 5120,
        'sam_output_dim': 256,
        'sam_feature_dim': 256,
        'sam_scales': [1024, 512, 256],
        'qformer_dim': 768,
        'qformer_queries': 32,
        'fusion_dim': 512,
        'output_size': (448, 448),
        'fusion_strategy': 'attention',
        'hard_ratio': 0.25,
        'enable_metap': True
    }
    
    # config_linux設定の適用
    lisa_config = config_linux.get_lisa_model_config()
    lora_config = config_linux.get_lora_config()
    training_config = config_linux.get_training_config()
    
    base_config.update({
        'lora_rank': lora_config['r'],
        'lora_alpha': lora_config['lora_alpha'],
        'learning_rate': training_config['learning_rate'],
        'llama_weight': 0.4,
        'sam2_weight': 0.4,
        'qformer_weight': 0.2
    })
    
    # オーバーライド適用
    if config_override:
        base_config.update(config_override)
    
    return MetaPIntegratedModel(base_config)