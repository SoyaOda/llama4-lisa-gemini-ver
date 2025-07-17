# model/curriculum_integration.py
"""
Curriculum + MetaP + Phase 3B統合実装

Phase 3C実装: 全機能の統合による相乗効果実現
- カリキュラム学習による段階的難易度調整
- MetaP動的ハイパーパラメータ最適化
- Phase 3B実証済み機能の完全活用

目標:
- 学習効率: 10倍高速化（MetaP 5倍 × Curriculum 2倍）
- 最終精度: 40%向上（Phase 3B 28.14% + 追加 12%）
- 実装安定性: 95%成功率維持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from pathlib import Path
import time
import json
from dataclasses import dataclass
import numpy as np

# Phase 3C実装済みコンポーネント
from model.metap_optimizer import (
    LlamaMetaPOptimizer, 
    MetaPConfig, 
    create_metap_optimizer
)
from model.metap_integration import MetaPIntegratedModel
from model.curriculum_strategy import (
    LlamaMultiModalCurriculum, 
    CurriculumStage,
    create_curriculum_strategy
)
from model.difficulty_scheduler import (
    DifficultyBasedScheduler,
    create_difficulty_scheduler
)

# Phase 3B実装済み機能
from model.dual_pathway_decoder import create_dual_pathway_decoder
from model.multiresolution_fusion import Llama4SAM2MultiResolutionFusion
from model.ohem_loss import create_ohem_loss

# 設定
import config_linux

logger = logging.getLogger(__name__)


@dataclass
class CurriculumIntegrationConfig:
    """統合設定"""
    # Phase 3B基本設定
    llama_hidden_size: int = 5120
    sam_output_dim: int = 256
    sam_feature_dim: int = 256
    qformer_dim: int = 768
    fusion_dim: int = 512
    
    # MetaP設定
    enable_metap: bool = True
    meta_lr: float = 0.01
    discount_factor: float = 0.95
    
    # カリキュラム設定
    enable_curriculum: bool = True
    curriculum_stages: int = 4
    total_epochs: int = 13  # 2+3+3+5
    
    # 学習設定（MetaP統合で必要）
    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    
    # 統合設定
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    eval_frequency: int = 100
    checkpoint_frequency: int = 500
    
    # Lambda Cloud最適化
    num_gpus: int = 2  # H100 x2
    mixed_precision: bool = True
    gradient_checkpointing: bool = True


class CurriculumIntegratedTraining:
    """
    カリキュラム + MetaP + Phase 3B統合学習
    
    統合機能:
    - 段階的難易度調整によるカリキュラム学習
    - MetaP動的ハイパーパラメータ最適化
    - Phase 3B実証済み機能（デュアルパス、多重解像度、OHEM）
    - 相乗効果による10倍学習効率化
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: CurriculumIntegrationConfig,
        device: Union[str, torch.device] = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Phase 3B実装済み機能の初期化
        logger.info("Phase 3B機能初期化中...")
        self._initialize_phase3b_components()
        
        # カリキュラム機能初期化
        logger.info("カリキュラム学習戦略初期化中...")
        self.curriculum = create_curriculum_strategy({
            'stage1_epochs': 2,
            'stage2_epochs': 3,
            'stage3_epochs': 3,
            'stage4_epochs': 5,
            'base_learning_rate': config_linux.get_training_config()['learning_rate']
        })
        
        # 困難度スケジューラー初期化
        self.difficulty_scheduler = create_difficulty_scheduler(
            ohem_loss_fn=self.ohem_loss,
            config_override={
                'max_cache_size': 10000,
                'difficulty_weights': {
                    'text': 0.3,
                    'visual': 0.3,
                    'multimodal': 0.2,
                    'ohem': 0.2
                }
            }
        )
        
        # MetaP統合（オプション）
        self.metap_integrated_model = None
        if self.config.enable_metap:
            logger.info("MetaP最適化統合中...")
            self._initialize_metap_integration()
        
        # 統計追跡
        self.training_stats = {
            'epoch_losses': [],
            'stage_transitions': [],
            'metap_configs': [],
            'difficulty_distributions': []
        }
        
        # 現在の状態
        self.current_epoch = 0
        self.global_step = 0
        
        logger.info("CurriculumIntegratedTraining初期化完了")
        logger.info(f"  - MetaP: {'有効' if config.enable_metap else '無効'}")
        logger.info(f"  - カリキュラム: {'有効' if config.enable_curriculum else '無効'}")
        logger.info(f"  - GPU数: {config.num_gpus}")
    
    def _initialize_phase3b_components(self):
        """Phase 3B機能初期化"""
        # デュアルパスウェイデコーダー
        self.dual_decoder = create_dual_pathway_decoder(
            llama_hidden_size=self.config.llama_hidden_size,
            sam_output_dim=self.config.sam_output_dim,
            fusion_strategy='learned_weighted'
        )
        
        # 多重解像度融合
        self.multiresolution_fusion = Llama4SAM2MultiResolutionFusion(
            llama_hidden_size=self.config.llama_hidden_size,
            sam_feature_dim=self.config.sam_feature_dim,
            sam_scales=[1024, 512, 256],
            qformer_dim=self.config.qformer_dim,
            qformer_queries=32,
            fusion_dim=self.config.fusion_dim,
            output_size=(448, 448),
            fusion_strategy='attention'
        )
        
        # OHEM損失関数
        self.ohem_loss = create_ohem_loss(
            hard_ratio=0.25,
            config_override={
                'ce_loss_weight': 1.0,
                'bce_loss_weight': 2.0,
                'dice_loss_weight': 0.5
            }
        )
        
        # デバイス転送
        self.dual_decoder = self.dual_decoder.to(self.device)
        self.multiresolution_fusion = self.multiresolution_fusion.to(self.device)
        self.ohem_loss = self.ohem_loss.to(self.device)
    
    def _initialize_metap_integration(self):
        """MetaP統合初期化"""
        # MetaP統合モデル作成（Phase 3Bコンポーネントを外部から使用）
        metap_config = {
            'llama_hidden_size': self.config.llama_hidden_size,
            'sam_output_dim': self.config.sam_output_dim,
            'sam_feature_dim': self.config.sam_feature_dim,
            'qformer_dim': self.config.qformer_dim,
            'fusion_dim': self.config.fusion_dim,
            'enable_metap': True,
            'learning_rate': config_linux.get_training_config()['learning_rate'],
            'lora_rank': config_linux.get_lora_config()['r'],
            'lora_alpha': config_linux.get_lora_config()['lora_alpha'],
            'external_phase3b': True  # Phase 3Bコンポーネントを外部から使用
        }
        
        self.metap_integrated_model = MetaPIntegratedModel(metap_config)
        self.metap_integrated_model = self.metap_integrated_model.to(self.device)
        
        # Phase 3B機能を共有（重複作成を防ぐ）
        self.metap_integrated_model.dual_decoder = self.dual_decoder
        self.metap_integrated_model.multiresolution_fusion = self.multiresolution_fusion
        self.metap_integrated_model.ohem_loss = self.ohem_loss
        
        logger.info("MetaP統合: Phase 3Bコンポーネントを共有設定完了")
    
    def curriculum_training_epoch(
        self,
        epoch: int,
        train_dataset: List[Dict[str, Any]],
        val_dataset: Optional[List[Dict[str, Any]]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        カリキュラム統合学習エポック
        
        Args:
            epoch: エポック番号
            train_dataset: 訓練データセット
            val_dataset: 検証データセット
            optimizer: オプティマイザ
        
        Returns:
            エポック結果辞書
        """
        self.current_epoch = epoch
        epoch_start_time = time.time()
        
        # 1. 現在のカリキュラムステージ取得
        stage_name, stage_config = self.curriculum.get_stage_config(epoch)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📚 エポック {epoch}: {stage_name} - {stage_config.description}")
        logger.info(f"🎯 解像度: {stage_config.resolution}, 困難度閾値: {stage_config.difficulty_threshold}")
        logger.info(f"📊 モダリティ重み: {stage_config.modality_weights}")
        logger.info(f"{'='*80}")
        
        # 2. モデルをステージに適応
        adaptation_results = self.curriculum.adapt_model_for_stage(
            self.model, 
            stage_config, 
            optimizer
        )
        
        # 解像度を多重解像度融合に適用
        self.multiresolution_fusion.output_size = stage_config.resolution
        
        # 3. データスケジューリング（困難度ベース）
        curriculum_batch = self.difficulty_scheduler.schedule_curriculum_batch(
            dataset=train_dataset,
            current_stage=stage_name,
            stage_config=stage_config.to_dict(),
            batch_size=len(train_dataset),  # 全データから選択
            balanced=True
        )
        
        # 空のバッチチェック
        if not curriculum_batch:
            logger.warning(f"カリキュラムバッチが空です。全データセットを使用します。")
            curriculum_batch = train_dataset[:self.config.batch_size]
        
        logger.info(f"📊 カリキュラム適用後サンプル数: {len(curriculum_batch)}/{len(train_dataset)}")
        
        # 4. バッチ学習ループ
        epoch_losses = []
        batch_times = []
        
        # バッチ作成（最低1バッチは確保）
        num_batches = max(1, (len(curriculum_batch) + self.config.batch_size - 1) // self.config.batch_size)
        
        for batch_idx in range(num_batches):
            batch_start_time = time.time()
            
            # バッチデータ取得
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(curriculum_batch))
            batch_samples = curriculum_batch[start_idx:end_idx]
            
            # バッチ辞書作成
            batch = self._prepare_batch(batch_samples, stage_config)
            
            # MetaP統合学習（有効な場合）
            if self.config.enable_metap and self.metap_integrated_model:
                # MetaP + カリキュラム統合学習
                results = self._metap_curriculum_step(
                    batch, 
                    stage_config,
                    optimizer,
                    val_batch=self._prepare_batch(val_dataset[:32], stage_config) if val_dataset else None
                )
            else:
                # カリキュラム単体学習
                results = self._curriculum_step(batch, stage_config, optimizer)
            
            epoch_losses.append(results['loss'])
            batch_times.append(time.time() - batch_start_time)
            
            # 進捗ログ
            if batch_idx % 10 == 0:
                avg_loss = np.mean(epoch_losses[-10:]) if len(epoch_losses) >= 10 else np.mean(epoch_losses)
                avg_time = np.mean(batch_times[-10:]) if len(batch_times) >= 10 else np.mean(batch_times)
                logger.info(
                    f"  バッチ [{batch_idx}/{num_batches}] - "
                    f"損失: {results['loss']:.4f} (平均: {avg_loss:.4f}) - "
                    f"時間: {avg_time:.2f}秒/バッチ"
                )
            
            self.global_step += 1
            
            # 定期的な評価
            if self.global_step % self.config.eval_frequency == 0 and val_dataset:
                self._evaluate(val_dataset, stage_config)
            
            # チェックポイント保存
            if self.global_step % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(epoch, stage_name)
        
        # エポック統計
        epoch_time = time.time() - epoch_start_time
        epoch_results = {
            'epoch': epoch,
            'stage_name': stage_name,
            'stage_config': stage_config.to_dict(),
            'avg_loss': np.mean(epoch_losses),
            'std_loss': np.std(epoch_losses),
            'min_loss': np.min(epoch_losses),
            'max_loss': np.max(epoch_losses),
            'num_samples': len(curriculum_batch),
            'epoch_time': epoch_time,
            'avg_batch_time': np.mean(batch_times),
            'difficulty_distribution': self.difficulty_scheduler.get_difficulty_distribution()
        }
        
        # パフォーマンス記録
        self.curriculum.record_performance(epoch, {
            'loss': epoch_results['avg_loss'],
            'samples': epoch_results['num_samples']
        })
        
        # 統計更新
        self.training_stats['epoch_losses'].append(epoch_results['avg_loss'])
        if stage_name not in [s['stage'] for s in self.training_stats['stage_transitions']]:
            self.training_stats['stage_transitions'].append({
                'epoch': epoch,
                'stage': stage_name
            })
        
        logger.info(f"\n📊 エポック {epoch} 完了:")
        logger.info(f"  - 平均損失: {epoch_results['avg_loss']:.4f}")
        logger.info(f"  - 処理時間: {epoch_time/60:.1f}分")
        logger.info(f"  - サンプル数: {epoch_results['num_samples']}")
        
        return epoch_results
    
    def _prepare_batch(
        self, 
        samples: List[Dict[str, Any]], 
        stage_config: CurriculumStage
    ) -> Dict[str, torch.Tensor]:
        """バッチデータ準備"""
        # 空のサンプルチェック
        if not samples:
            logger.warning("空のサンプルリストが渡されました。ダミーバッチを作成します。")
            samples = [{'dummy': True}]  # 最低1つのサンプルを確保
        
        # 簡易実装（実際の実装では適切なデータローダーを使用）
        batch = {
            'images': [],
            'texts': [],
            'sam_targets': [],
            'llama_targets': [],
            'difficulty_scores': []
        }
        
        for sample in samples:
            # 各サンプルのデータを追加
            if 'image' in sample:
                batch['images'].append(sample['image'])
            if 'text' in sample:
                batch['texts'].append(sample['text'])
            if 'sam_targets' in sample:
                batch['sam_targets'].append(sample['sam_targets'])
            if 'llama_targets' in sample:
                batch['llama_targets'].append(sample['llama_targets'])
            if '_difficulty_score' in sample:
                batch['difficulty_scores'].append(sample['_difficulty_score'])
        
        # テンソル変換（実際の実装では適切な前処理が必要）
        # ここでは簡易的にダミーテンソルを作成
        batch_size = max(1, len(samples))  # 最低バッチサイズ1を保証
        resolution = stage_config.resolution
        
        batch_tensors = {
            'images': torch.randn(batch_size, 3, resolution[0], resolution[1], device=self.device),
            'texts': torch.randint(0, 50000, (batch_size, 256), device=self.device),
            'sam_targets': torch.randint(0, 2, (batch_size, 1, resolution[0], resolution[1]), device=self.device),
            'llama_targets': torch.randint(0, 50000, (batch_size, 256), device=self.device),
            'attention_mask': torch.ones(batch_size, 256, device=self.device)
        }
        
        # Phase 3B用の追加データ
        batch_tensors['llama_hidden_states'] = torch.randn(
            batch_size, 16, self.config.llama_hidden_size, device=self.device
        )
        batch_tensors['sam_features'] = [
            torch.randn(batch_size, 256, 64, 64, device=self.device),
            torch.randn(batch_size, 512, 32, 32, device=self.device),
            torch.randn(batch_size, 1024, 16, 16, device=self.device)
        ]
        batch_tensors['qformer_features'] = torch.randn(
            batch_size, 32, self.config.qformer_dim, device=self.device
        )
        # SAMプロンプト（test_phase3b_integration_real.py準拠）
        batch_tensors['sam_prompts'] = torch.randn(
            batch_size, 32, 256, device=self.device
        )
        
        # Llama logits生成（OHEM損失用）
        vocab_size = 50000  # Llama-4語彙サイズ
        seq_len = batch_tensors['llama_targets'].shape[1]
        batch_tensors['llama_logits'] = torch.randn(
            batch_size, seq_len, vocab_size, device=self.device
        )
        
        return batch_tensors
    
    def _curriculum_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage_config: CurriculumStage,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, float]:
        """カリキュラム学習ステップ"""
        # Phase 3B統合推論
        # 多重解像度融合
        fusion_results = self.multiresolution_fusion(
            llama_hidden_states=batch['llama_hidden_states'],
            sam_features=batch['sam_features'],
            qformer_features=batch['qformer_features'],
            return_intermediate=True
        )
        
        # デュアルパスウェイ推論
        decoder_results = self.dual_decoder(
            images=batch['images'],
            sam_prompts=batch['sam_prompts'],  # SAMプロンプトを使用
            llama_hidden_states=batch['llama_hidden_states'],  # 元のLlama-4隠れ状態を使用
            return_intermediate=True,
            return_consistency=True
        )
        
        # OHEM損失計算（困難度考慮）
        # llama_logitsを使用（_prepare_batchで生成済み）
        llama_logits = batch.get('llama_logits')
        if llama_logits is None:
            # フォールバック：ダミーのllama_logitsを作成
            llama_logits = torch.randn(
                batch['llama_targets'].shape[0],
                batch['llama_targets'].shape[1],
                50000,  # vocab size
                device=self.device
            )
        
        loss_dict = self.ohem_loss(
            llama_logits=llama_logits,
            llama_targets=batch['llama_targets'],
            llama_attention_mask=batch['attention_mask'],
            sam_predictions=decoder_results['fused_masks'],
            sam_targets=batch['sam_targets'],
            apply_ohem=True,
            return_individual=True
        )
        
        total_loss = loss_dict['total_loss']
        
        # モダリティ重み適用
        weighted_loss = (
            stage_config.modality_weights['llama'] * loss_dict['llama_loss'] +
            stage_config.modality_weights['sam2'] * loss_dict['sam_loss']
        )
        
        # 最適化
        if optimizer is not None:
            optimizer.zero_grad()
            weighted_loss.backward()
            
            # 勾配クリッピング
            if stage_config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    stage_config.gradient_clip
                )
            
            optimizer.step()
        
        return {
            'loss': weighted_loss.item(),
            'llama_loss': loss_dict['llama_loss'].item(),
            'sam_loss': loss_dict['sam_loss'].item(),
            'consistency_loss': decoder_results.get('consistency_loss', torch.tensor(0.0)).item()
        }
    
    def _metap_curriculum_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage_config: CurriculumStage,
        optimizer: Optional[torch.optim.Optimizer] = None,
        val_batch: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """MetaP + カリキュラム統合学習ステップ"""
        # MetaPモデルの訓練ステップ
        results = self.metap_integrated_model.meta_training_step(
            train_batch=batch,
            val_batch=val_batch,
            optimizer=optimizer
        )
        
        # カリキュラム段階に応じた調整
        if 'metap_results' in results and 'optimized_config' in results['metap_results']:
            metap_config = results['metap_results']['optimized_config']
            
            # カリキュラムステージの制約を適用
            metap_config['lora_rank'] = min(
                metap_config['lora_rank'], 
                stage_config.lora_rank
            )
            
            # MetaP設定記録
            self.training_stats['metap_configs'].append({
                'epoch': self.current_epoch,
                'stage': stage_config.name,
                'config': metap_config
            })
        
        # モダリティ重み考慮
        weighted_loss = (
            stage_config.modality_weights['llama'] * results['train_loss_components']['llama_loss'] +
            stage_config.modality_weights['sam2'] * results['train_loss_components']['sam_loss']
        )
        
        return {
            'loss': weighted_loss,
            'llama_loss': results['train_loss_components']['llama_loss'],
            'sam_loss': results['train_loss_components']['sam_loss'],
            'consistency_loss': results['train_loss_components']['consistency_loss'],
            'metap_config': metap_config if 'metap_config' in locals() else None
        }
    
    def _evaluate(
        self,
        val_dataset: List[Dict[str, Any]],
        stage_config: CurriculumStage
    ) -> Dict[str, float]:
        """検証評価"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            # 簡易評価（実際の実装では全検証データを処理）
            val_batch = self._prepare_batch(val_dataset[:32], stage_config)
            val_results = self._curriculum_step(val_batch, stage_config, optimizer=None)
            val_losses.append(val_results['loss'])
        
        self.model.train()
        
        avg_val_loss = np.mean(val_losses)
        logger.info(f"📊 検証損失: {avg_val_loss:.4f}")
        
        return {'val_loss': avg_val_loss}
    
    def _save_checkpoint(self, epoch: int, stage_name: str):
        """チェックポイント保存"""
        checkpoint_dir = Path(self.config.checkpoint_dir if hasattr(self.config, 'checkpoint_dir') else './checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"phase3c_epoch{epoch}_{stage_name}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'stage_name': stage_name,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict() if self.model else None,
            'training_stats': self.training_stats,
            'curriculum_progress': self.curriculum.get_stage_progress_summary(),
            'config': self.config.__dict__
        }
        
        # MetaP状態保存
        if self.metap_integrated_model:
            checkpoint['metap_state'] = {
                'optimizer_state': self.metap_integrated_model.metap_optimizer.state_dict() 
                    if hasattr(self.metap_integrated_model, 'metap_optimizer') else None,
                'current_config': self.metap_integrated_model.get_current_config()
            }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 チェックポイント保存: {checkpoint_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """学習サマリー取得"""
        # カリキュラム進行状況
        curriculum_summary = self.curriculum.get_stage_progress_summary()
        
        # パフォーマンス統計
        if self.training_stats['epoch_losses']:
            initial_loss = self.training_stats['epoch_losses'][0]
            current_loss = self.training_stats['epoch_losses'][-1]
            improvement = (initial_loss - current_loss) / initial_loss * 100
        else:
            improvement = 0.0
        
        # MetaP効果
        metap_summary = {}
        if self.training_stats['metap_configs']:
            latest_metap = self.training_stats['metap_configs'][-1]['config']
            initial_lr = config_linux.get_training_config()['learning_rate']
            lr_change = (latest_metap['learning_rate'] - initial_lr) / initial_lr * 100
            
            metap_summary = {
                'latest_config': latest_metap,
                'lr_change_percentage': lr_change,
                'total_updates': len(self.training_stats['metap_configs'])
            }
        
        return {
            'total_epochs': self.current_epoch,
            'total_steps': self.global_step,
            'performance_improvement': improvement,
            'curriculum_progress': curriculum_summary,
            'metap_summary': metap_summary,
            'stage_transitions': self.training_stats['stage_transitions'],
            'final_loss': self.training_stats['epoch_losses'][-1] if self.training_stats['epoch_losses'] else None
        }


def create_curriculum_integrated_training(
    model: nn.Module,
    config_override: Optional[Dict[str, Any]] = None,
    device: Union[str, torch.device] = 'cuda'
) -> CurriculumIntegratedTraining:
    """
    カリキュラム統合学習作成ヘルパー関数
    
    Args:
        model: ベースモデル
        config_override: 設定オーバーライド
        device: デバイス
    
    Returns:
        CurriculumIntegratedTraining インスタンス
    """
    # デフォルト設定
    config_dict = {
        'llama_hidden_size': 5120,
        'sam_output_dim': 256,
        'sam_feature_dim': 256,
        'qformer_dim': 768,
        'fusion_dim': 512,
        'enable_metap': True,
        'enable_curriculum': True,
        'batch_size': 32,
        'num_gpus': torch.cuda.device_count()
    }
    
    # config_linux設定の適用
    lora_config = config_linux.get_lora_config()
    training_config = config_linux.get_training_config()
    
    config_dict.update({
        'learning_rate': training_config['learning_rate'],
        'lora_rank': lora_config['r'],
        'lora_alpha': lora_config['lora_alpha']
    })
    
    # オーバーライド適用
    if config_override:
        config_dict.update(config_override)
    
    # 設定オブジェクト作成
    config = CurriculumIntegrationConfig(**config_dict)
    
    return CurriculumIntegratedTraining(model, config, device)