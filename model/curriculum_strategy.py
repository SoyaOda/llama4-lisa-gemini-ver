# model/curriculum_strategy.py
"""
Curriculum Learning Strategy for Llama-4 + SAM2 + Q-Former

Phase 3C実装: 段階的難易度調整による学習効率化
- モーダル間バランスの段階的調整
- 解像度プログレッシブ学習
- 損失重み動的変更
- Phase 3B OHEM知見活用

目標:
- 収束速度: 2-3倍向上
- 最終精度: 5-10%改善
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """カリキュラム学習ステージ定義"""
    name: str
    epochs: int
    description: str
    resolution: Tuple[int, int]
    modality_weights: Dict[str, float]
    lora_rank: int
    difficulty_threshold: float
    learning_rate_multiplier: float
    
    # オプション設定
    enable_augmentation: bool = True
    dropout_rate: float = 0.1
    gradient_clip: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'name': self.name,
            'epochs': self.epochs,
            'description': self.description,
            'resolution': self.resolution,
            'modality_weights': self.modality_weights,
            'lora_rank': self.lora_rank,
            'difficulty_threshold': self.difficulty_threshold,
            'learning_rate_multiplier': self.learning_rate_multiplier,
            'enable_augmentation': self.enable_augmentation,
            'dropout_rate': self.dropout_rate,
            'gradient_clip': self.gradient_clip
        }


class LlamaMultiModalCurriculum:
    """
    Llama-4+SAM2+Q-Former特化カリキュラム学習戦略
    
    4段階プログレッシブ学習:
    1. テキスト理解基盤構築
    2. ビジョン機能段階導入
    3. Q-Former統合・クロスモーダル学習
    4. フル機能統合・高精度学習
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Phase 3B実装知見を活用したカリキュラム設計
        self.curriculum_stages = self._build_curriculum_stages()
        
        # 現在の状態
        self.current_stage_idx = 0
        self.current_epoch = 0
        self.stage_progress = {}
        
        # パフォーマンス追跡
        self.performance_history = []
        
        logger.info("LlamaMultiModalCurriculum初期化完了")
        logger.info(f"  - 総ステージ数: {len(self.curriculum_stages)}")
        logger.info(f"  - 総エポック数: {sum(stage.epochs for stage in self.curriculum_stages)}")
    
    def _build_curriculum_stages(self) -> List[CurriculumStage]:
        """カリキュラムステージ構築（2024-2025最新研究準拠）"""
        stages = []
        
        # Stage 1: テキスト理解基盤構築
        stages.append(CurriculumStage(
            name='stage1_text_foundation',
            epochs=self.config.get('stage1_epochs', 2),
            description='テキスト理解基盤構築',
            resolution=(112, 112),  # fast_inference解像度
            modality_weights={
                'llama': 1.0,       # テキストのみ集中
                'sam2': 0.0,
                'qformer': 0.0
            },
            lora_rank=8,            # 低ランクから開始
            difficulty_threshold=0.3,  # 簡単なサンプルのみ
            learning_rate_multiplier=1.5,  # 高めの学習率で高速学習
            enable_augmentation=False,  # Stage 1では無効
            dropout_rate=0.05  # 低めのドロップアウト
        ))
        
        # Stage 2: ビジョン機能段階導入
        stages.append(CurriculumStage(
            name='stage2_vision_introduction',
            epochs=self.config.get('stage2_epochs', 3),
            description='ビジョン機能段階導入',
            resolution=(224, 224),  # app_optimal解像度
            modality_weights={
                'llama': 0.7,
                'sam2': 0.3,        # SAM2段階導入
                'qformer': 0.0
            },
            lora_rank=12,
            difficulty_threshold=0.6,
            learning_rate_multiplier=1.2,
            enable_augmentation=True,
            dropout_rate=0.1
        ))
        
        # Stage 3: Q-Former統合・クロスモーダル学習
        stages.append(CurriculumStage(
            name='stage3_qformer_integration',
            epochs=self.config.get('stage3_epochs', 3),
            description='Q-Former統合・クロスモーダル学習',
            resolution=(336, 336),  # 中解像度
            modality_weights={
                'llama': 0.5,
                'sam2': 0.3,
                'qformer': 0.2      # Q-Former段階導入
            },
            lora_rank=16,           # 論文準拠値到達
            difficulty_threshold=0.8,
            learning_rate_multiplier=1.0,
            enable_augmentation=True,
            dropout_rate=0.1
        ))
        
        # Stage 4: フル機能統合・高精度学習
        stages.append(CurriculumStage(
            name='stage4_full_multimodal',
            epochs=self.config.get('stage4_epochs', 5),
            description='フル機能統合・高精度学習',
            resolution=(448, 448),  # high_accuracy解像度
            modality_weights={
                'llama': 0.4,       # Phase 3B実証値
                'sam2': 0.4,
                'qformer': 0.2
            },
            lora_rank=16,           # 論文準拠最終値
            difficulty_threshold=1.0,  # 全難易度対応
            learning_rate_multiplier=0.8,  # 安定した学習
            enable_augmentation=True,
            dropout_rate=0.1,
            gradient_clip=0.5  # より厳しいクリッピング
        ))
        
        return stages
    
    def get_stage_config(self, current_epoch: int) -> Tuple[str, CurriculumStage]:
        """
        現在エポックに基づくステージ設定取得
        
        Args:
            current_epoch: 現在のエポック数（0-indexed）
        
        Returns:
            (ステージ名, ステージ設定)のタプル
        """
        self.current_epoch = current_epoch
        cumulative_epochs = 0
        
        for idx, stage in enumerate(self.curriculum_stages):
            cumulative_epochs += stage.epochs
            if current_epoch < cumulative_epochs:
                self.current_stage_idx = idx
                return stage.name, stage
        
        # 最終ステージを返す
        self.current_stage_idx = len(self.curriculum_stages) - 1
        return self.curriculum_stages[-1].name, self.curriculum_stages[-1]
    
    def adapt_model_for_stage(
        self, 
        model: nn.Module, 
        stage_config: CurriculumStage,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        ステージ設定に基づくモデル動的適応
        
        Args:
            model: 対象モデル
            stage_config: ステージ設定
            optimizer: オプティマイザ（学習率調整用）
        
        Returns:
            適応結果の辞書
        """
        adaptation_results = {}
        
        try:
            # 1. 解像度動的変更（Phase 3B実装活用）
            if hasattr(model, 'multiresolution_fusion'):
                # Llama4SAM2MultiResolutionFusionの解像度設定
                model.multiresolution_fusion.output_size = stage_config.resolution
                adaptation_results['resolution'] = stage_config.resolution
                logger.info(f"解像度設定: {stage_config.resolution}")
            
            # 2. LoRA rank動的変更
            if hasattr(model, 'peft_config'):
                # PEFTConfigは不変のため、直接変更はできない
                # 現在の設定を記録するのみ
                adaptation_results['lora_rank'] = stage_config.lora_rank
                adaptation_results['lora_alpha'] = stage_config.lora_rank * 2
                logger.info(f"LoRA設定要求: r={stage_config.lora_rank}, α={stage_config.lora_rank * 2}")
                logger.info("注: PEFTConfigは不変のため、実行時の動的変更はサポートされていません")
            
            # 3. モダリティ重み調整（Phase 3B融合重み活用）
            if hasattr(model, 'fusion_weights'):
                for modality, weight in stage_config.modality_weights.items():
                    if hasattr(model.fusion_weights, f'{modality}_weight'):
                        setattr(model.fusion_weights, f'{modality}_weight', weight)
                adaptation_results['modality_weights'] = stage_config.modality_weights
                logger.info(f"モダリティ重み: {stage_config.modality_weights}")
            elif hasattr(model, 'dual_decoder') and hasattr(model.dual_decoder, 'fusion_weight'):
                # デュアルデコーダーの重み調整
                llama_weight = stage_config.modality_weights.get('llama', 0.5)
                sam_weight = stage_config.modality_weights.get('sam2', 0.5)
                model.dual_decoder.fusion_weight.data = torch.tensor([llama_weight, sam_weight])
                adaptation_results['dual_decoder_weights'] = [llama_weight, sam_weight]
            
            # 4. 学習率動的調整
            if optimizer is not None:
                base_lr = self.config.get('base_learning_rate', 1e-4)
                new_lr = base_lr * stage_config.learning_rate_multiplier
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                adaptation_results['learning_rate'] = new_lr
                logger.info(f"学習率: {new_lr:.6f}")
            
            # 5. ドロップアウト率調整
            if hasattr(model, 'dropout'):
                model.dropout.p = stage_config.dropout_rate
                adaptation_results['dropout_rate'] = stage_config.dropout_rate
            
            # 6. データ拡張設定
            adaptation_results['enable_augmentation'] = stage_config.enable_augmentation
            adaptation_results['gradient_clip'] = stage_config.gradient_clip
            
            # ステージ進行記録
            self.stage_progress[stage_config.name] = {
                'started_epoch': self.current_epoch,
                'adaptation_results': adaptation_results
            }
            
            logger.info(f"✓ ステージ適応完了: {stage_config.name}")
            
        except Exception as e:
            logger.error(f"ステージ適応エラー: {e}")
            adaptation_results['error'] = str(e)
        
        return adaptation_results
    
    def should_advance_stage(
        self, 
        current_metrics: Dict[str, float],
        min_improvement: float = 0.01
    ) -> bool:
        """
        ステージ進行判定（早期進行オプション）
        
        Args:
            current_metrics: 現在のメトリクス
            min_improvement: 最小改善率
        
        Returns:
            進行すべきかのフラグ
        """
        # 基本的にはエポック数で管理
        # 将来的には性能ベースの進行判定を実装可能
        return False
    
    def get_stage_progress_summary(self) -> Dict[str, Any]:
        """ステージ進行サマリー取得"""
        total_epochs = sum(stage.epochs for stage in self.curriculum_stages)
        completed_epochs = self.current_epoch
        
        stage_summaries = []
        cumulative_epochs = 0
        
        for stage in self.curriculum_stages:
            stage_start = cumulative_epochs
            stage_end = cumulative_epochs + stage.epochs
            
            if self.current_epoch >= stage_end:
                status = "completed"
                progress = 100.0
            elif self.current_epoch >= stage_start:
                status = "in_progress"
                progress = (self.current_epoch - stage_start) / stage.epochs * 100
            else:
                status = "pending"
                progress = 0.0
            
            stage_summaries.append({
                'name': stage.name,
                'status': status,
                'progress': progress,
                'epochs': stage.epochs,
                'description': stage.description
            })
            
            cumulative_epochs = stage_end
        
        return {
            'current_stage': self.curriculum_stages[self.current_stage_idx].name,
            'current_epoch': self.current_epoch,
            'total_epochs': total_epochs,
            'overall_progress': completed_epochs / total_epochs * 100,
            'stages': stage_summaries,
            'performance_history': self.performance_history[-10:]  # 最新10エントリ
        }
    
    def record_performance(self, epoch: int, metrics: Dict[str, float]):
        """パフォーマンス記録"""
        stage_name, _ = self.get_stage_config(epoch)
        
        self.performance_history.append({
            'epoch': epoch,
            'stage': stage_name,
            'metrics': metrics.copy(),
            'timestamp': torch.cuda.Event(enable_timing=True)
        })
    
    def get_curriculum_schedule(self) -> List[Dict[str, Any]]:
        """カリキュラム全体スケジュール取得"""
        schedule = []
        cumulative_epochs = 0
        
        for stage in self.curriculum_stages:
            schedule.append({
                'stage': stage.name,
                'start_epoch': cumulative_epochs,
                'end_epoch': cumulative_epochs + stage.epochs - 1,
                'epochs': stage.epochs,
                'resolution': stage.resolution,
                'difficulty_threshold': stage.difficulty_threshold,
                'description': stage.description
            })
            cumulative_epochs += stage.epochs
        
        return schedule


def create_curriculum_strategy(
    config_override: Optional[Dict[str, Any]] = None
) -> LlamaMultiModalCurriculum:
    """
    カリキュラム学習戦略作成ヘルパー関数
    
    Args:
        config_override: 設定オーバーライド
    
    Returns:
        LlamaMultiModalCurriculum インスタンス
    """
    base_config = {
        'stage1_epochs': 2,
        'stage2_epochs': 3,
        'stage3_epochs': 3,
        'stage4_epochs': 5,
        'base_learning_rate': 1e-4
    }
    
    if config_override:
        base_config.update(config_override)
    
    return LlamaMultiModalCurriculum(base_config)