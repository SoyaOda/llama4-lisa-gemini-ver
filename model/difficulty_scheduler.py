# model/difficulty_scheduler.py
"""
Difficulty-based Data Scheduler for Curriculum Learning

Phase 3C実装: 困難度ベースデータスケジューリング
- Phase 3B OHEM損失を困難度指標として活用
- サンプル困難度の動的評価
- カリキュラム段階に応じたバッチ調整

目標:
- 学習効率化による収束高速化
- 段階的な難易度増加による安定学習
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
import numpy as np
from collections import defaultdict, OrderedDict
import heapq
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DifficultyMetrics:
    """サンプル困難度メトリクス"""
    text_complexity: float = 0.0        # テキスト長、語彙難易度
    visual_complexity: float = 0.0      # SAM2セグメンテーション困難度
    multimodal_alignment: float = 0.0   # モーダル間整合性
    ohem_loss_value: float = 0.0        # OHEM損失値（Phase 3B活用）
    
    def overall_difficulty(self, weights: Optional[Dict[str, float]] = None) -> float:
        """総合困難度スコア計算"""
        if weights is None:
            weights = {
                'text': 0.3,
                'visual': 0.3,
                'multimodal': 0.2,
                'ohem': 0.2
            }
        
        score = (
            weights.get('text', 0.3) * self.text_complexity +
            weights.get('visual', 0.3) * self.visual_complexity +
            weights.get('multimodal', 0.2) * self.multimodal_alignment +
            weights.get('ohem', 0.2) * self.ohem_loss_value
        )
        
        return np.clip(score, 0.0, 1.0)


class DifficultyBasedScheduler:
    """
    困難度ベースデータスケジューリング（Phase 3B OHEM知見活用）
    
    機能:
    - サンプル困難度の動的評価
    - カリキュラム段階に応じたフィルタリング
    - 困難度キャッシュによる効率化
    - バッチ内困難度バランス調整
    """
    
    def __init__(
        self, 
        ohem_loss_fn: Optional[nn.Module] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.ohem_loss_fn = ohem_loss_fn
        self.config = config or {}
        
        # 困難度キャッシュ（メモリ効率化）
        self.difficulty_cache = OrderedDict()
        self.max_cache_size = self.config.get('max_cache_size', 10000)
        
        # 困難度統計
        self.difficulty_stats = {
            'mean': 0.5,
            'std': 0.15,
            'min': 0.0,
            'max': 1.0
        }
        
        # バッチ履歴（学習進行追跡）
        self.batch_history = []
        
        # 困難度評価重み（カスタマイズ可能）
        self.difficulty_weights = self.config.get('difficulty_weights', {
            'text': 0.3,
            'visual': 0.3,
            'multimodal': 0.2,
            'ohem': 0.2
        })
        
        logger.info("DifficultyBasedScheduler初期化完了")
        logger.info(f"  - OHEM損失関数: {'有効' if ohem_loss_fn else '無効'}")
        logger.info(f"  - キャッシュサイズ: {self.max_cache_size}")
    
    def evaluate_sample_difficulty(
        self, 
        sample: Dict[str, Any],
        use_cache: bool = True
    ) -> float:
        """
        サンプル困難度評価（OHEM損失活用）
        
        Args:
            sample: 評価対象サンプル
            use_cache: キャッシュ使用フラグ
        
        Returns:
            困難度スコア (0.0-1.0)
        """
        # サンプルID取得（キャッシュキー）
        sample_id = sample.get('id', self._compute_sample_hash(sample))
        
        # キャッシュチェック
        if use_cache and sample_id in self.difficulty_cache:
            return self.difficulty_cache[sample_id]
        
        # 困難度メトリクス計算
        metrics = DifficultyMetrics()
        
        # 1. テキスト複雑度評価
        metrics.text_complexity = self._evaluate_text_complexity(sample)
        
        # 2. 視覚複雑度評価
        metrics.visual_complexity = self._evaluate_visual_complexity(sample)
        
        # 3. マルチモーダル整合性評価
        metrics.multimodal_alignment = self._evaluate_multimodal_alignment(sample)
        
        # 4. OHEM損失による困難度評価（Phase 3B活用）
        if self.ohem_loss_fn is not None:
            metrics.ohem_loss_value = self._evaluate_ohem_difficulty(sample)
        
        # 総合困難度計算
        difficulty_score = metrics.overall_difficulty(self.difficulty_weights)
        
        # キャッシュ更新
        if use_cache:
            self._update_cache(sample_id, difficulty_score)
        
        # 統計更新
        self._update_difficulty_stats(difficulty_score)
        
        return difficulty_score
    
    def _evaluate_text_complexity(self, sample: Dict[str, Any]) -> float:
        """テキスト複雑度評価"""
        text = sample.get('text', '')
        if isinstance(text, torch.Tensor):
            # トークン化済みの場合
            text_len = text.shape[-1]
        else:
            # 生テキストの場合
            text_len = len(str(text).split())
        
        # 複雑度計算（正規化）
        # 短い: 0-50 words, 中程度: 50-150 words, 長い: 150+ words
        if text_len < 50:
            complexity = text_len / 50 * 0.3
        elif text_len < 150:
            complexity = 0.3 + (text_len - 50) / 100 * 0.4
        else:
            complexity = 0.7 + min((text_len - 150) / 200, 0.3)
        
        return np.clip(complexity, 0.0, 1.0)
    
    def _evaluate_visual_complexity(self, sample: Dict[str, Any]) -> float:
        """視覚複雑度評価"""
        # 画像サイズ
        image = sample.get('image')
        if image is None:
            return 0.5  # デフォルト中程度
        
        if isinstance(image, torch.Tensor):
            # テンソル形式
            if image.dim() == 4:  # batch dimension
                image = image[0]
            height, width = image.shape[-2:]
        else:
            # PIL Image等
            try:
                width, height = image.size
            except:
                return 0.5
        
        # 解像度ベース複雑度
        resolution_score = min((height * width) / (1024 * 1024), 1.0)
        
        # セグメンテーションマスクの複雑度（ある場合）
        mask = sample.get('mask') or sample.get('sam_targets')
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                # マスクの複雑度：エッジピクセル数で評価
                if mask.dim() > 2:
                    mask = mask.squeeze()
                
                # エッジ検出（簡易版）
                edge_v = torch.abs(mask[1:] - mask[:-1]).sum()
                edge_h = torch.abs(mask[:, 1:] - mask[:, :-1]).sum()
                edge_ratio = (edge_v + edge_h) / (height * width)
                
                mask_complexity = min(edge_ratio * 10, 1.0)
            else:
                mask_complexity = 0.5
        else:
            mask_complexity = 0.5
        
        # 総合視覚複雑度
        visual_complexity = 0.5 * resolution_score + 0.5 * mask_complexity
        
        return np.clip(visual_complexity, 0.0, 1.0)
    
    def _evaluate_multimodal_alignment(self, sample: Dict[str, Any]) -> float:
        """マルチモーダル整合性評価"""
        # テキストと画像の両方が必要
        has_text = 'text' in sample and sample['text']
        has_image = 'image' in sample and sample['image'] is not None
        
        if not (has_text and has_image):
            return 0.0  # シングルモーダルは整合性低
        
        # Q-Former特徴がある場合は活用
        if 'qformer_features' in sample:
            # 特徴量の分散を整合性指標として使用
            features = sample['qformer_features']
            if isinstance(features, torch.Tensor):
                feature_std = features.std().item()
                # 分散が大きいほど整合性が低い
                alignment_score = 1.0 - min(feature_std * 2, 1.0)
            else:
                alignment_score = 0.5
        else:
            # デフォルト：中程度の整合性
            alignment_score = 0.5
        
        return np.clip(alignment_score, 0.0, 1.0)
    
    def _evaluate_ohem_difficulty(self, sample: Dict[str, Any]) -> float:
        """OHEM損失による困難度評価（Phase 3B活用）"""
        try:
            with torch.no_grad():
                # 簡易推論による損失計算
                # 実際の実装では、モデルの forward pass が必要
                if 'ohem_loss' in sample:
                    # 事前計算済みの場合
                    ohem_loss = sample['ohem_loss']
                else:
                    # ダミー実装（実際にはモデル推論が必要）
                    ohem_loss = np.random.random() * 2.0  # 0-2の範囲
                
                # 損失を0-1に正規化
                difficulty_score = 1.0 - np.exp(-ohem_loss / 2.0)
                
                return np.clip(difficulty_score, 0.0, 1.0)
                
        except Exception as e:
            logger.warning(f"OHEM困難度評価エラー: {e}")
            return 0.5  # デフォルト中程度
    
    def schedule_curriculum_batch(
        self,
        dataset: List[Dict[str, Any]],
        current_stage: str,
        stage_config: Dict[str, Any],
        batch_size: int = 32,
        balanced: bool = True
    ) -> List[Dict[str, Any]]:
        """
        カリキュラム段階に応じたバッチ調整
        
        Args:
            dataset: データセット
            current_stage: 現在のステージ名
            stage_config: ステージ設定
            batch_size: バッチサイズ
            balanced: 困難度バランス調整フラグ
        
        Returns:
            調整済みバッチデータ
        """
        difficulty_threshold = stage_config.get('difficulty_threshold', 1.0)
        
        # 全サンプルの困難度評価
        sample_difficulties = []
        for idx, sample in enumerate(dataset):
            difficulty = self.evaluate_sample_difficulty(sample)
            if difficulty <= difficulty_threshold:
                sample_difficulties.append((difficulty, idx, sample))
        
        # データが空の場合は、困難度に関係なく最初のバッチサイズ分を使用
        if not sample_difficulties:
            logger.warning(
                f"困難度閾値 {difficulty_threshold} で適合するサンプルがありません。"
                f"最初の {min(batch_size, len(dataset))} サンプルを使用します。"
            )
            sample_difficulties = [
                (self.evaluate_sample_difficulty(sample), idx, sample)
                for idx, sample in enumerate(dataset[:batch_size])
            ]
        
        # 困難度でソート（簡単→難しい）
        sample_difficulties.sort(key=lambda x: x[0])
        
        # バッチ選択
        if balanced and len(sample_difficulties) > batch_size:
            # バランス調整：異なる困難度レベルから均等に選択
            selected_samples = self._balanced_selection(
                sample_difficulties, 
                batch_size,
                num_buckets=4
            )
        else:
            # 単純選択：困難度順
            selected_samples = [s[2] for s in sample_difficulties[:batch_size]]
        
        # 困難度情報を付与
        for i, sample in enumerate(selected_samples):
            if i < len(sample_difficulties):
                sample['_difficulty_score'] = sample_difficulties[i][0]
                sample['_difficulty_rank'] = i
        
        # バッチ履歴記録
        self.batch_history.append({
            'stage': current_stage,
            'threshold': difficulty_threshold,
            'selected_count': len(selected_samples),
            'total_count': len(dataset),
            'avg_difficulty': np.mean([s.get('_difficulty_score', 0.5) for s in selected_samples])
        })
        
        logger.info(f"カリキュラムバッチ作成: {len(selected_samples)}/{len(dataset)} サンプル選択")
        logger.info(f"  - 平均困難度: {self.batch_history[-1]['avg_difficulty']:.3f}")
        
        return selected_samples
    
    def _balanced_selection(
        self,
        sample_difficulties: List[Tuple[float, int, Dict]],
        batch_size: int,
        num_buckets: int = 4
    ) -> List[Dict[str, Any]]:
        """困難度バランス調整選択"""
        # 困難度バケット作成
        buckets = [[] for _ in range(num_buckets)]
        bucket_size = len(sample_difficulties) // num_buckets
        
        for i, (diff, idx, sample) in enumerate(sample_difficulties):
            bucket_idx = min(i // bucket_size, num_buckets - 1)
            buckets[bucket_idx].append(sample)
        
        # 各バケットから均等に選択
        selected = []
        per_bucket = batch_size // num_buckets
        remainder = batch_size % num_buckets
        
        for i, bucket in enumerate(buckets):
            n_select = per_bucket + (1 if i < remainder else 0)
            if bucket:
                # バケット内でランダム選択
                indices = np.random.choice(
                    len(bucket), 
                    size=min(n_select, len(bucket)), 
                    replace=False
                )
                selected.extend([bucket[j] for j in indices])
        
        return selected[:batch_size]
    
    def _compute_sample_hash(self, sample: Dict[str, Any]) -> str:
        """サンプルハッシュ計算（キャッシュキー用）"""
        # 簡易ハッシュ実装
        hash_components = []
        
        if 'text' in sample:
            text = str(sample['text'])[:100]  # 最初の100文字
            hash_components.append(f"text:{hash(text)}")
        
        if 'image' in sample and isinstance(sample['image'], torch.Tensor):
            img_shape = tuple(sample['image'].shape)
            hash_components.append(f"img:{img_shape}")
        
        return "_".join(hash_components)
    
    def _update_cache(self, sample_id: str, difficulty: float):
        """キャッシュ更新（LRU方式）"""
        # 既存エントリを削除
        if sample_id in self.difficulty_cache:
            del self.difficulty_cache[sample_id]
        
        # 新規追加
        self.difficulty_cache[sample_id] = difficulty
        
        # サイズ制限チェック
        if len(self.difficulty_cache) > self.max_cache_size:
            # 最古のエントリを削除（OrderedDictの特性活用）
            self.difficulty_cache.popitem(last=False)
    
    def _update_difficulty_stats(self, difficulty: float):
        """困難度統計更新"""
        # 簡易的なオンライン統計更新
        alpha = 0.01  # 学習率
        
        self.difficulty_stats['mean'] = (
            (1 - alpha) * self.difficulty_stats['mean'] + 
            alpha * difficulty
        )
        
        variance = (difficulty - self.difficulty_stats['mean']) ** 2
        self.difficulty_stats['std'] = np.sqrt(
            (1 - alpha) * self.difficulty_stats['std'] ** 2 + 
            alpha * variance
        )
        
        self.difficulty_stats['min'] = min(self.difficulty_stats['min'], difficulty)
        self.difficulty_stats['max'] = max(self.difficulty_stats['max'], difficulty)
    
    def get_difficulty_distribution(self) -> Dict[str, Any]:
        """困難度分布情報取得"""
        if not self.difficulty_cache:
            return {}
        
        difficulties = list(self.difficulty_cache.values())
        
        return {
            'mean': np.mean(difficulties),
            'std': np.std(difficulties),
            'min': np.min(difficulties),
            'max': np.max(difficulties),
            'percentiles': {
                '25': np.percentile(difficulties, 25),
                '50': np.percentile(difficulties, 50),
                '75': np.percentile(difficulties, 75),
                '90': np.percentile(difficulties, 90)
            },
            'cache_size': len(self.difficulty_cache)
        }
    
    def clear_cache(self):
        """キャッシュクリア"""
        self.difficulty_cache.clear()
        logger.info("困難度キャッシュをクリアしました")
    
    def save_difficulty_stats(self, save_path: str):
        """困難度統計保存"""
        import json
        
        stats = {
            'difficulty_stats': self.difficulty_stats,
            'distribution': self.get_difficulty_distribution(),
            'batch_history': self.batch_history[-100:]  # 最新100バッチ
        }
        
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"困難度統計保存: {save_path}")


def create_difficulty_scheduler(
    ohem_loss_fn: Optional[nn.Module] = None,
    config_override: Optional[Dict[str, Any]] = None
) -> DifficultyBasedScheduler:
    """
    困難度スケジューラー作成ヘルパー関数
    
    Args:
        ohem_loss_fn: OHEM損失関数（Phase 3B）
        config_override: 設定オーバーライド
    
    Returns:
        DifficultyBasedScheduler インスタンス
    """
    base_config = {
        'max_cache_size': 10000,
        'difficulty_weights': {
            'text': 0.3,
            'visual': 0.3,
            'multimodal': 0.2,
            'ohem': 0.2
        }
    }
    
    if config_override:
        base_config.update(config_override)
    
    return DifficultyBasedScheduler(ohem_loss_fn, base_config)