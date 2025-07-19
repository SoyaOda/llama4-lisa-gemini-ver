# model/dataset_adapter.py
"""
HybridDataset（シングルエンコーダー構成）とQFormerSegmentationBridgeの
インターフェースを調整するアダプター

使用例:
-------
1. 基本的な使用方法：
   ```python
   from model.dataset_adapter import adapt_dataset_for_qformer
   
   # HybridDatasetからのバッチ
   batch = {
       'sam_pixel_values': torch.randn(32, 3, 1024, 1024),
       'input_ids': torch.tensor(...),
       'attention_mask': torch.tensor(...),
   }
   
   # QFormerSegmentationBridge用に変換
   adapted_batch = adapt_dataset_for_qformer(batch)
   
   # QFormerSegmentationBridgeで使用
   outputs = qformer_bridge(
       images=adapted_batch['images'],
       input_ids=adapted_batch['input_ids'],
       attention_mask=adapted_batch['attention_mask']
   )
   ```

2. DataLoaderの変換：
   ```python
   from model.dataset_adapter import create_qformer_compatible_dataloader
   
   # 既存のDataLoader
   original_loader = DataLoader(hybrid_dataset, ...)
   
   # QFormer互換DataLoaderに変換
   qformer_loader = create_qformer_compatible_dataloader(original_loader)
   
   # 通常通り使用
   for batch in qformer_loader:
       outputs = qformer_bridge(**batch)
   ```
"""

import torch
from typing import Dict, Any


def adapt_dataset_for_qformer(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    HybridDatasetのシングルエンコーダー構成出力を
    QFormerSegmentationBridgeの期待する形式に変換
    
    Args:
        batch: HybridDatasetからのバッチデータ
            - sam_pixel_values: SAM用画像データ
            - input_ids: テキスト入力ID
            - attention_mask: アテンションマスク
            - labels: ラベル
            
    Returns:
        QFormerSegmentationBridge用に変換されたバッチ
            - images: sam_pixel_valuesをリネーム
            - その他のフィールドはそのまま
    """
    adapted_batch = batch.copy()
    
    # シングルエンコーダー構成：sam_pixel_valuesをimagesとして使用
    if 'sam_pixel_values' in batch and 'pixel_values' not in batch:
        adapted_batch['images'] = batch['sam_pixel_values']
        # 元のキーも保持（互換性のため）
        adapted_batch['sam_pixel_values'] = batch['sam_pixel_values']
        
        print(f"✓ データセットアダプター: sam_pixel_values → images 変換完了")
        print(f"  - 画像形状: {adapted_batch['images'].shape}")
    
    # pixel_valuesが存在する場合（デュアルストリーム構成）
    elif 'pixel_values' in batch:
        adapted_batch['images'] = batch['pixel_values']
        print(f"✓ データセットアダプター: pixel_values → images 変換完了")
        print(f"  - 画像形状: {adapted_batch['images'].shape}")
    
    # どちらも存在しない場合はエラー
    else:
        raise ValueError(
            "バッチデータに画像データが含まれていません。"
            "'sam_pixel_values' または 'pixel_values' が必要です。"
        )
    
    return adapted_batch


def create_qformer_compatible_dataloader(original_dataloader):
    """
    既存のDataLoaderをQFormerSegmentationBridge互換に変換
    
    Args:
        original_dataloader: HybridDatasetを使用したDataLoader
        
    Returns:
        QFormerSegmentationBridge互換のDataLoader（ラッパー）
    """
    class QFormerCompatibleDataLoader:
        def __init__(self, dataloader):
            self.dataloader = dataloader
            
        def __iter__(self):
            for batch in self.dataloader:
                yield adapt_dataset_for_qformer(batch)
                
        def __len__(self):
            return len(self.dataloader)
        
        @property
        def batch_size(self):
            return self.dataloader.batch_size
        
        @property
        def dataset(self):
            return self.dataloader.dataset
    
    return QFormerCompatibleDataLoader(original_dataloader)


# Option B実装：SAM2をビジョンエンコーダーとして使用する場合の設定
def configure_sam_as_vision_encoder(config):
    """
    SAM2をビジョンエンコーダーとして使用するための設定を追加
    
    Args:
        config: LlamaQFormerSAM2Config インスタンス
        
    Returns:
        更新されたconfig
    """
    # SAM2をビジョンエンコーダーとして使用
    config.use_sam_as_vision_encoder = True
    
    # Q-FormerがSAM2の特徴量を直接使用するための設定
    config.sam_feature_extraction = True
    config.skip_llama_vision_processing = True
    
    print("✓ SAM2ビジョンエンコーダー設定完了")
    print("  - use_sam_as_vision_encoder: True")
    print("  - sam_feature_extraction: True")
    print("  - skip_llama_vision_processing: True")
    
    return config