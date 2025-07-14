# model/sam2_integration.py
"""
SAM2統合モジュール: Segment Anything Model 2をQ-Formerと統合

SAM2の主な改善点:
- 6倍高速 + より高精度
- 画像・動画対応（統一アーキテクチャ）
- Transformer + streaming memoryアーキテクチャ
- リアルタイム処理対応（44 FPS）

参考:
- SAM2論文: https://arxiv.org/abs/2408.00714
- HuggingFace SAM2: https://huggingface.co/papers/2408.00714
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import sys
import os
import urllib.request
import hashlib
from pathlib import Path
import tempfile

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_linux

import subprocess
import sys

def check_python_and_sam2():
    """Python版本とSAM2インストール状況を確認"""
    python_version = sys.version_info
    
    if python_version < (3, 10):
        print(f"⚠️ Python {python_version.major}.{python_version.minor}では SAM2 は動作しません")
        print("Python 3.10以上が必要です")
        return False, "python_version_too_old"
    
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return True, "success"
    except ImportError as e:
        return False, str(e)

# Python 3.10環境でのSAM2インポート
sam2_available, sam2_status = check_python_and_sam2()

if sam2_available:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
    print("✅ Meta公式SAM2利用可能 (Python 3.10+)")
else:
    SAM2_AVAILABLE = False
    print("❌ Meta公式SAM2が利用できません")
    if sam2_status == "python_version_too_old":
        print("解決策: pyenv local 3.10.12 でPython 3.10に切り替えてください")
    else:
        print("解決策: Python 3.10環境で pip install sam2 を実行してください")
        print(f"エラー詳細: {sam2_status}")


class SAM2Wrapper(nn.Module):
    """
    Meta公式SAM2ラッパー (HuggingFace Hub自動取得)
    
    重みの手動管理不要、Web自動取得:
    - facebook/sam2-hiera-large (推奨)
    - facebook/sam2-hiera-base
    - facebook/sam2-hiera-small
    - facebook/sam2-hiera-tiny
    """
    
    def __init__(
        self, 
        model_id: str = "facebook/sam2-hiera-large",
        device: str = "cuda",
        target_dtype: torch.dtype = torch.bfloat16,
        debug_mode: bool = True
    ):
        super().__init__()
        
        if not SAM2_AVAILABLE:
            raise ImportError(
                "Meta公式SAM2が利用できません。\n"
                "必須インストール: pip install segment-anything-2\n"
                "また、huggingface_hubも必要: pip install huggingface_hub"
            )
        
        # 🔄 2025年ベストプラクティス: 即座データ型統一設定
        self.device = "cuda"
        self.model_id = model_id
        self.debug_mode = debug_mode
        # GPU専用デバイス・データ型管理
        self._target_device = self.device
        self._target_dtype = target_dtype
        
        if debug_mode:
            print(f"  🔧 SAM2Wrapper設定:")
            print(f"    - target_dtype: {target_dtype}")
            print(f"    - debug_mode: {debug_mode}")
        
        print(f"🔄 Meta公式SAM2初期化中...")
        print(f"  - モデルID: {model_id}")
        print(f"  - デバイス: {self.device}")
        print(f"  - 自動取得: HuggingFace Hub")
        
        try:
            # 🔄 Meta公式SAM2 (自動重み取得)
            self.predictor = SAM2ImagePredictor.from_pretrained(
                model_id,
                device=self.device
            )
            
            print("✅ Meta公式SAM2初期化成功")
            print("✅ 重みファイル自動取得完了")
            
        except Exception as e:
            error_msg = (
                f"❌ Meta公式SAM2初期化失敗: {e}\n"
                "必須要件:\n"
                "1. pip install segment-anything-2\n"
                "2. pip install huggingface_hub\n" 
                "3. インターネット接続\n"
                "4. HuggingFace Hub アクセス可能"
            )
            print(error_msg)
            raise RuntimeError(error_msg) from e
    
    def set_image(self, image) -> None:
        """
        画像をSAM2に設定 (公式API + 2025年BFloat16対応)
        
        Args:
            image: numpy array (H, W, 3) RGB画像 [0, 255] またはPIL Image
        """
        # PyTorchテンソルをnumpy配列に変換（SAM2要求形式）
        # 2025年ベストプラクティス: BFloat16 → Float32 → uint8変換
        if isinstance(image, torch.Tensor):
            # Web調査結果: BFloat16のNumPy非対応のためFloat32へ先に変換
            if image.dtype == torch.bfloat16:
                image = image.to(torch.float32)
            
            image_np = image.detach().cpu().numpy().astype('uint8')
        else:
            image_np = image
            
        # 🔄 Meta公式SAM2 API
        self.predictor.set_image(image_np)
    
    def predict_with_prompts(
        self,
        prompt_embeddings: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        multimask_output: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Q-Formerプロンプトでセグメンテーション実行 (Meta公式API)
        
        Args:
            prompt_embeddings: (num_queries, embed_dim) Q-Formerからの埋め込み
            point_coords: (num_points, 2) ポイント座標（オプション）
            point_labels: (num_points,) ポイントラベル（オプション）
            boxes: (num_boxes, 4) バウンディングボックス（オプション）
            multimask_output: 複数マスク出力するかどうか
            
        Returns:
            Dict containing:
                - masks: (num_masks, H, W) 予測マスク
                - iou_predictions: (num_masks,) IoU予測値
                - low_res_logits: (num_masks, H//4, W//4) 低解像度ロジット
        """
        
        # SAM2では現在prompt_embeddingsを直接サポートしていないため、
        # 代替プロンプト形式を使用
        
        if point_coords is not None or boxes is not None:
            # 既存のプロンプト形式がある場合
            # 2025年ベストプラクティス: BFloat16 → Float32変換対応
            point_coords_np = None
            if point_coords is not None:
                coords_tensor = point_coords.to(torch.float32) if point_coords.dtype == torch.bfloat16 else point_coords
                point_coords_np = coords_tensor.detach().cpu().numpy()
            
            point_labels_np = None
            if point_labels is not None:
                labels_tensor = point_labels.to(torch.float32) if point_labels.dtype == torch.bfloat16 else point_labels
                point_labels_np = labels_tensor.detach().cpu().numpy()
            
            box_np = None
            if boxes is not None:
                box_tensor = boxes[0].to(torch.float32) if boxes[0].dtype == torch.bfloat16 else boxes[0]
                box_np = box_tensor.detach().cpu().numpy()
            
            masks, iou_predictions, low_res_logits = self.predictor.predict(
                point_coords=point_coords_np,
                point_labels=point_labels_np,
                box=box_np,
                multimask_output=multimask_output
            )
        else:
            # Q-Formerプロンプトから推定座標生成（簡易版）
            # 将来: prompt_embeddingsを座標に変換するネットワーク追加
            num_queries = prompt_embeddings.shape[0]
            
            # 複数クエリから複数ポイント生成
            grid_size = int(num_queries ** 0.5) or 2
            H, W = 1024, 1024  # SAM2解像度
            
            points = []
            labels = []
            
            for i in range(min(num_queries, 4)):  # 最大4ポイント
                x = (i % grid_size + 1) * W // (grid_size + 1)
                y = (i // grid_size + 1) * H // (grid_size + 1)
                points.append([x, y])
                labels.append(1)  # 前景
            
            point_coords_np = torch.tensor(points, dtype=torch.float32).numpy()
            point_labels_np = torch.tensor(labels, dtype=torch.int32).numpy()
            
            # 🔄 Meta公式SAM2 API
            masks, iou_predictions, low_res_logits = self.predictor.predict(
                point_coords=point_coords_np,
                point_labels=point_labels_np,
                multimask_output=multimask_output
            )
        
        # numpy配列をPyTorchテンソルに変換（GPU専用環境 + 2025年ベストプラクティス）
        device = getattr(self, '_target_device', 'cuda')
        target_dtype = getattr(self, '_target_dtype', torch.bfloat16)
        
        # 🔄 2025年ベストプラクティス: 即座データ型統一（Float32混入防止）
        masks = torch.from_numpy(masks).to(device=device, dtype=target_dtype)
        iou_predictions = torch.from_numpy(iou_predictions).to(device=device, dtype=target_dtype) 
        low_res_logits = torch.from_numpy(low_res_logits).to(device=device, dtype=target_dtype)
        
        # デバッグ情報
        print(f"  📊 SAM2出力統計:")
        print(f"    - masks: {masks.shape}, {masks.dtype}, device: {masks.device}")
        print(f"    - iou_predictions: {iou_predictions.shape}, {iou_predictions.dtype}")
        print(f"    - 平均IoU: {iou_predictions.mean().item():.3f}")
        
        return {
            'masks': masks,
            'iou_predictions': iou_predictions, 
            'low_res_logits': low_res_logits
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """SAM2モデル情報取得"""
        try:
            # SAM2ImagePredictorから実際のモデルにアクセス
            actual_model = self.predictor.model if hasattr(self.predictor, 'model') else None
            
            if actual_model:
                parameters = sum(p.numel() for p in actual_model.parameters())
                image_encoder_type = type(actual_model.image_encoder).__name__ if hasattr(actual_model, 'image_encoder') else 'Unknown'
                mask_decoder_type = type(actual_model.mask_decoder).__name__ if hasattr(actual_model, 'mask_decoder') else 'Unknown'
            else:
                parameters = 0
                image_encoder_type = 'Unknown'
                mask_decoder_type = 'Unknown'
                
            return {
                'model_type': 'Meta SAM2',
                'model_id': self.model_id,
                'device': self.device,
                'parameters': parameters,
                'image_encoder': image_encoder_type,
                'mask_decoder': mask_decoder_type,
            }
        except Exception as e:
            return {
                'model_type': 'Meta SAM2',
                'model_id': self.model_id,
                'device': self.device,
                'parameters': 'Unknown',
                'error': str(e)
            }


# MockSAM2Wrapper削除 - Meta公式API一本化


def get_sam2_wrapper(
    model_id: str = "facebook/sam2-hiera-large", 
    target_dtype: torch.dtype = torch.bfloat16,
    debug_mode: bool = True,
    **kwargs
) -> SAM2Wrapper:
    """
    Meta公式SAM2ファクトリ関数 (フォールバック無し)
    
    Args:
        model_id: HuggingFace Hub上のSAM2モデルID
                 - facebook/sam2-hiera-large (推奨, 最高精度)
                 - facebook/sam2-hiera-base
                 - facebook/sam2-hiera-small  
                 - facebook/sam2-hiera-tiny
        target_dtype: 出力テンソルのデータ型 (デフォルト: BFloat16)
        **kwargs: その他のパラメータ
        
    Returns:
        SAM2Wrapper (Meta公式のみ)
        
    Raises:
        ImportError: SAM2が利用できない場合
        RuntimeError: SAM2初期化に失敗した場合
    """
    if not SAM2_AVAILABLE:
        raise ImportError(
            "Meta公式SAM2が利用できません。\n"
            "必須インストール:\n"
            "1. pip install segment-anything-2\n"
            "2. pip install huggingface_hub"
        )
    
    print(f"🔄 Meta公式SAM2を使用: {model_id}")
    print(f"  - データ型統一: {target_dtype}")
    print(f"  - デバッグモード: {debug_mode}")
    return SAM2Wrapper(model_id=model_id, target_dtype=target_dtype, debug_mode=debug_mode, **kwargs)


def test_sam2_integration():
    """Meta公式SAM2統合テスト (フォールバック無し)"""
    print("=== Meta公式SAM2統合テスト ===")
    
    try:
        # Meta公式SAM2初期化 (エラー時は例外発生)
        sam2 = get_sam2_wrapper()
        
        # モデル情報表示
        info = sam2.get_model_info()
        print(f"\nSAM2モデル情報:")
        for key, value in info.items():
            print(f"  - {key}: {value}")
        
        # ダミー画像とプロンプト作成
        image = torch.randint(0, 255, (1024, 1024, 3), dtype=torch.uint8)
        prompt_embeddings = torch.randn(8, 256)  # Q-Formerからの8個のクエリ
        
        print(f"\nテストデータ:")
        print(f"  - 画像: {image.shape}")
        print(f"  - プロンプト埋め込み: {prompt_embeddings.shape}")
        
        # 画像設定
        sam2.set_image(image)
        
        # セグメンテーション実行
        with torch.no_grad():
            results = sam2.predict_with_prompts(prompt_embeddings)
        
        print(f"\n結果:")
        print(f"  - マスク: {results['masks'].shape}")
        print(f"  - IoU予測: {results['iou_predictions'].shape}")
        print(f"  - 低解像度ロジット: {results['low_res_logits'].shape}")
        
        # マスク品質確認
        masks = results['masks']
        print(f"\nマスク品質:")
        print(f"  - 平均IoU: {results['iou_predictions'].mean().item():.3f}")
        print(f"  - マスク平均値: {masks.mean().item():.3f}")
        print(f"  - マスク標準偏差: {masks.std().item():.3f}")
        
        print("\n✅ Meta公式SAM2統合テスト完了")
        
    except (ImportError, RuntimeError) as e:
        print(f"\n❌ Meta公式SAM2テスト失敗: {e}")
        print("Lambda Cloud環境での実行を推奨")


if __name__ == "__main__":
    test_sam2_integration()