# model/losses_qformer_sam2.py
"""
LISA-Llama4-Scout + Q-Former + SAM2 統合モデル専用損失関数モジュール

Web調査ベース2025年最新ベストプラクティス:
- Focal Tversky Loss: 複雑セグメンテーションに最適（Web推奨）
- Lovász-Softmax Loss: IoU直接最適化
- Q-Former専用損失: マルチモーダル特徴学習
- SAM2統合損失: 高精度プロンプト最適化
- 段階的学習プロトコル対応

参考:
- Focal Tversky: "presents the most promising segmentation results" (2025)
- Lovász-Softmax: "better mIoU scores compared to cross-entropy" (2025)
- Q-Former: BLIP-2準拠多重学習目標
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple
import math


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss (Web調査推奨: 最高性能セグメンテーション損失)
    
    "Focal Tversky Loss presents the most promising segmentation results,
    correctly identifying all organs with only minor variations"
    
    Args:
        alpha: False Negative重み (デフォルト: 0.3)
        beta: False Positive重み (デフォルト: 0.7)
        gamma: Focal重み (デフォルト: 2.0)
        smooth: 数値安定化 (デフォルト: 1e-6)
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 2.0, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 予測マスク (B, H, W) または (B, 1, H, W)
            target: 正解マスク (B, H, W) または (B, 1, H, W)
        """
        # 次元統一
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
            
        # Sigmoid適用
        pred = torch.sigmoid(pred)
        
        # フラット化
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Tversky係数計算
        tp = (pred_flat * target_flat).sum()
        fp = ((1 - target_flat) * pred_flat).sum()
        fn = (target_flat * (1 - pred_flat)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        # Focal適用
        focal_tversky = torch.pow(1 - tversky, self.gamma)
        
        return focal_tversky


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax Loss (IoU直接最適化、Web調査推奨)
    
    "The Lovász-Softmax loss has better mIoU scores compared to training 
    with cross-entropy loss"
    """
    
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 予測ロジット (B, C, H, W) または (B, H, W)
            target: 正解ラベル (B, H, W)
        """
        if pred.dim() == 3:
            # バイナリセグメンテーション
            pred = pred.unsqueeze(1)  # (B, 1, H, W)
            
        batch_size = pred.size(0)
        total_loss = 0
        
        for i in range(batch_size):
            loss_i = self._lovasz_softmax_single(pred[i], target[i])
            total_loss += loss_i
            
        return total_loss / batch_size
    
    def _lovasz_softmax_single(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """単一サンプルのLovász-Softmax損失"""
        if pred.dim() == 3 and pred.size(0) == 1:
            # バイナリケース
            pred_flat = pred.view(-1)
            target_flat = target.view(-1)
            
            # 有効ピクセルマスク
            valid = target_flat != self.ignore_index
            pred_flat = pred_flat[valid]
            target_flat = target_flat[valid]
            
            # Lovász拡張
            signs = 2.0 * target_flat.float() - 1.0
            errors = 1.0 - pred_flat * signs
            errors_sorted, perm = torch.sort(errors, descending=True)
            gt_sorted = target_flat[perm]
            
            grad = self._lovasz_grad(gt_sorted)
            loss = torch.dot(F.relu(errors_sorted), grad)
            
            return loss
        else:
            # マルチクラス（今回は使用しないが将来対応）
            return torch.tensor(0.0, device=pred.device)
    
    def _lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
        """Lovász勾配計算"""
        p = len(gt_sorted)
        gts = gt_sorted.sum().float()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard


class QFormerMultiModalLoss(nn.Module):
    """
    Q-Former専用マルチモーダル損失 (BLIP-2準拠)
    
    3つの学習目標:
    1. ITC: Image-Text Contrastive Learning
    2. ITM: Image-Text Matching
    3. ITG: Image-Grounded Text Generation
    """
    
    def __init__(self, temperature: float = 0.07, itm_weight: float = 1.0, itg_weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.itm_weight = itm_weight
        self.itg_weight = itg_weight
        
    def forward(
        self, 
        query_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        itm_logits: Optional[torch.Tensor] = None,
        itm_labels: Optional[torch.Tensor] = None,
        itg_logits: Optional[torch.Tensor] = None,
        itg_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            query_embeds: Q-Formerクエリ埋め込み (B, num_queries, hidden_size)
            text_embeds: テキスト埋め込み (B, hidden_size)
            itm_logits: ITMロジット (B, 2)
            itm_labels: ITMラベル (B,)
            itg_logits: ITGロジット (B, seq_len, vocab_size)
            itg_labels: ITGラベル (B, seq_len)
        """
        losses = {}
        
        # 1. ITC Loss (Image-Text Contrastive)
        itc_loss = self._compute_itc_loss(query_embeds, text_embeds)
        losses['itc_loss'] = itc_loss
        
        # 2. ITM Loss (Image-Text Matching)
        if itm_logits is not None and itm_labels is not None:
            itm_loss = F.cross_entropy(itm_logits, itm_labels)
            losses['itm_loss'] = itm_loss * self.itm_weight
        
        # 3. ITG Loss (Image-Grounded Text Generation)
        if itg_logits is not None and itg_labels is not None:
            itg_loss = F.cross_entropy(
                itg_logits.view(-1, itg_logits.size(-1)),
                itg_labels.view(-1),
                ignore_index=-100
            )
            losses['itg_loss'] = itg_loss * self.itg_weight
        
        return losses
    
    def _compute_itc_loss(self, query_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """ITC損失計算（対照学習）"""
        # クエリ埋め込みを平均化
        image_feat = query_embeds.mean(dim=1)  # (B, hidden_size)
        text_feat = text_embeds  # (B, hidden_size)
        
        # 正規化
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        
        # 類似度行列
        sim_matrix = torch.matmul(image_feat, text_feat.t()) / self.temperature
        
        # ラベル (対角線が正例)
        batch_size = sim_matrix.size(0)
        labels = torch.arange(batch_size, device=sim_matrix.device)
        
        # 双方向損失
        loss_i2t = F.cross_entropy(sim_matrix, labels)
        loss_t2i = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2.0


class SAM2PromptOptimizationLoss(nn.Module):
    """
    SAM2プロンプト最適化損失
    
    高品質なプロンプト生成を促進:
    1. プロンプト多様性損失
    2. プロンプト-マスク整合性損失
    3. プロンプト安定性損失
    """
    
    def __init__(self, diversity_weight: float = 0.1, consistency_weight: float = 1.0, stability_weight: float = 0.1):
        super().__init__()
        self.diversity_weight = diversity_weight
        self.consistency_weight = consistency_weight
        self.stability_weight = stability_weight
        
    def forward(
        self,
        sam_prompts: torch.Tensor,
        predicted_masks: torch.Tensor,
        target_masks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            sam_prompts: SAMプロンプト (B, num_queries, prompt_dim)
            predicted_masks: 予測マスク (B, num_queries, H, W)
            target_masks: 正解マスク (B, H, W)
        """
        losses = {}
        
        # 1. プロンプト多様性損失
        diversity_loss = self._compute_diversity_loss(sam_prompts)
        losses['prompt_diversity_loss'] = diversity_loss * self.diversity_weight
        
        # 2. プロンプト-マスク整合性損失
        consistency_loss = self._compute_consistency_loss(sam_prompts, predicted_masks, target_masks)
        losses['prompt_consistency_loss'] = consistency_loss * self.consistency_weight
        
        # 3. プロンプト安定性損失
        stability_loss = self._compute_stability_loss(sam_prompts)
        losses['prompt_stability_loss'] = stability_loss * self.stability_weight
        
        return losses
    
    def _compute_diversity_loss(self, sam_prompts: torch.Tensor) -> torch.Tensor:
        """プロンプト多様性損失（クエリ間の類似度を下げる）"""
        batch_size, num_queries, prompt_dim = sam_prompts.shape
        
        # 正規化
        normalized_prompts = F.normalize(sam_prompts, dim=-1)
        
        # クエリ間類似度行列
        sim_matrix = torch.bmm(normalized_prompts, normalized_prompts.transpose(1, 2))
        
        # 対角線を除去（自己類似度は1で固定）
        mask = torch.eye(num_queries, device=sam_prompts.device).unsqueeze(0).expand(batch_size, -1, -1)
        sim_matrix = sim_matrix * (1 - mask)
        
        # 類似度の2乗平均（多様性を促進）
        diversity_loss = (sim_matrix ** 2).mean()
        
        return diversity_loss
    
    def _compute_consistency_loss(
        self, 
        sam_prompts: torch.Tensor, 
        predicted_masks: torch.Tensor, 
        target_masks: torch.Tensor
    ) -> torch.Tensor:
        """プロンプト-マスク整合性損失"""
        batch_size, num_queries = predicted_masks.shape[:2]
        
        # 各クエリマスクと正解マスクのIoU計算
        target_masks = target_masks.unsqueeze(1).expand(-1, num_queries, -1, -1)
        
        # IoU計算
        intersection = (predicted_masks * target_masks).sum(dim=(2, 3))
        union = ((predicted_masks + target_masks) > 0).float().sum(dim=(2, 3))
        iou = intersection / (union + 1e-6)
        
        # 最高IoUクエリを特定
        best_query_indices = iou.argmax(dim=1)
        
        # 最高性能クエリのプロンプトを強化
        best_prompts = sam_prompts[torch.arange(batch_size), best_query_indices]
        
        # プロンプトの正規化とIoUの相関を促進
        prompt_norms = torch.norm(best_prompts, dim=-1)
        best_ious = iou[torch.arange(batch_size), best_query_indices]
        
        # 高IoUクエリは高ノルムプロンプトを持つべき
        consistency_loss = F.mse_loss(prompt_norms, best_ious)
        
        return consistency_loss
    
    def _compute_stability_loss(self, sam_prompts: torch.Tensor) -> torch.Tensor:
        """プロンプト安定性損失（過度な変動を防ぐ）"""
        # プロンプトの標準偏差を制限
        prompt_std = sam_prompts.std(dim=1).mean()
        
        # 適度な安定性を促進（完全な均一化は避ける）
        target_std = 0.5
        stability_loss = F.mse_loss(prompt_std, torch.tensor(target_std, device=sam_prompts.device))
        
        return stability_loss


class CompositeLossQFormerSAM2(nn.Module):
    """
    統合複合損失 (Web調査ベース最適構成)
    
    段階的学習プロトコル対応:
    - Stage 1: インターフェースアライメント (QFormer + 基本セグメンテーション)
    - Stage 2: フルスタック適応 (+ SAM2プロンプト最適化)
    - Stage 3: 推論能力専門化 (+ 高度損失)
    """
    
    def __init__(
        self,
        # 基本セグメンテーション損失重み
        focal_tversky_weight: float = 1.0,
        lovasz_weight: float = 0.5,
        dice_weight: float = 0.3,
        
        # マルチモーダル損失重み
        qformer_weight: float = 0.2,
        
        # SAM2最適化損失重み
        sam2_prompt_weight: float = 0.1,
        
        # 段階的学習設定
        stage: int = 1  # 1: Interface, 2: Full-stack, 3: Reasoning
    ):
        super().__init__()
        
        # 基本セグメンテーション損失（Web調査推奨）
        self.focal_tversky = FocalTverskyLoss()
        self.lovasz = LovaszSoftmaxLoss()
        self.dice = nn.BCEWithLogitsLoss()
        
        # マルチモーダル損失
        self.qformer_loss = QFormerMultiModalLoss()
        
        # SAM2最適化損失
        self.sam2_prompt_loss = SAM2PromptOptimizationLoss()
        
        # 重み設定
        self.focal_tversky_weight = focal_tversky_weight
        self.lovasz_weight = lovasz_weight
        self.dice_weight = dice_weight
        self.qformer_weight = qformer_weight
        self.sam2_prompt_weight = sam2_prompt_weight
        
        # 段階的学習段階
        self.stage = stage
        
    def forward(
        self,
        # セグメンテーション関連
        predicted_masks: torch.Tensor,
        target_masks: torch.Tensor,
        
        # Q-Former関連
        query_embeds: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        
        # SAM2関連
        sam_prompts: Optional[torch.Tensor] = None,
        
        # その他
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        統合損失計算（2025年ベストプラクティス: デバイス統一）
        
        Args:
            predicted_masks: 予測マスク (B, H, W) または (B, num_queries, H, W)
            target_masks: 正解マスク (B, H, W)
            query_embeds: Q-Formerクエリ埋め込み (B, num_queries, hidden_size)
            text_embeds: テキスト埋め込み (B, hidden_size)
            sam_prompts: SAMプロンプト (B, num_queries, prompt_dim)
        """
        # 2025年ベストプラクティス: デバイス・データ型統一
        target_device = predicted_masks.device
        
        # 特別データ型検出: SAM2出力がFloat32の場合のBFloat16変換
        target_dtype = torch.bfloat16  # Llama-4統一基準
        
        # predicted_masksのデータ型確認・変換
        if predicted_masks.dtype != target_dtype:
            print(f"  🔄 損失計算: {predicted_masks.dtype} → {target_dtype} 変換")
            predicted_masks = predicted_masks.to(dtype=target_dtype)
        
        # デバイス・データ型統一（GPU優先 + BFloat16統一）
        target_masks = target_masks.to(device=target_device, dtype=target_dtype)
        if query_embeds is not None:
            query_embeds = query_embeds.to(device=target_device, dtype=target_dtype)
        if text_embeds is not None:
            text_embeds = text_embeds.to(device=target_device, dtype=target_dtype)
        if sam_prompts is not None:
            sam_prompts = sam_prompts.to(device=target_device, dtype=target_dtype)
        
        losses = {}
        total_loss = 0.0
        
        # 1. 基本セグメンテーション損失（全段階で使用）
        if predicted_masks.dim() == 4:
            # 複数クエリマスクの場合、最良マスクを選択
            num_queries = predicted_masks.size(1)
            target_expanded = target_masks.unsqueeze(1).expand(-1, num_queries, -1, -1)
            
            # 各クエリのDice係数を計算
            dice_scores = []
            for i in range(num_queries):
                pred_i = torch.sigmoid(predicted_masks[:, i])
                dice_i = 1 - self._compute_dice(pred_i, target_masks)
                dice_scores.append(dice_i)
            
            # 最良クエリを選択
            dice_scores = torch.stack(dice_scores, dim=0)
            best_idx = dice_scores.argmin(dim=0)
            best_masks = predicted_masks[torch.arange(predicted_masks.size(0)), best_idx]
        else:
            best_masks = predicted_masks
        
        # Focal Tversky Loss (Web推奨最高性能)
        focal_tversky_loss = self.focal_tversky(best_masks, target_masks)
        losses['focal_tversky_loss'] = focal_tversky_loss
        total_loss += focal_tversky_loss * self.focal_tversky_weight
        
        # Lovász-Softmax Loss (IoU直接最適化)
        lovasz_loss = self.lovasz(best_masks, target_masks)
        losses['lovasz_loss'] = lovasz_loss
        total_loss += lovasz_loss * self.lovasz_weight
        
        # Dice Loss (バランス)
        dice_loss = self.dice(best_masks, target_masks.float())
        losses['dice_loss'] = dice_loss
        total_loss += dice_loss * self.dice_weight
        
        # 2. Stage 1以降: Q-Former マルチモーダル損失
        if self.stage >= 1 and query_embeds is not None and text_embeds is not None:
            qformer_losses = self.qformer_loss(query_embeds, text_embeds)
            for key, value in qformer_losses.items():
                losses[f'qformer_{key}'] = value
                total_loss += value * self.qformer_weight
        
        # 3. Stage 2以降: SAM2プロンプト最適化損失
        if self.stage >= 2 and sam_prompts is not None and predicted_masks.dim() == 4:
            sam2_losses = self.sam2_prompt_loss(sam_prompts, predicted_masks, target_masks)
            for key, value in sam2_losses.items():
                losses[f'sam2_{key}'] = value
                total_loss += value * self.sam2_prompt_weight
        
        losses['total_loss'] = total_loss
        return losses
    
    def _compute_dice(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """Dice係数計算"""
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
        
        return dice.mean()
    
    def set_stage(self, stage: int):
        """学習段階を設定"""
        self.stage = stage
        print(f"🔄 損失関数段階を Stage {stage} に設定")


# ファクトリー関数
def get_composite_loss_qformer_sam2(
    stage: int = 1,
    device: str = 'cuda'
) -> CompositeLossQFormerSAM2:
    """
    統合複合損失のファクトリー関数
    
    Args:
        stage: 学習段階 (1: Interface, 2: Full-stack, 3: Reasoning)
        device: デバイス
        
    Returns:
        CompositeLossQFormerSAM2: 設定済み複合損失
    """
    
    # 段階別重み設定（Web調査ベース）
    if stage == 1:
        # Stage 1: インターフェースアライメント重視
        config = {
            'focal_tversky_weight': 1.0,
            'lovasz_weight': 0.5,
            'dice_weight': 0.3,
            'qformer_weight': 0.5,  # Q-Former学習重視
            'sam2_prompt_weight': 0.0,  # SAM2凍結
            'stage': 1
        }
    elif stage == 2:
        # Stage 2: フルスタック適応
        config = {
            'focal_tversky_weight': 1.0,
            'lovasz_weight': 0.5,
            'dice_weight': 0.3,
            'qformer_weight': 0.2,
            'sam2_prompt_weight': 0.3,  # SAM2最適化開始
            'stage': 2
        }
    else:  # stage == 3
        # Stage 3: 推論能力専門化
        config = {
            'focal_tversky_weight': 1.0,
            'lovasz_weight': 0.8,  # IoU最適化強化
            'dice_weight': 0.2,
            'qformer_weight': 0.1,
            'sam2_prompt_weight': 0.5,  # SAM2最適化最大
            'stage': 3
        }
    
    loss_fn = CompositeLossQFormerSAM2(**config)
    loss_fn = loss_fn.to(device)
    
    print(f"✅ Stage {stage} 複合損失初期化完了")
    print(f"  - Focal Tversky (最高性能): {config['focal_tversky_weight']}")
    print(f"  - Lovász-Softmax (IoU): {config['lovasz_weight']}")
    print(f"  - Q-Former (マルチモーダル): {config['qformer_weight']}")
    print(f"  - SAM2プロンプト: {config['sam2_prompt_weight']}")
    
    return loss_fn


# テスト関数
def test_composite_loss():
    """複合損失のテスト"""
    print("=== CompositeLossQFormerSAM2 テスト ===")
    
    # テスト用データ
    batch_size = 2
    num_queries = 64
    height, width = 256, 256
    hidden_size = 5120
    prompt_dim = 256
    
    # ダミーデータ作成
    predicted_masks = torch.randn(batch_size, num_queries, height, width)
    target_masks = torch.randint(0, 2, (batch_size, height, width)).float()
    query_embeds = torch.randn(batch_size, num_queries, hidden_size)
    text_embeds = torch.randn(batch_size, hidden_size)
    sam_prompts = torch.randn(batch_size, num_queries, prompt_dim)
    
    # 各段階テスト
    for stage in [1, 2, 3]:
        print(f"\n--- Stage {stage} テスト ---")
        
        loss_fn = get_composite_loss_qformer_sam2(stage=stage, device='cuda')
        
        with torch.no_grad():
            losses = loss_fn(
                predicted_masks=predicted_masks,
                target_masks=target_masks,
                query_embeds=query_embeds,
                text_embeds=text_embeds,
                sam_prompts=sam_prompts
            )
        
        print(f"  総損失: {losses['total_loss'].item():.4f}")
        for key, value in losses.items():
            if key != 'total_loss':
                print(f"  - {key}: {value.item():.4f}")
    
    print("\n✅ 複合損失テスト完了")


if __name__ == "__main__":
    test_composite_loss()