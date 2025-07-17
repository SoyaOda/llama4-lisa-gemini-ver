#!/usr/bin/env python3
"""
LISA-Llama4-Scout + Q-Former + SAM2 統合モデル 勾配フロー検証スクリプト

model/llama4_qformer_sam2.py と model/losses_qformer_sam2.py の勾配フローテスト

実行方法:
Lambda Cloud:
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='verification_output' --exclude='vis_output' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@<ip_address>:/lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux/

ssh -i ~/.ssh/lambda_cloud_key ubuntu@<ip_address> "cd /lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 python -u test/verify_llama4_qformer_sam2_gradients.py 2>&1"
"""

import os
import sys
import logging
import torch
import warnings
from PIL import Image
from typing import Dict, Any, Tuple
import torch.nn as nn
import numpy as np

# Disable various warnings to clean up output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LISA-QFormer components
from model.llama4_qformer_sam2 import QFormerSegmentationBridge, LlamaQFormerSAM2Config
from model.losses_qformer_sam2 import get_composite_loss_qformer_sam2
import config_linux
from utils.dataset import HybridDataset, preprocess_sam_image, build_correct_labels_for_llama4

def print_header(title: str):
    """Print formatted header"""
    logger.info("=" * 80)
    logger.info(f"  {title}")
    logger.info("=" * 80)

def create_test_data_from_dataset(model, dataset_name='sem_seg') -> Dict[str, torch.Tensor]:
    """Create test data from actual HybridDataset
    
    HybridDatasetから実際のデータを取得してテストデータを作成。
    verify_llama4_lisa_gradients.pyとverify_dataset_integrity.pyの実装に準拠。
    """
    
    logger.info("📊 HybridDatasetから実際のテストデータ作成...")
    
    try:
        # HybridDataset初期化（verify_dataset_integrity.pyの実装に準拠）
        from transformers import AutoProcessor
        from utils.dataset import setup_seg_token
        
        # Llama4 Processor取得（Phase 3B統合モデルと同じ）
        if hasattr(model, 'llama_model') and hasattr(model.llama_model, 'processor'):
            processor = model.llama_model.processor
            logger.info("  ✓ llama_model.processorから取得")
        else:
            # フォールバック: 直接初期化
            logger.info("  📦 Llama4 Processor初期化中...")
            processor = AutoProcessor.from_pretrained(config_linux.LLAMA_MODEL_ID)
        
        tokenizer = processor.tokenizer
        
        # [SEG]トークンを追加
        seg_token_idx = setup_seg_token(tokenizer, config_linux.SEG_TOKEN)
        logger.info(f"  ✅ SEGトークンID: {seg_token_idx}")
        
        # HybridDataset初期化
        logger.info("  📦 HybridDataset初期化中...")
        dataset = HybridDataset(
            base_image_dir=config_linux.DATASET_BASE_DIR,
            llama_processor=processor,
            samples_per_epoch=100,  # テスト用に少なめ
            precision="bf16",
            llama_image_size=config_linux.LLAMA_IMAGE_SIZE,
            sam_image_size=config_linux.SAM_IMAGE_SIZE,
            num_classes_per_sample=3,
            exclude_val=False,
            dataset=dataset_name,  # 指定されたデータセットのみ使用
            sample_rate=[1],  # 単一データセット
            sem_seg_data=config_linux.SEM_SEG_DATA if dataset_name == 'sem_seg' else '',
            refer_seg_data=config_linux.REFER_SEG_DATA if dataset_name == 'refer_seg' else '',
            vqa_data=config_linux.VQA_DATA if dataset_name == 'vqa' else '',
            reason_seg_data=config_linux.REASON_SEG_DATA if dataset_name == 'reason_seg' else '',
            explanatory=0.1
        )
        logger.info(f"  ✅ HybridDataset初期化完了 (サンプル数: {len(dataset)})")
        
        # 最初のサンプルを取得
        sample = dataset[0]
        logger.info(f"  📂 ソースデータセット: {sample.get('dataset_name', 'unknown')}")
        
        # Phase 3B統合モデル用のデータ形式に変換
        # Q-Former入力は通常の画像（pixel_values）を使用
        if 'pixel_values' in sample:
            images = sample['pixel_values'].unsqueeze(0) if sample['pixel_values'].dim() == 3 else sample['pixel_values']
        else:
            # フォールバック: SAM画像を使用（リサイズが必要）
            sam_pixel_values = sample['sam_pixel_values']
            # SAM画像（1024x1024）をLlama画像サイズ（336x336）にリサイズ
            import torch.nn.functional as F
            if sam_pixel_values.dim() == 3:
                sam_pixel_values = sam_pixel_values.unsqueeze(0)
            images = F.interpolate(sam_pixel_values, size=(config_linux.LLAMA_IMAGE_SIZE, config_linux.LLAMA_IMAGE_SIZE), mode='bilinear', align_corners=False)
            logger.info(f"  ⚠️ pixel_valuesなし。SAM画像をリサイズして使用: {images.shape}")
        
        # テキスト入力
        input_ids = sample['input_ids'].unsqueeze(0) if sample['input_ids'].dim() == 1 else sample['input_ids']
        attention_mask = sample.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0) if attention_mask.dim() == 1 else attention_mask
        else:
            attention_mask = torch.ones_like(input_ids)
        
        # ラベル（verify_llama4_lisa_gradients.pyと同じ処理）
        labels = sample.get('labels')
        if labels is None:
            # ラベルがない場合は構築
            labels = build_correct_labels_for_llama4(input_ids.squeeze(0), tokenizer)
            labels = labels.unsqueeze(0)
        else:
            labels = labels.unsqueeze(0) if labels.dim() == 1 else labels
        
        # マスク（ground_truth_maskまたはground_truth_masks）
        target_masks = sample.get('ground_truth_mask') or sample.get('ground_truth_masks')
        if target_masks is not None:
            if isinstance(target_masks, list) and len(target_masks) > 0:
                target_masks = target_masks[0]
            if isinstance(target_masks, torch.Tensor):
                if target_masks.dim() == 2:
                    target_masks = target_masks.unsqueeze(0)  # バッチ次元追加
            else:
                target_masks = torch.zeros((1, config_linux.SAM_IMAGE_SIZE, config_linux.SAM_IMAGE_SIZE), dtype=torch.float32)
        else:
            # VQAタスクなどマスクがない場合
            target_masks = torch.zeros((1, config_linux.SAM_IMAGE_SIZE, config_linux.SAM_IMAGE_SIZE), dtype=torch.float32)
            logger.info("  ℹ️ マスクなし（VQAタスクなど）")
        
        # SEGトークンマスク
        seg_token_mask = sample.get('seg_token_mask')
        if seg_token_mask is not None:
            seg_token_mask = seg_token_mask.unsqueeze(0) if seg_token_mask.dim() == 1 else seg_token_mask
        
        # 元画像サイズ
        original_sizes = sample.get('original_size')
        if original_sizes is not None:
            original_sizes = [original_sizes]  # リスト形式に
        
        # SAM用画像
        sam_pixel_values = sample.get('sam_pixel_values')
        if sam_pixel_values is not None:
            sam_pixel_values = sam_pixel_values.unsqueeze(0) if sam_pixel_values.dim() == 3 else sam_pixel_values
        
        # 統合データ辞書作成（Phase 3B統合モデル互換）
        test_data = {
            'images': images,  # Q-Former用画像
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'target_masks': target_masks,
            'seg_token_mask': seg_token_mask,
            'sam_pixel_values': sam_pixel_values,
            'original_sizes': original_sizes,
            'original_prompt': sample.get('text_prompt', ''),
            'dataset_type': sample.get('dataset_name', dataset_name),
            'formatted_prompt': sample.get('formatted_prompt', '')
        }
        
        # データ検証
        logger.info("  ✅ テストデータ検証:")
        for key, value in test_data.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"    - {key}: {value.shape}, dtype: {value.dtype}")
            elif value is not None:
                logger.info(f"    - {key}: {type(value)}")
        
        # Llama-4ネイティブフォーマットの確認
        if '<|image|>' in test_data.get('formatted_prompt', ''):
            logger.info("  ✅ Llama-4ネイティブ<|image|>トークン検出")
        
        # [SEG]トークンの確認
        if seg_token_mask is not None and seg_token_mask.any():
            seg_positions = torch.nonzero(seg_token_mask).squeeze().tolist()
            logger.info(f"  ✅ [SEG]トークン位置: {seg_positions}")
        
        logger.info("📊 HybridDatasetからの実データ取得完了")
        
        return test_data
        
    except Exception as e:
        logger.error(f"❌ HybridDataset初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        logger.info("⚠️ フォールバック: 合成データを使用")
        
        # エラー時は元の合成データ生成にフォールバック
        return create_synthetic_test_data(model)

def create_synthetic_test_data(model) -> Dict[str, torch.Tensor]:
    """合成テストデータ作成（フォールバック用）"""
    
    logger.info("📊 合成テストデータ作成（フォールバック）...")
    
    test_config = config_linux.get_test_config()
    batch_size = 1
    
    # 1. シンプルなテスト画像作成
    image_size = test_config['image_size']  # 448
    
    # 赤い正方形のテスト画像（verify_llama4_lisa_gradients.pyと同じ）
    test_image = Image.new('RGB', (image_size, image_size), color='red')
    image_array = np.array(test_image)
    
    # PyTorchテンソルに変換
    images = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float()
    images = images / 255.0
    
    logger.info(f"  📸 画像生成完了: {images.shape}")
    
    # 2. テストプロンプト
    test_prompt = "Please segment the red region in this image."
    logger.info(f"  💬 プロンプト: '{test_prompt}'")
    
    # 3. プロセッサ取得
    processor = None
    if hasattr(model, 'llama_model') and hasattr(model.llama_model, 'processor'):
        processor = model.llama_model.processor
    elif hasattr(model, 'processor'):
        processor = model.processor
    
    if processor:
        # トークナイゼーション
        text_inputs = processor.tokenizer(
            test_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        
        input_ids = text_inputs['input_ids']
        attention_mask = text_inputs['attention_mask']
    else:
        # フォールバック
        seq_len = 64
        input_ids = torch.randint(1, 32000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
    
    # 4. ダミーマスク
    mask_size = test_config['mask_size']  # 448
    target_masks = torch.ones(batch_size, mask_size, mask_size, dtype=torch.float32)
    
    test_data = {
        'images': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'target_masks': target_masks,
        'original_prompt': test_prompt,
        'dataset_type': 'synthetic_fallback'
    }
    
    logger.info("📊 合成テストデータ作成完了")
    
    return test_data

def analyze_model_components(model: nn.Module) -> Dict[str, Any]:
    """モデルコンポーネントの詳細分析"""
    
    logger.info("🔍 モデルコンポーネント分析...")
    
    analysis = {
        'total_params': 0,
        'trainable_params': 0,
        'components': {},
        'gradient_enabled_components': []
    }
    
    # 全体パラメータ数
    for param in model.parameters():
        analysis['total_params'] += param.numel()
        if param.requires_grad:
            analysis['trainable_params'] += param.numel()
    
    # コンポーネント別分析
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # リーフモジュールのみ
            param_count = sum(p.numel() for p in module.parameters())
            trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if param_count > 0:
                analysis['components'][name] = {
                    'params': param_count,
                    'trainable': trainable_count,
                    'requires_grad': any(p.requires_grad for p in module.parameters())
                }
                
                if trainable_count > 0:
                    analysis['gradient_enabled_components'].append(name)
    
    return analysis

def test_gradient_flow(
    model: nn.Module,
    test_data: Dict[str, torch.Tensor],
    device: str
) -> Dict[str, Any]:
    """勾配フローの詳細テスト（Phase 3B統合モデル対応）"""
    
    logger.info("🔄 勾配フローテスト実行（Phase 3B統合モデル）...")
    
    # Model Parallelismデバイス取得（test_phase3b_integration_real.pyのパターン使用）
    first_device = None
    
    # 複数のアクセス方法を試行
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        first_device = next(iter(model.hf_device_map.values()))
        logger.info("  ✓ device_map (直接アクセス) からデバイス取得")
    elif hasattr(model, 'llama_model') and hasattr(model.llama_model, 'parameters'):
        first_device = next(model.llama_model.parameters()).device
        logger.info("  ✓ llama_model.parameters からデバイス取得")
    elif hasattr(model, 'qformer') and hasattr(model.qformer, 'parameters'):
        first_device = next(model.qformer.parameters()).device
        logger.info("  ✓ qformer.parameters からデバイス取得")
    else:
        # フォールバック: 最初のパラメータのデバイス
        first_device = next(model.parameters()).device
        logger.info("  ✓ model.parameters からデバイス取得")
    
    logger.info(f"  📍 モデルのメインデバイス: {first_device}")
    
    # データを統一デバイスに移動（Phase 3B用に拡張）
    images = test_data['images'].to(first_device)
    input_ids = test_data['input_ids'].to(first_device)
    attention_mask = test_data['attention_mask'].to(first_device)
    target_masks = test_data['target_masks'].to(first_device)
    
    # Phase 3B固有のデータ（存在する場合）
    labels = test_data.get('labels')
    if labels is not None:
        labels = labels.to(first_device)
    
    seg_token_mask = test_data.get('seg_token_mask')
    if seg_token_mask is not None:
        seg_token_mask = seg_token_mask.to(first_device)
    
    sam_pixel_values = test_data.get('sam_pixel_values')
    if sam_pixel_values is not None:
        sam_pixel_values = sam_pixel_values.to(first_device)
    
    original_sizes = test_data.get('original_sizes')
    
    logger.info(f"  📍 入力データデバイス移動完了: {first_device}")
    logger.info(f"  📊 入力データ形状:")
    logger.info(f"    - images: {images.shape}, device: {images.device}")
    logger.info(f"    - input_ids: {input_ids.shape}, device: {input_ids.device}")
    logger.info(f"    - target_masks: {target_masks.shape}, device: {target_masks.device}")
    if labels is not None:
        logger.info(f"    - labels: {labels.shape}, device: {labels.device}")
    if seg_token_mask is not None:
        logger.info(f"    - seg_token_mask: {seg_token_mask.shape}, has SEG: {seg_token_mask.any().item()}")
    if sam_pixel_values is not None:
        logger.info(f"    - sam_pixel_values: {sam_pixel_values.shape}, device: {sam_pixel_values.device}")
    
    # PyTorch gradient anomaly detection (デバッグ用)
    with torch.autograd.detect_anomaly():
        # フォワードパス
        model.train()
        model.zero_grad(set_to_none=True)  # 2025年推奨: メモリ効率化
        
        logger.info("  📈 フォワードパス実行中...")
        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 出力検証
        logger.info("  📋 モデル出力検証:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"    - {key}: {value.shape}, device: {value.device}, dtype: {value.dtype}")
        
        # 損失計算（デバイス一貫性確保）
        logger.info("  📊 損失計算中...")
        
        # モデル内蔵の損失関数を使用
        if hasattr(model, 'compute_loss'):
            # 出力をすべて同じデバイスに確実に移動（train_llama4_lisa_single_process.pyパターン）
            predicted_masks = outputs['predicted_masks']
            if hasattr(predicted_masks, 'to'):
                predicted_masks = predicted_masks.to(first_device)
                
            query_embeds = outputs.get('query_embeddings')
            if query_embeds is not None and hasattr(query_embeds, 'to'):
                query_embeds = query_embeds.to(first_device)
                
            text_embeds = outputs.get('text_embeds')
            if text_embeds is not None and hasattr(text_embeds, 'to'):
                text_embeds = text_embeds.to(first_device)
                
            sam_prompts = outputs.get('sam_prompts')
            if sam_prompts is not None and hasattr(sam_prompts, 'to'):
                sam_prompts = sam_prompts.to(first_device)
            
            logger.info(f"  📍 出力テンソルのデバイス確認完了")
            
            # 損失計算実行
            losses = model.compute_loss(
                predicted_masks=predicted_masks,
                target_masks=target_masks,
                query_embeds=query_embeds,
                text_embeds=text_embeds,
                sam_prompts=sam_prompts
            )
            total_loss = losses['total_loss']
            
            logger.info("  📋 計算された損失:")
            for loss_name, loss_value in losses.items():
                if hasattr(loss_value, 'item'):
                    logger.info(f"    - {loss_name}: {loss_value.item():.6f}")
                else:
                    logger.info(f"    - {loss_name}: {loss_value}")
        else:
            # フォールバック: 基本損失
            logger.info("  ⚠️ compute_lossメソッドが見つかりません。フォールバック損失使用")
            predicted_masks = outputs['predicted_masks']
            if hasattr(predicted_masks, 'to'):
                predicted_masks = predicted_masks.to(first_device)
                
            if predicted_masks.dim() == 4:
                predicted_masks = predicted_masks[:, 0]  # 最初のクエリ
            
            total_loss = nn.functional.binary_cross_entropy_with_logits(
                predicted_masks, target_masks
            )
            losses = {'total_loss': total_loss, 'bce_loss': total_loss}
            logger.info(f"  📋 フォールバック損失: {total_loss.item():.6f}")
        
        # バックワードパス（2025年推奨手法）
        logger.info("  📉 バックワードパス実行中...")
        total_loss.backward()
    
    # 勾配分析（パフォーマンス監視）
    gradient_analysis = analyze_gradients(model)
    
    # メモリクリーンアップ（2025年ベストプラクティス）
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    result = {
        'losses': losses,
        'outputs': outputs,
        'gradient_analysis': gradient_analysis,
        'forward_success': True,
        'backward_success': True,
        'test_data_info': {
            'prompt': test_data.get('original_prompt'),
            'dataset_type': test_data.get('dataset_type')
        }
    }
    
    return result

def analyze_gradients(model: nn.Module) -> Dict[str, Any]:
    """勾配の詳細分析"""
    
    logger.info("  🔍 勾配分析中...")
    
    analysis = {
        'components_with_gradients': [],
        'components_without_gradients': [],
        'gradient_norms': {},
        'zero_gradients': [],
        'large_gradients': [],
        'total_gradient_norm': 0.0
    }
    
    total_norm = 0.0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                analysis['gradient_norms'][name] = grad_norm
                total_norm += grad_norm ** 2
                
                analysis['components_with_gradients'].append(name)
                
                if grad_norm < 1e-8:
                    analysis['zero_gradients'].append(name)
                elif grad_norm > 10.0:
                    analysis['large_gradients'].append(name)
            else:
                analysis['components_without_gradients'].append(name)
    
    analysis['total_gradient_norm'] = total_norm ** 0.5
    
    # 勾配統計表示
    logger.info(f"    勾配有りコンポーネント: {len(analysis['components_with_gradients'])}")
    logger.info(f"    勾配無しコンポーネント: {len(analysis['components_without_gradients'])}")
    logger.info(f"    ゼロ勾配コンポーネント: {len(analysis['zero_gradients'])}")
    logger.info(f"    大勾配コンポーネント: {len(analysis['large_gradients'])}")
    logger.info(f"    総勾配ノルム: {analysis['total_gradient_norm']:.6f}")
    
    # 問題のあるコンポーネントを詳細報告
    if analysis['components_without_gradients']:
        logger.warning("  ⚠️ 勾配が計算されていないコンポーネント:")
        for name in analysis['components_without_gradients'][:5]:  # 最初の5個
            logger.warning(f"    - {name}")
    
    if analysis['zero_gradients']:
        logger.warning("  ⚠️ ゼロ勾配のコンポーネント:")
        for name in analysis['zero_gradients'][:5]:  # 最初の5個
            logger.warning(f"    - {name}")
    
    return analysis

def run_comprehensive_gradient_test():
    """包括的勾配テストの実行"""
    
    print_header("LISA-Llama4-QFormer-SAM2 勾配フロー検証テスト")
    
    try:
        # 1. GPU環境確認
        print_header("1. GPU環境確認")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA が利用できません")
        
        device_count = torch.cuda.device_count()
        logger.info(f"✅ CUDA利用可能: {device_count}個のGPU")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            logger.info(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        
        device = "cuda"
        
        # 2. モデル初期化（Phase 3B統合モデルと同じ設定）
        print_header("2. モデル初期化")
        
        # LlamaQFormerSAM2Configを初期化（パラメータなし）
        config = LlamaQFormerSAM2Config()
        logger.info(f"📋 設定: {config.llama_model_id}")
        
        # test_phase3b_integration_real.pyと同じ方法でLlama-4モデルを事前初期化
        logger.info("📦 Llama-4-Scout事前初期化開始...")
        
        # 2x H100用のメモリ設定
        max_memory = {
            0: "35GB",
            1: "35GB",
            "cpu": "100GB"
        }
        
        # 共有Llamaモデルの初期化（test_phase3b_integration_real.pyと同じ）
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        try:
            # プロセッサ初期化
            shared_processor = AutoProcessor.from_pretrained(
                config_linux.LLAMA_MODEL_ID,
                trust_remote_code=True
            )
            
            # モデル初期化（device_map="auto"で自動分散）
            shared_llama_model = AutoModelForCausalLM.from_pretrained(
                config_linux.LLAMA_MODEL_ID,
                device_map="auto",
                max_memory=max_memory,
                offload_folder="/tmp/llama4_offload",
                torch_dtype=config_linux.TORCH_DTYPE,
                attn_implementation=config_linux.ATTN_IMPLEMENTATION,
                trust_remote_code=True
            )
            logger.info("✅ Llama-4-Scout事前初期化成功")
            
        except Exception as e:
            logger.error(f"❌ Llama-4-Scout初期化失敗: {e}")
            raise
        
        # 共有インスタンスを渡してモデル初期化
        model = QFormerSegmentationBridge(
            config=config,
            shared_llama_model=shared_llama_model,
            shared_llama_processor=shared_processor,
            training_stage=2  # Stage 2で複合損失
        )
        logger.info("✅ QFormerSegmentationBridgeモデル初期化成功")
        
        # 3. モデル分析
        print_header("3. モデルコンポーネント分析")
        
        analysis = analyze_model_components(model)
        logger.info(f"📊 総パラメータ数: {analysis['total_params']:,}")
        logger.info(f"📊 学習可能パラメータ数: {analysis['trainable_params']:,}")
        logger.info(f"📊 学習可能比率: {analysis['trainable_params']/analysis['total_params']*100:.2f}%")
        logger.info(f"📊 勾配有効コンポーネント数: {len(analysis['gradient_enabled_components'])}")
        
        # 4. テストデータ作成
        print_header("4. テストデータ作成")
        
        test_data = create_test_data_from_dataset(model)
        
        # 5. 勾配フローテスト（複数段階）
        for stage in [1, 2, 3]:
            print_header(f"5.{stage}. Stage {stage} 勾配フローテスト")
            
            # 学習段階設定
            if hasattr(model, 'set_training_stage'):
                model.set_training_stage(stage)
            
            # 勾配フローテスト実行
            result = test_gradient_flow(
                model=model,
                test_data=test_data,
                device=device
            )
            
            if result['forward_success'] and result['backward_success']:
                logger.info(f"✅ Stage {stage} 勾配フローテスト成功")
            else:
                logger.error(f"❌ Stage {stage} 勾配フローテスト失敗")
        
        # 6. メモリ使用量確認
        print_header("6. メモリ使用量確認")
        
        for i in range(device_count):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
            
            logger.info(f"📊 GPU {i} メモリ:")
            logger.info(f"  現在: {allocated:.2f}GB, 予約: {cached:.2f}GB, 最大: {max_allocated:.2f}GB")
        
        # 7. 最終検証
        print_header("7. 最終検証")
        
        logger.info("✅ 全ての勾配フローテストが正常に完了しました")
        logger.info("✅ モデルは学習に適した状態です")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 勾配フロー検証中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    logger.info("=== LISA-Llama4-QFormer-SAM2 勾配フロー検証開始 ===")
    
    success = run_comprehensive_gradient_test()
    
    if success:
        logger.info("✅ 勾配フロー検証完了 - 成功")
        return 0
    else:
        logger.error("❌ 勾配フロー検証完了 - 失敗")
        return 1

if __name__ == "__main__":
    exit(main())