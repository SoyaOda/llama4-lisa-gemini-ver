#!/usr/bin/env python3
"""
LISA-Llama4統合モデル 勾配フロー検証スクリプト
verify_llama4_loss_and_gradients.pyを参考に、LISA統合モデル特有の機能もテスト

実行方法:
Lambda Cloud (129.213.148.184):
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='verification_output' --exclude='vis_output' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@129.213.148.184:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

ssh -i ~/.ssh/lambda_cloud_key ubuntu@129.213.148.184 "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 python -u verify_llama4_lisa_gradients.py 2>&1"
"""

import os
import sys
import logging
import torch
import warnings
from PIL import Image
from typing import Dict, Any, Tuple
import torch.nn as nn

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
sys.path.insert(0, '.')

# Import LISA components
from model.llama4_lisa import LisaLlama4ForCausalLM, LisaLlama4Config
import config_linux

def print_header(title: str):
    """Print formatted header"""
    logger.info("=" * 80)
    logger.info(f"  {title}")
    logger.info("=" * 80)

def create_test_data() -> Tuple[Image.Image, str]:
    """Create test image and text data"""
    # Simple test image
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # Test prompt for segmentation
    test_prompt = "この画像で赤い領域を[SEG]してください。"
    
    return test_image, test_prompt

def analyze_gradients(model: nn.Module) -> Dict[str, Any]:
    """Analyze gradient information"""
    gradient_info = {
        'total_params': 0,
        'trainable_params': 0,
        'params_with_grad': 0,
        'zero_grad_params': 0,
        'gradient_norm': 0.0,
        'component_analysis': {}
    }
    
    component_grads = {}
    
    for name, param in model.named_parameters():
        gradient_info['total_params'] += 1
        
        if param.requires_grad:
            gradient_info['trainable_params'] += 1
            
            if param.grad is not None:
                gradient_info['params_with_grad'] += 1
                
                # Calculate gradient norm
                grad_norm = param.grad.norm().item()
                gradient_info['gradient_norm'] += grad_norm ** 2
                
                # Component-wise analysis
                component = name.split('.')[0] if '.' in name else name
                if component not in component_grads:
                    component_grads[component] = {
                        'count': 0, 
                        'norm': 0.0,
                        'params': []
                    }
                component_grads[component]['count'] += 1
                component_grads[component]['norm'] += grad_norm ** 2
                component_grads[component]['params'].append(name)
                
                if grad_norm == 0:
                    gradient_info['zero_grad_params'] += 1
    
    # Calculate L2 norm
    gradient_info['gradient_norm'] = gradient_info['gradient_norm'] ** 0.5
    
    # Calculate component norms
    for component in component_grads:
        component_grads[component]['norm'] = component_grads[component]['norm'] ** 0.5
    
    gradient_info['component_analysis'] = component_grads
    
    return gradient_info

def verify_loss_calculation(model: nn.Module, inputs: Dict[str, torch.Tensor], verbose: bool = True) -> torch.Tensor:
    """Verify loss calculation with detailed debugging"""
    model.train()
    
    if verbose:
        logger.info("=== 損失計算検証 ===")
        logger.info(f"入力形状:")
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: {value.shape}")
    
    # Forward pass - 実際のLISA統合モデルを使用（SAM機能付き）
    try:
        if verbose:
            logger.info("LISA統合モデル：完全なフォワードパス実行（SAM機能付き）")
        
        # 実際のLISA統合モデルの完全フォワードパス
        # SAMを有効にしてセグメンテーション機能も検証
        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            pixel_values=inputs.get('pixel_values'),
            labels=inputs['input_ids'],  # 言語モデリング用
            generate_mask=True  # SAM機能を有効化
        )
        
        if verbose:
            logger.info(f"出力型: {type(model_outputs)}")
            logger.info(f"出力キー: {list(model_outputs.keys()) if isinstance(model_outputs, dict) else 'no keys'}")
        
        # CompositeLoss統合による損失取得
        if isinstance(model_outputs, dict):
            # CompositeLossからの統一損失
            if 'text_loss' in model_outputs:
                loss = model_outputs['text_loss']
                if verbose:
                    logger.info("✅ CompositeLoss統合損失を使用")
            # 予備処理：lossキーも確認
            elif 'loss' in model_outputs:
                loss = model_outputs['loss']
                if verbose:
                    logger.info("✅ 通常のloss損失を使用")
            # フォールバック：手動計算
            else:
                if verbose:
                    logger.info("手動損失計算に切り替え...")
                
                logits = model_outputs.get('logits')
                if logits is None:
                    raise ValueError("logitsが見つかりません")
                
                labels = inputs.get('input_ids')
                if labels is None:
                    raise ValueError("input_idsが見つかりません")
                
                # 言語モデリング損失を手動計算
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )
                
                if verbose:
                    logger.info("✅ 手動交差エントロピー損失計算完了")
        else:
            # 非辞書型出力の場合
            if hasattr(model_outputs, 'loss'):
                loss = model_outputs.loss
            else:
                raise ValueError("損失が見つかりません")
        
        # SAM機能確認
        if isinstance(model_outputs, dict) and 'predicted_masks' in model_outputs:
            masks = model_outputs['predicted_masks']
            if masks is not None:
                if verbose:
                    logger.info(f"✅ SAMマスク生成成功: {masks.shape if hasattr(masks, 'shape') else type(masks)}")
            else:
                if verbose:
                    logger.info("ℹ️ SAMマスク未生成（SEGトークンなしまたはSAM無効）")
        
        if verbose:
            logger.info(f"✅ 損失値: {loss.item():.6f}")
            logger.info(f"損失のデバイス: {loss.device}")
            logger.info(f"損失のrequires_grad: {loss.requires_grad}")
        
        return loss
        
    except Exception as e:
        logger.error(f"❌ 損失計算でエラー: {e}")
        raise

def main():
    """Main verification function"""
    print_header("LISA-Llama4統合モデル 勾配フロー検証")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("❌ CUDA が利用できません")
        return False
    
    logger.info(f"✅ CUDA利用可能: {torch.cuda.device_count()}個のGPU")
    for i in range(torch.cuda.device_count()):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    try:
        # Disable dynamic compilation for distributed setup
        torch.compiler.disable()
        logger.info("✅ 動的コンパイル無効化: GPU分散エラー回避のため")
        
        # Step 1: Model initialization
        print_header("ステップ1: モデル初期化")
        
        config = LisaLlama4Config(**config_linux.get_lisa_model_config())
        
        logger.info("LISA-Llama4モデル初期化中...")
        model = LisaLlama4ForCausalLM(config)
        logger.info("✅ モデル初期化完了")
        
        # Step 2: Apply LoRA for training
        print_header("ステップ2: LoRA設定適用")
        
        from peft import LoraConfig, get_peft_model
        
        lora_config_dict = config_linux.get_lora_config()
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            **lora_config_dict
        )
        
        logger.info("LoRA設定適用中...")
        model = get_peft_model(model, lora_config)
        
        # Freeze embeddings and LM head
        for name, param in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = False
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"総パラメータ: {total_params:,}")
        logger.info(f"学習可能パラメータ: {trainable_params:,} ({100*trainable_params/total_params:.3f}%)")
        logger.info("✅ LoRA適用完了")
        
        # Step 3: Create test data
        print_header("ステップ3: テストデータ作成")
        
        test_image, test_prompt = create_test_data()
        logger.info(f"テスト画像: {test_image.size}")
        logger.info(f"テストプロンプト: {test_prompt}")
        
        # 公式推奨方法：学習時も推論時と同じインターフェースを使用
        # メッセージ形式で準備（HuggingFace公式ドキュメント準拠）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},
                    {"type": "text", "text": test_prompt}
                ]
            }
        ]
        
        # Llama4 Processorで直接処理（成功した単独モデルと同じ方法）
        inputs = model.llama_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # BatchFeatureを辞書に変換（既存の修正を活用）
        if hasattr(inputs, 'keys') and hasattr(inputs, '__getitem__'):
            result_dict = {}
            for key in inputs.keys():
                result_dict[key] = inputs[key]
            inputs = result_dict
            logger.info(f"✅ BatchFeature→dict変換完了: {list(inputs.keys())}")
        
        # 最初のGPUに移動
        first_device = next(iter(model.hf_device_map.values())) if hasattr(model, 'hf_device_map') else next(model.parameters()).device
        inputs = {k: v.to(first_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        logger.info(f"入力準備完了: {list(inputs.keys())}, device: {first_device}")
        
        # Step 4: Initial gradient state
        print_header("ステップ4: 初期勾配状態確認")
        
        model.zero_grad()
        initial_grad_info = analyze_gradients(model)
        logger.info(f"初期状態:")
        logger.info(f"  総パラメータ: {initial_grad_info['total_params']}")
        logger.info(f"  学習可能パラメータ: {initial_grad_info['trainable_params']}")
        logger.info(f"  勾配保持パラメータ: {initial_grad_info['params_with_grad']}")
        
        # Step 5: Forward pass and loss calculation
        print_header("ステップ5: 順伝播と損失計算")
        
        loss = verify_loss_calculation(model, inputs, verbose=True)
        logger.info(f"✅ 損失計算成功: {loss.item():.6f}")
        
        # Step 6: Backward pass
        print_header("ステップ6: 逆伝播")
        
        logger.info("逆伝播実行中...")
        loss.backward()
        logger.info("✅ 逆伝播完了")
        
        # Step 7: Post-backward gradient analysis
        print_header("ステップ7: 逆伝播後勾配分析")
        
        post_grad_info = analyze_gradients(model)
        logger.info(f"逆伝播後:")
        logger.info(f"  勾配保持パラメータ: {post_grad_info['params_with_grad']}")
        logger.info(f"  ゼロ勾配パラメータ: {post_grad_info['zero_grad_params']}")
        logger.info(f"  勾配L2ノルム: {post_grad_info['gradient_norm']:.6f}")
        
        logger.info("\nコンポーネント別勾配分析:")
        for component, info in post_grad_info['component_analysis'].items():
            logger.info(f"  {component}: {info['count']}個, ノルム={info['norm']:.6f}")
        
        # Step 8: Gradient flow verification
        print_header("ステップ8: 勾配フロー検証")
        
        if post_grad_info['params_with_grad'] > 0:
            logger.info(f"✅ 勾配フロー正常: {post_grad_info['params_with_grad']}個のパラメータに勾配")
            
            # Check if gradients are meaningful (not all zeros)
            if post_grad_info['gradient_norm'] > 1e-8:
                logger.info(f"✅ 勾配は意味のある値です (L2ノルム: {post_grad_info['gradient_norm']:.6f})")
            else:
                logger.warning(f"⚠️ 勾配が非常に小さいです (L2ノルム: {post_grad_info['gradient_norm']:.6f})")
            
            # Check LoRA components specifically
            lora_components = [comp for comp in post_grad_info['component_analysis'].keys() 
                             if 'lora' in comp.lower() or 'base_layer' in comp.lower()]
            if lora_components:
                logger.info(f"✅ LoRAコンポーネントに勾配フロー: {lora_components}")
            
            # Check multi-modal projector
            projector_components = [comp for comp in post_grad_info['component_analysis'].keys() 
                                  if 'projector' in comp.lower()]
            if projector_components:
                logger.info(f"✅ マルチモーダルプロジェクタに勾配フロー: {projector_components}")
            
            success = True
        else:
            logger.error("❌ 勾配フローが検出されません")
            success = False
        
        # Step 9: Summary
        print_header("検証結果サマリ")
        
        logger.info(f"モデル初期化: ✅")
        logger.info(f"LoRA適用: ✅")
        logger.info(f"データ準備: ✅")
        logger.info(f"順伝播: ✅")
        logger.info(f"損失計算: ✅ (損失値: {loss.item():.6f})")
        logger.info(f"逆伝播: ✅")
        logger.info(f"勾配フロー: {'✅' if success else '❌'}")
        
        if success:
            logger.info("🎉 勾配フロー検証成功！LISA-Llama4統合モデルは学習準備完了です")
        else:
            logger.error("❌ 勾配フロー検証失敗")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ 検証中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("LISA-Llama4統合モデル 勾配フロー検証開始")
    success = main()
    if success:
        logger.info("🎉 検証完了 - 成功")
        sys.exit(0)
    else:
        logger.error("❌ 検証失敗")
        sys.exit(1) 