"""
SAM2のPEFT/LoRA適用デバッグスクリプト
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.sam2_integration import get_sam2_wrapper
import config_linux

def analyze_sam2_for_peft():
    """SAM2のPEFT/LoRA適用可能性を分析"""
    print("=== SAM2 PEFT/LoRA適用分析 ===\n")
    
    try:
        # SAM2をロード
        print("🔄 SAM2をロード中...")
        sam2_wrapper = get_sam2_wrapper(debug_mode=False)
        
        # SAM2モデルの実体を取得
        sam2_model = None
        if hasattr(sam2_wrapper, 'predictor'):
            predictor = sam2_wrapper.predictor
            if hasattr(predictor, 'model'):
                sam2_model = predictor.model
            else:
                sam2_model = predictor
        else:
            sam2_model = sam2_wrapper
        
        print(f"✅ SAM2モデル取得: {type(sam2_model)}")
        
        # 1. config属性の確認
        print("\n📋 config属性の確認:")
        if hasattr(sam2_model, 'config'):
            print(f"  ✅ config属性あり")
            config = sam2_model.config
            if hasattr(config, 'hidden_size'):
                print(f"  - hidden_size: {config.hidden_size}")
            else:
                print(f"  - hidden_size属性なし")
        else:
            print(f"  ❌ config属性なし")
        
        # 2. hidden_sizeの推測
        print("\n📋 hidden_size推測:")
        possible_hidden_sizes = []
        
        # Linear層から推測
        for name, module in sam2_model.named_modules():
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                
                # 一般的なhidden_sizeの候補
                for size in [in_features, out_features]:
                    if size in [256, 384, 512, 768, 1024, 1280, 2048]:
                        possible_hidden_sizes.append(size)
        
        if possible_hidden_sizes:
            from collections import Counter
            size_counts = Counter(possible_hidden_sizes)
            most_common_size = size_counts.most_common(1)[0][0]
            print(f"  - 推測されるhidden_size: {most_common_size}")
            print(f"  - サイズ分布: {dict(size_counts.most_common(5))}")
        
        # 3. PEFT適用可能なモジュールの特定
        print("\n📋 PEFT適用可能モジュール:")
        peft_compatible_modules = []
        
        for name, module in sam2_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # パラメータ数が大きいモジュールを優先
                param_count = sum(p.numel() for p in module.parameters())
                if param_count > 10000:  # 閾値
                    peft_compatible_modules.append({
                        'name': name,
                        'type': type(module).__name__,
                        'params': param_count,
                        'in_features': getattr(module, 'in_features', getattr(module, 'in_channels', None)),
                        'out_features': getattr(module, 'out_features', getattr(module, 'out_channels', None))
                    })
        
        # パラメータ数でソート
        peft_compatible_modules.sort(key=lambda x: x['params'], reverse=True)
        
        print(f"  - 総モジュール数: {len(peft_compatible_modules)}")
        print("\n  上位10モジュール:")
        for i, mod in enumerate(peft_compatible_modules[:10]):
            print(f"  {i+1}. {mod['name']}")
            print(f"     - タイプ: {mod['type']}")
            print(f"     - パラメータ数: {mod['params']:,}")
            print(f"     - 次元: {mod['in_features']} -> {mod['out_features']}")
        
        # 4. image_encoder構造の詳細分析
        if hasattr(sam2_model, 'image_encoder'):
            print("\n📋 image_encoder詳細分析:")
            image_encoder = sam2_model.image_encoder
            
            # trunk (Hiera backbone)
            if hasattr(image_encoder, 'trunk'):
                trunk = image_encoder.trunk
                print(f"  - trunk型: {type(trunk).__name__}")
                
                # blocks構造
                if hasattr(trunk, 'blocks'):
                    blocks = trunk.blocks
                    if hasattr(blocks, '__len__'):
                        print(f"  - ブロック数: {len(blocks)}")
                        
                        # 最初のブロックを詳細分析
                        if len(blocks) > 0:
                            first_block = blocks[0]
                            print(f"\n  最初のブロック構造:")
                            
                            # attention層を探す
                            if hasattr(first_block, 'attn'):
                                attn = first_block.attn
                                print(f"    - attention型: {type(attn).__name__}")
                                
                                # QKV層を探す
                                for attr_name in ['qkv', 'q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value']:
                                    if hasattr(attn, attr_name):
                                        layer = getattr(attn, attr_name)
                                        if isinstance(layer, nn.Module):
                                            print(f"    - {attr_name}層あり: {type(layer).__name__}")
                                            if hasattr(layer, 'in_features'):
                                                print(f"      入力次元: {layer.in_features}")
                                                print(f"      出力次元: {layer.out_features}")
        
        # 5. PEFT適用推奨設定
        print("\n✨ PEFT/LoRA適用推奨設定:")
        
        # ワイルドカード展開テスト
        print("\n  ワイルドカード展開テスト:")
        target_patterns = config_linux.SAM2_TARGET_MODULES
        expanded_targets = []
        
        for pattern in target_patterns:
            if '*' in pattern:
                import re
                regex_pattern = pattern.replace('*', r'\\d+')
                regex_pattern = f"^{regex_pattern}$"
                regex = re.compile(regex_pattern)
                
                matches = []
                for name, module in sam2_model.named_modules():
                    if isinstance(module, nn.Linear) and regex.match(name):
                        matches.append(name)
                
                if matches:
                    print(f"  - パターン '{pattern}' -> {len(matches)}個のモジュールにマッチ")
                    for match in matches[:3]:
                        print(f"    例: {match}")
                    expanded_targets.extend(matches)
                else:
                    print(f"  - パターン '{pattern}' -> マッチなし")
        
        # 推奨設定出力
        print("\n  推奨LoRA設定:")
        print("  ```python")
        print("  # SAM2Base対応ラッパー")
        print("  class SAM2LoRAWrapper(nn.Module):")
        print("      def __init__(self, sam2_model):")
        print("          super().__init__()")
        print("          self.model = sam2_model")
        print("          # configがない場合の対処")
        print("          if not hasattr(sam2_model, 'config'):")
        print("              self.config = type('Config', (), {")
        print(f"                  'hidden_size': {most_common_size if 'most_common_size' in locals() else 256}")
        print("              })()")
        print("  ")
        print("  # LoRA設定")
        print("  lora_config = LoraConfig(")
        print("      r=16,")
        print("      lora_alpha=32,")
        print("      target_modules=expanded_targets[:10],  # 上位10モジュール")
        print("      lora_dropout=0.1,")
        print("      bias='none',")
        print("      task_type=TaskType.FEATURE_EXTRACTION")
        print("  )")
        print("  ```")
        
        print("\n✅ 分析完了")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_sam2_for_peft()