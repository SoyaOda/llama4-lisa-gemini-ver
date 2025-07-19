"""
SAM2のモジュール構造を確認するスクリプト
"""
import torch
import torch.nn as nn
from typing import List
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def find_linear_modules(model: nn.Module, prefix: str = "") -> List[str]:
    """モデル内のLinearモジュール名を再帰的に検索"""
    linear_modules = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            linear_modules.append(name)
    
    return linear_modules

def analyze_sam2_structure():
    """SAM2のモデル構造を分析"""
    print("=== SAM2モジュール構造分析 ===\n")
    
    try:
        # SAM2をロード
        from model.sam2_integration import get_sam2_wrapper
        
        print("🔄 SAM2をロード中...")
        sam2_wrapper = get_sam2_wrapper(debug_mode=False)
        
        # SAM2モデルの実体を取得
        sam2_model = None
        if hasattr(sam2_wrapper, 'predictor'):
            predictor = sam2_wrapper.predictor
            print(f"✔️ sam2_wrapper.predictorを使用: {type(predictor)}")
            
            # predictorからモデルを取得
            if hasattr(predictor, 'model'):
                sam2_model = predictor.model
                print(f"✔️ predictor.modelを使用: {type(sam2_model)}")
            elif hasattr(predictor, 'sam2_model'):
                sam2_model = predictor.sam2_model
                print(f"✔️ predictor.sam2_modelを使用: {type(sam2_model)}")
            else:
                # predictor自体がモデルの可能性
                sam2_model = predictor
                print(f"✔️ predictor直接使用: {type(predictor)}")
        elif hasattr(sam2_wrapper, 'model'):
            sam2_model = sam2_wrapper.model
            print(f"✔️ sam2_wrapper.model使用: {type(sam2_model)}")
        else:
            # ラップされていない場合
            sam2_model = sam2_wrapper
            print(f"✔️ sam2_wrapper直接使用: {type(sam2_model)}")
        
        # 属性を確認
        print("\n📄 SAM2モデルの属性:")
        attrs = []
        for attr in dir(sam2_model):
            if not attr.startswith('_'):
                try:
                    val = getattr(sam2_model, attr)
                    if not callable(val):
                        attrs.append(attr)
                except:
                    pass
        
        for attr in sorted(attrs)[:10]:
            print(f"  - {attr}")
        
        print("\n✅ SAM2ロード完了\n")
        
        # 1. トップレベルモジュールを表示
        print("📋 トップレベルモジュール:")
        for name, module in sam2_model.named_children():
            print(f"  - {name}: {module.__class__.__name__}")
        
        # 2. image_encoderの構造を詳しく見る
        if hasattr(sam2_model, 'image_encoder'):
            print("\n📋 image_encoder内のモジュール:")
            image_encoder = sam2_model.image_encoder
            
            for name, module in image_encoder.named_children():
                print(f"  - image_encoder.{name}: {module.__class__.__name__}")
            
            # trunk（Hiera backbone）を探す
            if hasattr(image_encoder, 'trunk'):
                print("\n📋 image_encoder.trunk内のモジュール:")
                trunk = image_encoder.trunk
                
                for name, module in trunk.named_children():
                    print(f"  - trunk.{name}: {module.__class__.__name__}")
                
                # blocksを探す
                if hasattr(trunk, 'blocks'):
                    blocks = trunk.blocks
                    print(f"\n📋 trunk.blocksの構造:")
                    
                    if hasattr(blocks, '__len__'):
                        print(f"  - ブロック数: {len(blocks)}")
                        if len(blocks) > 0:
                            # 最初のブロックの構造
                            first_block = blocks[0]
                            print(f"\n📋 最初のblockの詳細:")
                            for name, module in first_block.named_children():
                                print(f"    - blocks[0].{name}: {module.__class__.__name__}")
                                
                                # attnモジュールを詳しく見る
                                if name == 'attn':
                                    for attn_name, attn_module in module.named_children():
                                        print(f"      - attn.{attn_name}: {attn_module.__class__.__name__}")
            
            # neck（FPN）を探す  
            if hasattr(image_encoder, 'neck'):
                print("\n📋 image_encoder.neck内のモジュール:")
                neck = image_encoder.neck
                
                for name, module in neck.named_children():
                    print(f"  - neck.{name}: {module.__class__.__name__}")
        
        # 3. Linear/Conv2dモジュールを検索
        print("\n📋 Linear/Conv2dモジュール一覧（最初の30個）:")
        linear_modules = find_linear_modules(sam2_model)
        
        for i, name in enumerate(linear_modules[:30]):
            print(f"  {i+1}. {name}")
        
        if len(linear_modules) > 30:
            print(f"  ... 他 {len(linear_modules) - 30} 個のモジュール")
        
        # 4. 推奨されるtarget_modules
        print("\n🎯 推奨されるtarget_modules:")
        
        # Hieraアーキテクチャに基づく推奨モジュール
        recommended_modules = []
        
        # qkv関連（Hiera特有）
        qkv_modules = [m for m in linear_modules if 'qkv' in m]
        if qkv_modules:
            print("\n  # Attention層（qkv統合型）:")
            for m in qkv_modules[:5]:
                print(f"  - '{m}'")
                recommended_modules.append(m)
        
        # q_proj, k_proj, v_proj関連（標準Transformer）
        qkv_separate = [m for m in linear_modules if any(x in m for x in ['q_proj', 'k_proj', 'v_proj'])]
        if qkv_separate:
            print("\n  # Attention層（分離型）:")
            for m in qkv_separate[:5]:
                print(f"  - '{m}'")
                recommended_modules.append(m)
        
        # proj関連（出力投影）
        proj_modules = [m for m in linear_modules if 'proj' in m and not any(x in m for x in ['q_proj', 'k_proj', 'v_proj', 'qkv'])]
        if proj_modules:
            print("\n  # Projection層:")
            for m in proj_modules[:5]:
                print(f"  - '{m}'")
                recommended_modules.append(m)
        
        # MLP/FC関連
        mlp_modules = [m for m in linear_modules if any(x in m for x in ['mlp', 'fc', 'ffn'])]
        if mlp_modules:
            print("\n  # MLP層:")
            for m in mlp_modules[:5]:
                print(f"  - '{m}'")
        
        # 最終的な推奨
        print("\n✨ 最適なtarget_modules（論文ベース）:")
        if qkv_modules:
            # Hiera型（qkv統合）
            print("  # Hiera型SAM2の場合:")
            print("  target_modules = [")
            print("    'image_encoder.trunk.blocks.*.attn.qkv',  # Attention")
            print("    'image_encoder.trunk.blocks.*.attn.proj', # Output projection")
            print("  ]")
        elif qkv_separate:
            # 標準Transformer型
            print("  # 標準Transformer型SAM2の場合:")
            print("  target_modules = [")
            print("    'image_encoder.trunk.blocks.*.attn.q_proj',")
            print("    'image_encoder.trunk.blocks.*.attn.k_proj',")
            print("    'image_encoder.trunk.blocks.*.attn.v_proj',")
            print("    'image_encoder.trunk.blocks.*.attn.o_proj',")
            print("  ]")
        
        # 実際に見つかったモジュールから推奨
        if recommended_modules:
            print("\n  # 実際に検出されたモジュール:")
            print("  actual_target_modules = [")
            # 最初の数個を表示
            for m in recommended_modules[:4]:
                # ワイルドカード形式に変換
                if '.blocks.' in m and any(char.isdigit() for char in m):
                    # 数字をワイルドカードに置換
                    import re
                    wildcard = re.sub(r'\.blocks\.\d+\.', '.blocks.*.', m)
                    print(f"    '{wildcard}',")
                else:
                    print(f"    '{m}',")
            print("  ]")
        
        print("\n✅ 分析完了")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_sam2_structure()