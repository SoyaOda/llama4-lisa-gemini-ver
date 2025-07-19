"""
SAM2の正確なモジュール名を取得するスクリプト
"""
import torch
import torch.nn as nn
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.sam2_integration import get_sam2_wrapper

def get_exact_module_names():
    """SAM2の正確なモジュール名を取得"""
    print("=== SAM2正確なモジュール名取得 ===\n")
    
    try:
        # SAM2をロード
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
        
        print(f"✅ SAM2モデル取得: {type(sam2_model)}\n")
        
        # qkvとprojを含むモジュールを探す
        print("📋 qkvを含むモジュール:")
        qkv_modules = []
        for name, module in sam2_model.named_modules():
            if 'qkv' in name and isinstance(module, nn.Linear):
                qkv_modules.append(name)
                print(f"  - {name}")
        
        print(f"\n  合計: {len(qkv_modules)}個")
        
        print("\n📋 projを含むモジュール (qkv以外):")
        proj_modules = []
        for name, module in sam2_model.named_modules():
            if 'proj' in name and 'qkv' not in name and isinstance(module, nn.Linear):
                proj_modules.append(name)
                print(f"  - {name}")
        
        print(f"\n  合計: {len(proj_modules)}個")
        
        # MLPモジュール
        print("\n📋 mlpを含むモジュール:")
        mlp_modules = []
        for name, module in sam2_model.named_modules():
            if 'mlp' in name and isinstance(module, nn.Linear):
                mlp_modules.append(name)
                if len(mlp_modules) <= 10:
                    print(f"  - {name}")
        
        print(f"\n  合計: {len(mlp_modules)}個")
        
        # 推奨target_modules
        print("\n✨ 推奨target_modules:")
        print("```python")
        print("SAM2_TARGET_MODULES = [")
        
        # qkvモジュール（最初の数個）
        for mod in qkv_modules[:3]:
            print(f'    "{mod}",')
        if len(qkv_modules) > 3:
            print(f"    # ... 他 {len(qkv_modules) - 3} 個のqkvモジュール")
        
        # projモジュール（最初の数個）
        for mod in proj_modules[:3]:
            print(f'    "{mod}",')
        if len(proj_modules) > 3:
            print(f"    # ... 他 {len(proj_modules) - 3} 個のprojモジュール")
        
        print("]")
        print("```")
        
        # 正規表現パターンの提案
        print("\n📋 正規表現パターン提案:")
        if qkv_modules:
            # パターンを推測
            first_qkv = qkv_modules[0]
            # 例: "image_encoder.trunk.blocks.0.attn.qkv" -> "image_encoder.trunk.blocks.[0-9]+.attn.qkv"
            import re
            pattern = re.sub(r'\.blocks\.\d+\.', r'.blocks.[0-9]+.', first_qkv)
            print(f'  - qkvパターン: "{pattern}"')
        
        if proj_modules:
            first_proj = proj_modules[0]
            pattern = re.sub(r'\.blocks\.\d+\.', r'.blocks.[0-9]+.', first_proj)
            print(f'  - projパターン: "{pattern}"')
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    get_exact_module_names()