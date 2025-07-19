"""
peft_model属性の存在確認デバッグ
"""
import torch
import torch.nn as nn
from model.moe_adapters import LoRAExpert
from peft import TaskType

# シンプルなダミーモデル
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 768)
        # PEFTが期待するconfig形式
        class ConfigDict(dict):
            def __init__(self):
                super().__init__()
                self['hidden_size'] = 768
                self['model_type'] = 'dummy'
                self['tie_word_embeddings'] = False
                
            def __getattr__(self, key):
                return self.get(key, None)
                
            def __setattr__(self, key, value):
                self[key] = value
            
            def to_dict(self):
                return dict(self)
        
        self.config = ConfigDict()

# テスト実行
print("=== peft_model属性確認 ===")
try:
    dummy_model = DummyModel()
    print("ダミーモデル作成: OK")
    
    # LoRAExpert作成
    expert = LoRAExpert(
        base_model=dummy_model,
        target_modules=["linear"],
        rank=16,
        alpha=32,
        modal_type="test",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    print("LoRAExpert作成: OK")
    
    # 属性を確認
    print(f"\nexpertの属性一覧:")
    for attr in dir(expert):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    
    print(f"\npeft_model属性の存在: {hasattr(expert, 'peft_model')}")
    
    # 実際にpeft_modelにアクセス
    if hasattr(expert, 'peft_model'):
        print(f"peft_modelの型: {type(expert.peft_model)}")
        print(f"peft_modelの属性例: {list(vars(expert.peft_model).keys())[:5]}")
    else:
        print("peft_model属性が見つかりません")
        
        # さらに詳しく調査
        print(f"\nexpertのvars():")
        for key, value in vars(expert).items():
            print(f"  - {key}: {type(value)}")
    
except Exception as e:
    print(f"エラー発生: {e}")
    import traceback
    traceback.print_exc()