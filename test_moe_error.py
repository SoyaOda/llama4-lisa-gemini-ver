"""
MoEエラーのデバッグスクリプト
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
print("=== LoRAExpert初期化テスト ===")
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
    
    # get_expert_info呼び出し
    info = expert.get_expert_info()
    print(f"エキスパート情報: {info}")
    
except Exception as e:
    print(f"エラー発生: {e}")
    import traceback
    traceback.print_exc()