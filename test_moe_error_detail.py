"""
MoEエラーの詳細デバッグスクリプト
"""
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.moe_adapters import LoRAExpert
from peft import TaskType, LoraConfig, get_peft_model

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

# LoRAExpertの初期化を段階的にテスト
print("=== LoRAExpert詳細デバッグ ===")

# Step 1: 基本的なPEFT適用テスト
print("\nStep 1: 基本的なPEFT適用テスト")
try:
    dummy_model = DummyModel()
    print("ダミーモデル作成: OK")
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["linear"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    print("LoRA設定作成: OK")
    
    peft_model = get_peft_model(dummy_model, lora_config, autocast_adapter_dtype=False)
    print("PEFT適用: OK")
    print(f"Trainable params: {peft_model.get_nb_trainable_parameters()}")
    
except Exception as e:
    print(f"エラー発生: {e}")
    import traceback
    traceback.print_exc()

# Step 2: LoRAExpertクラスの内部動作確認
print("\n\nStep 2: LoRAExpertクラスのインスタンス作成")
try:
    dummy_model2 = DummyModel()
    
    # LoRAExpertを手動で部分的に初期化
    class PartialLoRAExpert(nn.Module):
        def __init__(self, base_model, target_modules, rank=16, alpha=32, modal_type="test"):
            super().__init__()
            self.modal_type = modal_type
            self.rank = rank
            self.alpha = alpha
            
            print(f"  - modal_type設定: {modal_type}")
            print(f"  - rank/alpha設定: {rank}/{alpha}")
            
            # LoRA設定
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            print(f"  - LoRA設定作成: OK")
            
            # PEFT適用
            try:
                self.peft_model = get_peft_model(
                    base_model, 
                    lora_config, 
                    autocast_adapter_dtype=False
                )
                print(f"  - PEFT適用: OK")
                print(f"  - self.peft_model存在確認: {hasattr(self, 'peft_model')}")
            except Exception as e:
                print(f"  - PEFT適用エラー: {e}")
                raise
            
            # hidden_size取得
            if hasattr(base_model, 'config') and hasattr(base_model.config, 'hidden_size'):
                hidden_size = base_model.config.hidden_size
            else:
                hidden_size = 768
            
            self.modal_projection = nn.Linear(hidden_size, hidden_size)
            self.modal_norm = nn.LayerNorm(hidden_size)
            print(f"  - モーダル特化レイヤー作成: OK")
    
    expert = PartialLoRAExpert(dummy_model2, ["linear"])
    print("PartialLoRAExpert作成: OK")
    print(f"最終的なpeft_model存在確認: {hasattr(expert, 'peft_model')}")
    
except Exception as e:
    print(f"エラー発生: {e}")
    import traceback
    traceback.print_exc()