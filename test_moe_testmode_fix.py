"""
MoE test_mode修正の検証スクリプト
"""
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.moe_adapters import LoRAExpert, HeterogeneousMoEAdapter
from peft import TaskType

# シンプルなダミーモデル
class DummyLlamaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32000, 5120)
        self.linear = nn.Linear(5120, 5120)
        
        # Llama-4風のconfig
        class ConfigDict(dict):
            def __init__(self):
                super().__init__()
                self['hidden_size'] = 5120
                self['model_type'] = 'llama'
                self['tie_word_embeddings'] = False
                
            def __getattr__(self, key):
                return self.get(key, None)
                
            def __setattr__(self, key, value):
                self[key] = value
            
            def to_dict(self):
                return dict(self)
        
        self.config = ConfigDict()
    
    def forward(self, input_ids=None, **kwargs):
        # 通常モード: input_idsから埋め込みを生成
        if input_ids is not None:
            embeds = self.embed_tokens(input_ids)
            output = self.linear(embeds)
            
            class Output:
                def __init__(self, hidden_states):
                    self.last_hidden_state = hidden_states
            
            return Output(output)
        else:
            raise ValueError("input_ids is required for LlamaModel")
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """PEFT CAUSAL_LMタスクに必要なメソッド"""
        return {"input_ids": input_ids}

class DummySAM2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 256)
        
        class ConfigDict(dict):
            def __init__(self):
                super().__init__()
                self['hidden_size'] = 256
                self['model_type'] = 'sam2'
                self['tie_word_embeddings'] = False
                
            def __getattr__(self, key):
                return self.get(key, None)
                
            def __setattr__(self, key, value):
                self[key] = value
            
            def to_dict(self):
                return dict(self)
        
        self.config = ConfigDict()

class DummyQFormerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 768)
        
        class ConfigDict(dict):
            def __init__(self):
                super().__init__()
                self['hidden_size'] = 768
                self['model_type'] = 'qformer'
                self['tie_word_embeddings'] = False
                
            def __getattr__(self, key):
                return self.get(key, None)
                
            def __setattr__(self, key, value):
                self[key] = value
            
            def to_dict(self):
                return dict(self)
        
        self.config = ConfigDict()

# テスト実行
print("=== MoE test_mode修正検証 ===")

try:
    # 1. ベースモデル作成
    base_models = {
        "llama": DummyLlamaModel(),
        "sam2": DummySAM2Model(),
        "qformer": DummyQFormerModel()
    }
    print("✅ ベースモデル作成完了")
    
    # 2. MoEアダプター作成（テスト用に簡略化されたconfig）
    test_moe_config = {
        "lora_rank": 16,
        "lora_alpha": 32,
        "num_experts": 3,
        "active_experts": 2,
        "expert_capacity_factor": 1.25,
        "expert_weights": {
            "llama": 0.4,
            "sam2": 0.4,
            "qformer": 0.2
        }
    }
    
    # config_linux.pyのtarget_modulesを一時的に上書き
    import config_linux
    original_mle_config = config_linux.get_mle_config()
    config_linux.get_mle_config = lambda: {
        **original_mle_config,
        'target_modules': {
            'llama': ['linear'],      # ダミーモデル用に簡略化
            'sam2': ['linear'],       # ダミーモデル用に簡略化
            'qformer': ['linear']     # ダミーモデル用に簡略化
        }
    }
    
    # Llamaモデル専用: より単純なタスクタイプに変更
    # LoRAExpertクラスを一時的にパッチ
    from model import moe_adapters
    original_init = moe_adapters.LoRAExpert.__init__
    
    def patched_init(self, base_model, target_modules, rank=16, alpha=32, dropout=0.1, 
                     modal_type="general", task_type=TaskType.CAUSAL_LM):
        # Llamaモデルの場合、より単純なタスクタイプを使用
        if modal_type == "language" and hasattr(base_model, '__class__') and 'Dummy' in base_model.__class__.__name__:
            task_type = TaskType.FEATURE_EXTRACTION
        original_init(self, base_model, target_modules, rank, alpha, dropout, modal_type, task_type)
    
    moe_adapters.LoRAExpert.__init__ = patched_init
    
    moe_adapter = HeterogeneousMoEAdapter(
        base_models=base_models,
        hidden_size=5120,
        moe_config=test_moe_config
    )
    print("✅ MoEアダプター作成完了")
    
    # 3. テスト入力（hidden_states形式）
    batch_size = 2
    seq_len = 16
    hidden_size = 5120
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_input = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16).to(device)
    
    print(f"\nテスト入力形状: {test_input.shape}")
    print(f"テスト入力デバイス: {test_input.device}")
    
    # 4. test_mode=Trueでのテスト（修正版）
    print("\n📊 test_mode=True での実行:")
    try:
        with torch.no_grad():
            # 動的ルーティング
            output1, info1 = moe_adapter(test_input, test_mode=True)
            print(f"  ✅ 動的ルーティング成功: 出力形状 {output1.shape}")
            
            # 各エキスパート強制使用
            for expert_name in ['llama', 'sam2', 'qformer']:
                output2, info2 = moe_adapter(test_input, expert_type=expert_name, test_mode=True)
                print(f"  ✅ {expert_name}エキスパート成功: 出力形状 {output2.shape}")
        
        print("\n🎉 test_mode修正が正常に動作しています！")
        
    except Exception as e:
        print(f"\n❌ test_mode=True でエラー: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. test_mode=Falseでのテスト（従来の問題を確認）
    print("\n📊 test_mode=False での実行（従来の問題確認）:")
    try:
        with torch.no_grad():
            output3, info3 = moe_adapter(test_input, expert_type='llama', test_mode=False)
            print(f"  ⚠️ 予想外の成功？: {output3.shape}")
    except Exception as e:
        print(f"  ✅ 期待通りのエラー: {type(e).__name__}: {str(e)[:100]}...")
        print("     → Llamaモデルはinput_idsを要求するため、hidden_statesでは動作しない")
    
    print("\n✅ 修正検証完了")
    
    # 元の設定を復元
    config_linux.get_mle_config = lambda: original_mle_config
    moe_adapters.LoRAExpert.__init__ = original_init
    
except Exception as e:
    print(f"❌ エラー発生: {e}")
    import traceback
    traceback.print_exc()
    
    # エラー時も元の設定を復元
    if 'original_mle_config' in locals():
        config_linux.get_mle_config = lambda: original_mle_config
    if 'original_init' in locals():
        moe_adapters.LoRAExpert.__init__ = original_init