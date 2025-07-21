# Phase 3B メモリ分散最適化提案

## 🚨 現状の問題分析

### GPU使用率の不均衡
- **GPU 0**: 74.65GB/79.19GB (94%使用率) ← 🔥 過負荷
- **GPU 1**: 69.77GB/79.19GB (88%使用率) ← 余裕あり
- **差分**: 約5GB の不均等分散

### Webリサーチ結果
1. **Llama-4-Scout 17B 16E**: 2×H100でギリギリ、4×H100が推奨
2. **MoE架構**: 16エキスパートで17B active/109B total
3. **量子化必須**: INT4/NF4量子化でメモリ効率化
4. **Phase 3B複雑性**: Q-Former + SAM2 + デュアルパスウェイで追加メモリ

## 💡 解決策オプション

### Option A: 2×H100メモリ分散最適化（即座実装可能）

```python
# train_phase3b_qformer_bridge.py 修正点

def create_phase3b_model_optimized(logger):
    """メモリ分散最適化版Phase 3Bモデル作成"""
    
    # 🔥 最適化1: より厳格なメモリ制限
    gpu_count = torch.cuda.device_count()
    if gpu_count >= 2:
        max_memory = {
            0: "32GiB",   # GPU0: 40%使用（OOM回避）
            1: "45GiB",   # GPU1: 57%使用（バランス調整）
        }
        device_map = "balanced"  # より均等な分散
    
    # 🔥 最適化2: Llama-4専用量子化強化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_threshold=6.0,
        # 🆕 追加最適化
        bnb_4bit_quant_storage=torch.uint8,  # メモリ効率化
        load_in_8bit_fp32_cpu_offload=True,  # CPU補助
    )
    
    # 🔥 最適化3: CPU offload活用
    shared_llama4_model = Llama4ForConditionalGeneration.from_pretrained(
        qformer_config.llama_model_id,
        quantization_config=quantization_config,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        offload_folder="/tmp/llama4_offload",  # 🆕 CPU offload
        offload_state_dict=True,  # 🆕 状態辞書offload
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
```

### Option B: 4×H100推奨構成（確実解決）

```python
# 4×H100用最適化設定

def create_phase3b_model_4gpu(logger):
    """4×H100用Phase 3Bモデル作成"""
    
    gpu_count = torch.cuda.device_count()
    if gpu_count >= 4:
        # より分散されたメモリ配置
        max_memory = {
            0: "25GiB",   # GPU0: Embedding + 初期層
            1: "30GiB",   # GPU1: 中間層 (主要処理)
            2: "30GiB",   # GPU2: 中間層 + MoE
            3: "25GiB",   # GPU3: 最終層 + Q-Former
        }
        device_map = "balanced_low_0"
        
        # Phase 3Bコンポーネント分散配置
        # - Llama-4: GPU 0-2 (自動分散)
        # - Q-Former: GPU 3
        # - SAM2: GPU 2-3
        # - デュアルパスウェイ: GPU 1-2
```

### Option C: メモリ効率化最適化（2×H100継続）

```python
# メモリ使用量削減の実装

def optimize_memory_usage():
    """Phase 3Bメモリ使用量削減"""
    
    # 1. Gradient Checkpointing強化
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        "use_reentrant": False,
        "preserve_rng_state": False  # メモリ削減
    })
    
    # 2. バッチサイズ動的調整
    def get_dynamic_batch_size(input_length):
        if input_length > 1500:
            return 1  # 長いシーケンス
        elif input_length > 1000:
            return 1  # 中程度
        else:
            return 2  # 短いシーケンス
    
    # 3. Mixed Precision強化
    scaler = GradScaler(
        init_scale=512,  # 低い初期スケール
        growth_factor=1.5,  # 控えめな成長
        backoff_factor=0.8,  # 積極的な縮小
        growth_interval=1000
    )
    
    # 4. メモリクリア強化
    def aggressive_memory_clear():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
```

## 📊 実装優先順位

### 🥇 優先度1: Option A (即座実装)
- 現在の2×H100でメモリ分散最適化
- CPU offload活用
- 量子化パラメータ調整

### 🥈 優先度2: Option C (追加最適化)
- バッチサイズ動的調整
- メモリクリア強化
- Gradient Checkpointing最適化

### 🥉 優先度3: Option B (リソース追加)
- 4×H100環境への移行
- より確実だが、リソースコスト増

## 🚀 即座実装可能な修正

### train_phase3b_qformer_bridge.py修正点

```python
# 242行目付近を修正
        if gpu_count >= 2:
            max_memory = {
                0: "32GiB",   # 🔥 GPU0負荷軽減: 40%使用
                1: "45GiB",   # 🔥 GPU1活用増: 57%使用
            }
            device_map = "balanced"  # 🔥 より均等分散
            
            if gpu_count > 2:
                for i in range(2, gpu_count):
                    max_memory[i] = "35GiB"
        else:
            max_memory = {0: "70GiB"}
            device_map = "auto"

# 追加のoffload設定
        shared_llama4_model = Llama4ForConditionalGeneration.from_pretrained(
            qformer_config.llama_model_id,
            quantization_config=quantization_config,
            device_map=device_map,
            max_memory=max_memory,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            offload_folder="/tmp/llama4_offload",      # 🆕 CPU offload
            offload_state_dict=True,                   # 🆕 状態辞書offload
        )
```

## 🎯 推奨アクション

1. **即座**: train_phase3b_qformer_bridge.pyのメモリ設定を修正
2. **短期**: バッチサイズを1に固定、steps_per_epochを10に削減
3. **中期**: 4×H100環境への移行検討
4. **長期**: より軽量なPhase 3Bバリアント開発

これらの修正により、2×H100でもPhase 3B学習が可能になると予想されます。