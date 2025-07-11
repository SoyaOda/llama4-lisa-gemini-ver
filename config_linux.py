# config_linux.py
"""
LISA-Llama4統合モデル統一設定ファイル

このファイルは以下の設定を一元管理します：
- LISA-Llama4統合モデルの基本設定
- LoRA（Parameter Efficient Fine-tuning）設定  
- 分散学習・最適化設定
- パス・ディレクトリ設定

使用方法:
    import config_linux as config
    model_config = config.get_lisa_model_config()
    lora_config = config.get_lora_config()
"""
import os
from pathlib import Path
from typing import Dict, Any

# ==============================================================================
# 1. 基本パス設定 (Lambda Cloud環境)
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent

# データセットベースディレクトリ
DATASET_BASE_DIR = os.environ.get("LISA_DATASET_BASE_DIR", "/lambda/nfs/lisa-gemma-project-fs/data/dataset")

# SAMチェックポイントパス（ViT-H）
SAM_CHECKPOINT_PATH = os.environ.get("LISA_SAM_CHECKPOINT_PATH", "/lambda/nfs/lisa-gemma-project-fs/data/weights/sam_vit_h_4b8939.pth")

# Hugging Faceキャッシュディレクトリ
HF_CACHE_DIR = os.environ.get('HF_HOME', None)

# ログ・出力ディレクトリ
LOG_BASE_DIR = str(PROJECT_ROOT / "runs")
WEIGHTS_DIR = str(PROJECT_ROOT / "weights")

# ==============================================================================
# 2. LISA-Llama4統合モデル設定（実際使用値に統一）
# ==============================================================================
# 使用モデル
LLAMA_MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

# 高速ロード設定
USE_SAFETENSORS = True  # safetensors形式を優先的に使用
SAFETENSORS_MODEL_PATH = "./models/llama4-scout-safetensors"  # 変換済みモデルパス
LOW_CPU_MEM_USAGE = True  # CPU→GPU転送を最適化（メモリ使用量削減）

# Llama-4-Scout-17B-16E-Instruct設定（テストスクリプト実使用値）
ATTN_IMPLEMENTATION = "eager"           # flex_attentionバグ回避（実証済み）
DEVICE_MAP = "auto"                     # GPU自動分散（実使用値）
TORCH_DTYPE = "bfloat16"               # 推奨精度（実使用値）

# モデル構造パラメータ
LLAMA_HIDDEN_SIZE = 5120               # Llama4-Scout隠れ層サイズ
SAM_PROMPT_EMBED_DIM = 256             # SAM-ViT-H埋め込み次元
LLAMA_IMAGE_SIZE = 448                 # Llama4画像タイルサイズ
SAM_IMAGE_SIZE = 1024                  # SAMエンコーダ入力サイズ
MODEL_MAX_LENGTH = 131072              # Llama4最大コンテキスト長（128K）

# セグメンテーション特別トークン
SEG_TOKEN = "[SEG]"

# ==============================================================================
# 3. LoRA（PEFT）設定（2024年最適化推奨値）
# ==============================================================================
LORA_R = 64                            # LoRAランク（2024年推奨：大型マルチモーダルモデル用）
LORA_ALPHA = 128                       # LoRAアルファ（2024年推奨：2:1 ratio）
LORA_DROPOUT = 0.05                    # LoRAドロップアウト（維持）

# ターゲットモジュール（2024年推奨：全線形層ターゲット）
LORA_TARGET_MODULES = "all"            # 全線形層をターゲット（LlamaFactory推奨設定）
# 従来設定（参考用）:
# LORA_TARGET_MODULES = [
#     "q_proj", "k_proj", "v_proj", "o_proj",  # Attention層
#     "gate_proj", "up_proj", "down_proj"      # FFN層
# ]

# ==============================================================================
# 4. 学習・最適化設定（A100 80GB × 8GPU最適化）
# ==============================================================================
# 基本学習設定（2024年推奨値）
LEARNING_RATE = 2e-4                   # AdamW学習率（2024年LoRA標準）
WEIGHT_DECAY = 1e-2                    # 重み減衰（維持）
BETA1 = 0.9                            # Adam beta1（維持）
BETA2 = 0.95                           # Adam beta2（維持）

# エポック・ステップ設定
EPOCHS = 10                            # デフォルトエポック数
STEPS_PER_EPOCH = 500                  # ステップ/エポック

# バッチサイズ・勾配設定（A100 80GB × 8GPU最適化）
BATCH_SIZE_PER_GPU = 2                 # GPU単位バッチサイズ（A100 80GB最適化）
GRADIENT_ACCUMULATION_STEPS = 8        # 勾配蓄積ステップ数（維持）
# 実効バッチサイズ = 2 × 8 × 8GPU = 128（従来64から倍増）

# システム最適化設定
MIXED_PRECISION = True                 # BF16混合精度学習
GRADIENT_CHECKPOINTING = True          # メモリ効率化勾配チェックポイント
DATALOADER_NUM_WORKERS = 4             # データローダワーカー数

# 推論設定
MAX_NEW_TOKENS = 100                   # 生成時最大新規トークン数

# 量子化設定（Vision層とMoEルーター用）
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "quantize_vision_layers": True,      # Vision層の量子化を有効化
    "quantize_moe_routers": True         # MoEルーターの量子化を有効化
}

# ==============================================================================
# 5. データセット設定（最小限）
# ==============================================================================
# データセット種別（使用される場合の基本設定）
DATASET_SAMPLE_RATES = "9,3,3,1"       # sem_seg:refer_seg:vqa:reason_seg
SAMPLES_PER_EPOCH = 500                # デバッグ用サンプル数

# データセット名（将来の完全学習用）
SEM_SEG_DATA = "ade20k||cocostuff||mapillary||pascal_part||paco_lvis"
REFER_SEG_DATA = "refclef||refcoco||refcoco+||refcocog"
VQA_DATA = "llava_instruct_150k"
REASON_SEG_DATA = "ReasonSeg|train"
VAL_DATASET = "ReasonSeg|val"

# ==============================================================================
# 6. 統一設定取得関数
# ==============================================================================
def get_lisa_model_config() -> Dict[str, Any]:
    """
    LISA-Llama4統合モデル設定取得
    
    Returns:
        Dict: LisaLlama4Config用設定辞書
    """
    return {
        "llama_model_id": LLAMA_MODEL_ID,
        "sam_checkpoint_path": SAM_CHECKPOINT_PATH,
        "seg_token": SEG_TOKEN,
        "llama_hidden_size": LLAMA_HIDDEN_SIZE,
        "sam_prompt_embed_dim": SAM_PROMPT_EMBED_DIM,
        "llama_image_size": LLAMA_IMAGE_SIZE,
        "sam_image_size": SAM_IMAGE_SIZE,
        "model_max_length": MODEL_MAX_LENGTH,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "device_map": DEVICE_MAP,
        "torch_dtype": TORCH_DTYPE,
    }

def get_lora_config() -> Dict[str, Any]:
    """
    LoRA設定取得
    
    Returns:
        Dict: LoraConfig用設定辞書
    """
    # target_modules の処理：文字列"all"の場合とリストの場合に対応
    if LORA_TARGET_MODULES == "all":
        # PEFTライブラリでは"all"は直接サポートされていないため、
        # 主要な線形層を明示的に指定
        target_modules = [
            # Attention プロジェクション層
            "q_proj", "k_proj", "v_proj", "o_proj",
            # FFN プロジェクション層  
            "gate_proj", "up_proj", "down_proj",
            # 追加の線形層（全線形層ターゲットのため）
            "lm_head", "embed_tokens"
        ]
    else:
        target_modules = LORA_TARGET_MODULES
    
    return {
        "r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": target_modules,
        "bias": "none",
        "use_rslora": False
    }

def get_training_config() -> Dict[str, Any]:
    """
    学習設定取得（分散学習対応）
    
    Returns:
        Dict: 学習設定辞書
    """
    return {
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "beta1": BETA1,
        "beta2": BETA2,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "mixed_precision": MIXED_PRECISION,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "quantization_config": QUANTIZATION_CONFIG,
    }

def get_quantization_config() -> Dict[str, Any]:
    """
    量子化設定取得
    
    Returns:
        Dict: 量子化設定辞書
    """
    return QUANTIZATION_CONFIG

def get_path_config() -> Dict[str, str]:
    """
    パス設定取得
    
    Returns:
        Dict: パス設定辞書
    """
    return {
        "dataset_base_dir": DATASET_BASE_DIR,
        "sam_checkpoint_path": SAM_CHECKPOINT_PATH,
        "hf_cache_dir": HF_CACHE_DIR,
        "log_base_dir": LOG_BASE_DIR,
        "weights_dir": WEIGHTS_DIR,
    }

# ==============================================================================
# 7. 互換性維持（既存コード用）
# ==============================================================================
# 既存のテストスクリプトとの互換性のため、直接アクセス可能な設定を維持
def check_environment() -> bool:
    """
    環境設定確認
    
    Returns:
        bool: 必要なパス・設定が適切かどうか
    """
    required_paths = [
        ("SAMチェックポイント", SAM_CHECKPOINT_PATH),
        ("データセットベース", DATASET_BASE_DIR),
    ]
    
    all_valid = True
    print("=== 環境設定確認 ===")
    
    for name, path in required_paths:
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}が見つかりません: {path}")
            all_valid = False
    
    return all_valid

def print_config_summary():
    """設定サマリ出力"""
    print("=== LISA-Llama4統合設定サマリ ===")
    print(f"Model: {LLAMA_MODEL_ID}")
    print(f"SAM: {SAM_CHECKPOINT_PATH}")
    print(f"Attention: {ATTN_IMPLEMENTATION}")
    print(f"Device Map: {DEVICE_MAP}")
    print(f"Dtype: {TORCH_DTYPE}")
    print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"Batch Size: {BATCH_SIZE_PER_GPU} (per GPU)")
    print(f"Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Mixed Precision: {MIXED_PRECISION}")

# ==============================================================================
# メイン実行（設定確認用）
# ==============================================================================
if __name__ == "__main__":
    print_config_summary()
    print("\n")
    env_ok = check_environment()
    
    if env_ok:
        print("\n🎉 設定確認完了: 全て正常です")
    else:
        print("\n⚠️ 設定確認完了: 一部不備があります")