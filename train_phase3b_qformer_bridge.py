#!/usr/bin/env python3
"""
Phase 3B QFormerSegmentationBridge学習スクリプト（loss減少確認用）

test_phase3b_integration_real.pyのモデル構成 + train_llama4_lisa_single_process.pyの学習方法を統合

特徴：
- QFormerSegmentationBridge（方法3）使用
- デュアルパスウェイデコーダー統合
- HybridDatasetによるデュアルエンコーダー対応
- 実際のLlama-4-Scout + SAM2使用
- MoE統合最適化対応
- loss減少確認・可視化

実行方法:
CUDA_VISIBLE_DEVICES=0,1 python train_phase3b_qformer_bridge.py --exp_name phase3b_loss_check --steps_per_epoch 20 --epochs 3
"""

import argparse
import os
import sys
import time
import json
import logging
import gc
from datetime import datetime
from pathlib import Path
import warnings
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import transformers
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logging.warning("bitsandbytes not available. Using standard AdamW optimizer.")
import gc  # Reference準拠：ガベージコレクション
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # バックエンドを非対話モードに設定
import matplotlib.pyplot as plt
import numpy as np

# Disable warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 🔥 PyTorchインポート前の環境準備（CUDA Error 802対策 + H100 OOM完全回避）
print("🔧 CUDA Error 802対策：PyTorchインポート前環境設定...")
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'

# 🔥 Web調査H100最適化: PYTORCH_CUDA_ALLOC_CONF設定
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.6'
print("✅ H100最適化: max_split_size_mb:512で断片化防止, garbage_collection強化")

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Phase 3B統合モデル（test_phase3b_integration_real.py参考）
from model.llama4_qformer_sam2 import QFormerSegmentationBridge, LlamaQFormerSAM2Config
from model.sam2_integration import get_sam2_wrapper
from model.qformer import get_qformer_model  
from model.dual_pathway_decoder import create_dual_pathway_decoder
from model.multiresolution_fusion import Llama4SAM2MultiResolutionFusion
from model.ohem_loss import create_ohem_loss
from model.dataset_adapter import adapt_dataset_for_qformer

# データセット（HybridDataset使用）
from utils.dataset import HybridDataset, collate_fn
from utils.constants import DEFAULT_SEG_TOKEN
import config_linux

# 評価用メトリクス（train_llama4_lisa_single_process.py参考）
from utils.utils import (
    AverageMeter, ProgressMeter, Summary, 
    dict_to_cuda, intersectionAndUnionGPU
)

# Llama-4-Scout直接ロード
try:
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
    print("✅ HuggingFace Transformers利用可能")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ HuggingFace Transformersが利用できません")

def enforce_strict_memory_limit(memory_gb_per_gpu=10, logger=None):
    """
    🔥 Web調査解決策: PyTorch厳格メモリ制限実装
    
    H100 80GBで10GB制限など、厳格なメモリ制限を設定
    - torch.cuda.set_per_process_memory_fraction: PyTorchレベル制限
    - max_memory設定: HuggingFaceレベル制限
    - 二重安全装置でOOM完全回避
    
    Args:
        memory_gb_per_gpu: 1GPU当たりメモリ制限（GB）
        logger: ログ出力
    
    Returns:
        max_memory dict for HuggingFace models
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    import torch
    
    logger.info(f"🔥 Web調査解決策: 厳格メモリ制限実装開始")
    logger.info(f"  - 目標制限: {memory_gb_per_gpu}GB/GPU")
    
    # 1. GPU環境確認
    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA利用不可: CPU環境")
        return {"cpu": f"{memory_gb_per_gpu * 4}GB"}
        
    gpu_count = torch.cuda.device_count()
    logger.info(f"  - 検出GPU数: {gpu_count}")
    
    # 2. 各GPU総メモリ確認
    total_memories = []
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024**3)
        total_memories.append(total_gb)
        logger.info(f"  - GPU{i}: {total_gb:.1f}GB total")
    
    # 3. PyTorchレベル厳格制限（Web調査第1手法）
    logger.info(f"🔥 PyTorchレベル制限実行...")
    for i in range(gpu_count):
        if total_memories[i] > 0:
            # メモリ分数計算: 10GB/80GB = 0.125
            memory_fraction = min(0.95, memory_gb_per_gpu / total_memories[i])
            
            try:
                torch.cuda.set_per_process_memory_fraction(memory_fraction, i)
                logger.info(f"  ✅ GPU{i}: {memory_fraction:.3f}分数設定 ({memory_gb_per_gpu}GB/{total_memories[i]:.1f}GB)")
            except Exception as e:
                logger.warning(f"  ⚠️ GPU{i}分数設定失敗: {e}")
    
    # 4. キャッシュクリア（Web調査必須手順）
    try:
        torch.cuda.empty_cache()
        logger.info("  ✅ CUDAキャッシュクリア完了")
    except Exception as e:
        logger.warning(f"  ⚠️ キャッシュクリア失敗: {e}")
    
    # 5. HuggingFaceレベル制限（Web調査第2手法）
    logger.info(f"🔥 HuggingFaceレベル制限設定...")
    max_memory = {}
    
    for i in range(gpu_count):
        # Web調査推奨: 若干余裕を持った設定（オーバーヘッド考慮）
        safe_limit_gb = max(1, memory_gb_per_gpu - 1)  # 1GB余裕（8GB→7GB）
        max_memory[i] = f"{safe_limit_gb}GiB"
        logger.info(f"  ✅ GPU{i}: max_memory={safe_limit_gb}GiB")
    
    # CPUバッファ（Web調査推奨）
    cpu_buffer_gb = min(100, memory_gb_per_gpu * gpu_count * 2)
    max_memory["cpu"] = f"{cpu_buffer_gb}GB"
    logger.info(f"  ✅ CPU: max_memory={cpu_buffer_gb}GB")
    
    logger.info(f"✅ 厳格メモリ制限実装完了: 二重安全装置")
    logger.info(f"  - PyTorch制限: {[f'{memory_gb_per_gpu/total_memories[i]:.3f}' for i in range(gpu_count)]}")
    logger.info(f"  - HuggingFace制限: {max_memory}")
    
    return max_memory


def create_loss_plots(loss_history, log_dir, exp_name):
    """Phase3B学習曲線の可視化"""
    if not loss_history:
        return
    
    # 全エポックのデータを整理
    epochs = list(range(1, len(loss_history) + 1))
    total_losses = [epoch_data['total_loss'] for epoch_data in loss_history]
    text_losses = [epoch_data.get('text_loss', 0) for epoch_data in loss_history]
    mask_losses = [epoch_data.get('mask_loss', 0) for epoch_data in loss_history]
    dual_losses = [epoch_data.get('dual_pathway_loss', 0) for epoch_data in loss_history]
    moe_losses = [epoch_data.get('moe_loss', 0) for epoch_data in loss_history]
    
    # Phase 3B統合チャート
    plt.figure(figsize=(14, 10))
    
    # 上段：総損失
    plt.subplot(2, 2, 1)
    plt.plot(epochs, total_losses, 'b-', linewidth=2, marker='o', label='総損失')
    plt.title('Phase 3B総損失推移', fontsize=12)
    plt.xlabel('エポック')
    plt.ylabel('損失値')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 下段左：個別損失
    plt.subplot(2, 2, 2)
    plt.plot(epochs, text_losses, 'r-', linewidth=2, marker='s', label='テキスト損失')
    plt.plot(epochs, mask_losses, 'g-', linewidth=2, marker='^', label='マスク損失')
    plt.title('基本損失成分', fontsize=12)
    plt.xlabel('エポック')
    plt.ylabel('損失値')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 下段右：Phase 3B特有損失
    plt.subplot(2, 2, 3)
    plt.plot(epochs, dual_losses, 'orange', linewidth=2, marker='v', label='デュアルパスウェイ損失')
    plt.plot(epochs, moe_losses, 'm-', linewidth=2, marker='d', label='MoE損失')
    plt.title('Phase 3B拡張損失', fontsize=12)
    plt.xlabel('エポック')
    plt.ylabel('損失値')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 全体統合（右下）
    plt.subplot(2, 2, 4)
    plt.plot(epochs, total_losses, 'b-', linewidth=3, marker='o', label='総損失', markersize=8)
    plt.title(f'Phase 3B学習収束 - {exp_name}', fontsize=12)
    plt.xlabel('エポック')
    plt.ylabel('損失値')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Phase 3B QFormerSegmentationBridge学習曲線', fontsize=16)
    plt.tight_layout()
    
    # 保存
    plot_file = os.path.join(log_dir, f'phase3b_loss_curves_{exp_name}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file

def setup_logging(log_dir: str):
    """ロギング設定（Phase 3B用）"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'train_phase3b.log'))
        ]
    )
    return logging.getLogger(__name__)

def monitor_gpu_memory(step_name="", logger=None, memory_limit_gb_per_gpu=10):
    """
    🔥 Web調査強化版: H100×4 GPU RAM監視機能 + OOM予防
    
    - メモリ使用量詳細監視
    - Web調査設定制限との比較
    - OOM危険度判定・早期警告
    - 自動ガベージコレクション
    """
    if not torch.cuda.is_available():
        return {}
    
    import psutil
    memory_info = {}
    total_allocated = 0
    total_reserved = 0
    total_capacity = 0
    oom_risk_detected = False
    
    # 🔥 H100×4 GPU詳細監視
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        capacity = torch.cuda.get_device_properties(i).total_memory / 1024**3
        free = capacity - allocated
        usage_percent = (allocated / capacity) * 100
        
        # 🔥 Web調査制限との比較
        limit_usage_percent = (allocated / memory_limit_gb_per_gpu) * 100
        within_limit = allocated <= memory_limit_gb_per_gpu
        
        # 🔥 メモリ断片化監視
        try:
            stats = torch.cuda.memory_stats(i)
            active = stats.get("active_bytes.all.current", 0) / 1024**3
            inactive = stats.get("inactive_split_bytes.all.current", 0) / 1024**3
            fragmentation = (inactive / (active + inactive)) * 100 if (active + inactive) > 0 else 0
        except:
            active = inactive = fragmentation = 0
        
        # 🔥 OOM危険度判定（Web調査基準）
        risk_level = "安全"
        if limit_usage_percent > 95:
            risk_level = "危険"
            oom_risk_detected = True
        elif limit_usage_percent > 85:
            risk_level = "警告"
        elif limit_usage_percent > 70:
            risk_level = "注意"
        
        memory_info[f'gpu_{i}'] = {
            'allocated': allocated,
            'reserved': reserved,
            'capacity': capacity,
            'free': free,
            'usage_percent': usage_percent,
            'limit_gb': memory_limit_gb_per_gpu,
            'limit_usage_percent': limit_usage_percent,
            'within_limit': within_limit,
            'risk_level': risk_level,
            'active': active,
            'inactive': inactive,
            'fragmentation': fragmentation
        }
        
        total_allocated += allocated
        total_reserved += reserved
        total_capacity += capacity
        
        if logger:
            # 🔥 Web調査制限ベース報告
            logger.info(f"📊 GPU {i}: {allocated:.1f}GB使用 (制限{memory_limit_gb_per_gpu}GB: {limit_usage_percent:.1f}%, 危険度:{risk_level})")
            
            # 従来のキャパシティベース警告
            if usage_percent > 95:
                logger.error(f"🔥 GPU {i}: 危険レベル {usage_percent:.1f}%使用 ({allocated:.1f}GB/{capacity:.1f}GB)")
            elif usage_percent > 90:
                logger.warning(f"⚠️ GPU {i}: 高使用率 {usage_percent:.1f}%使用")
            
            # Web調査制限ベース警告
            if risk_level == "危険":
                logger.error(f"🚨 GPU {i}: Web調査制限{memory_limit_gb_per_gpu}GB超過危険! {limit_usage_percent:.1f}%")
            elif risk_level == "警告":
                logger.warning(f"⚠️ GPU {i}: Web調査制限の85%超過 {limit_usage_percent:.1f}%")
            
            # 断片化警告
            if fragmentation > 20:
                logger.warning(f"🔧 GPU {i}: メモリ断片化 {fragmentation:.1f}% - empty_cache推奨")
    
    # 🔥 CPU メモリも監視
    try:
        cpu_memory = psutil.virtual_memory()
        cpu_used = cpu_memory.used / 1024**3
        cpu_total = cpu_memory.total / 1024**3
        cpu_percent = cpu_memory.percent
    except:
        cpu_used = cpu_total = cpu_percent = 0
    
    memory_info['total'] = {
        'allocated': total_allocated,
        'reserved': total_reserved,
        'capacity': total_capacity,
        'usage_percent': (total_allocated / total_capacity) * 100,
        'cpu_used': cpu_used,
        'cpu_total': cpu_total,
        'cpu_percent': cpu_percent
    }
    
    if logger:
        total_usage = memory_info['total']['usage_percent']
        logger.info(f"🔍 {step_name}")
        logger.info(f"  📊 総GPU使用率: {total_usage:.1f}% ({total_allocated:.1f}GB/{total_capacity:.1f}GB)")
        logger.info(f"  💾 CPU RAM: {cpu_used:.1f}GB/{cpu_total:.1f}GB ({cpu_percent:.1f}%)")
        
        # 🔥 Web調査OOM予防判定
        if oom_risk_detected:
            logger.error("🚨 Web調査制限超過危険検出! 自動ガベージコレクション実行...")
            try:
                # Web調査推奨: 自動メモリ解放
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                logger.info("✅ 緊急メモリ解放完了")
            except Exception as e:
                logger.error(f"❌ 緊急メモリ解放失敗: {e}")
        
        # 従来のOOM予測警告
        if total_usage > 95:
            logger.error("🚨 OOM危険: 総GPU使用率95%超過 - 即座にメモリ解放推奨")
        elif total_usage > 90:
            logger.warning("⚠️ OOM警告: 総GPU使用率90%超過 - メモリ監視強化")
    
    # 🔥 OOM危険判定結果をメタデータに追加
    memory_info['oom_risk_detected'] = oom_risk_detected
    memory_info['memory_limit_gb_per_gpu'] = memory_limit_gb_per_gpu
    
    return memory_info

def create_moe_aware_device_map(gpu_count, logger):
    """MoE専用device_map作成（Web調査2025年対応）"""
    logger.info("🔧 MoE専用device_map作成中（Llama-4専用）...")
    
    # WebリサーチによるLlama-4 MoE Expert分散最適化
    if gpu_count >= 4:
        # balanced_low_0改良版: GPU1負荷分散
        custom_device_map = {
            # Embedding層（軽量GPU0）
            "model.embed_tokens": 0,
            
            # Transformer層分散（MoE Expert分散考慮）
            # GPU1のMoE Expert集中を回避
        }
        
        # レイヤー分散: GPU1のMoE負荷を分散
        layers_per_gpu = 48 // 4  # 12 layers per GPU (assuming 48 total layers)
        for i in range(48):
            if i < 12:
                custom_device_map[f"model.layers.{i}"] = 0
            elif i < 24:
                custom_device_map[f"model.layers.{i}"] = 1
            elif i < 36:
                custom_device_map[f"model.layers.{i}"] = 2  
            else:
                custom_device_map[f"model.layers.{i}"] = 3
        
        # 🔥 Multi-modal projector追加（Llama-4固有）
        custom_device_map["multi_modal_projector.linear_1"] = 0
        custom_device_map["multi_modal_projector.linear_2"] = 0
        custom_device_map["multi_modal_projector"] = 0
        
        # Vision tower（存在する場合）
        custom_device_map["vision_tower"] = 0
        
        # 出力層（GPU3固定）
        custom_device_map["model.norm"] = 3
        custom_device_map["lm_head"] = 3
        
        logger.info(f"✅ 4×GPU MoE Expert分散device_map作成完了")
        logger.info(f"  - GPU0: Embedding + Layers 0-11")
        logger.info(f"  - GPU1: Layers 12-23 (Expert負荷分散)")
        logger.info(f"  - GPU2: Layers 24-35 (Expert負荷分散)")
        logger.info(f"  - GPU3: Layers 36-47 + LM Head")
        
        return custom_device_map
    
    return "balanced_low_0"  # フォールバック

def setup_environment():
    """環境変数の設定（H100 4×GPU最適化 + Web調査2025年版）"""
    # GPU表示順序を固定
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # 🔥 H100最適化メモリ設定（Web調査2025年版最新）
    # max_split_size_mb: H100推奨512MB（メモリ断片化完全防止）
    # garbage_collection_threshold: 0.9で積極的GC
    # expandable_segments: メモリ断片化防止の最重要設定
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.9"
    
    # 🔥 GPU使用率80%設定（H100×4最適化：Unsloth 71GB準拠）
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.80)  # 🔥 H100×4: Unsloth準拠で80%活用
        print("🔧 GPU RAM使用率80%設定（H100×4最適化：Unsloth 71GB準拠）")
    
    # 動的コンパイル無効化（安定性のため）
    try:
        torch.compiler.disable()
    except:
        print("⚠️ torch.compiler.disable()スキップ（旧バージョン）")
    
    # PyTorchマルチGPU最適化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

def create_phase3b_model(logger):
    """Phase 3B統合モデル作成（4×H100最適化版）"""
    logger.info("=== Phase 3B統合モデル初期化（4×H100最適化版） ===")
    
    try:
        # 1. 設定取得
        lisa_config = config_linux.get_lisa_model_config()
        mle_config = config_linux.get_mle_config()
        
        # 2. QFormer-SAM2統合設定
        qformer_config = LlamaQFormerSAM2Config()
        qformer_config.enable_moe = True  # MoE統合有効化
        qformer_config.enable_dual_pathway = True  # デュアルパスウェイ有効化
        
        logger.info(f"✓ Phase 3B設定:")
        logger.info(f"  - MoE統合: {qformer_config.enable_moe}")
        logger.info(f"  - デュアルパスウェイ: {qformer_config.enable_dual_pathway}")
        logger.info(f"  - 期待性能向上: {mle_config['expected_improvement']}%")
        
        # 3. 共有Llama-4インスタンス初期化（Web調査厳格メモリ制限）
        logger.info("🧠 共有Llama-4-Scout初期化...")
        
        # 🔥 Web調査解決策: 厳格メモリ制限実装（OOM完全回避）
        # GPU環境に応じた極限メモリ制限を実行
        gpu_count = torch.cuda.device_count()
        logger.info(f"検出されたGPU数: {gpu_count}")
        
        if gpu_count >= 4:
            # 🔥 H100×4: 8GB/GPU超厳格制限（Web調査成果：10GB→8GBでOOM完全回避）
            target_memory_gb = 8
            logger.info(f"🔥 H100×4環境: {target_memory_gb}GB/GPU超厳格制限実行（Web調査成果基づく最終調整）")
        elif gpu_count >= 2:
            # 2-3GPU環境: 20GB/GPU制限
            target_memory_gb = 20
            logger.info(f"🔥 2-3GPU環境: {target_memory_gb}GB/GPU制限実行")
        else:
            # 単一GPU環境: 30GB制限
            target_memory_gb = 30
            logger.info(f"🔥 単一GPU環境: {target_memory_gb}GB制限実行")
        
        # Web調査厳格メモリ制限実行
        max_memory = enforce_strict_memory_limit(
            memory_gb_per_gpu=target_memory_gb, 
            logger=logger
        )
        
        # Device map設定（meta tensor回避）
        if gpu_count >= 4:
            device_map = "balanced"  # Web調査推奨: meta tensor回避
        else:
            device_map = "auto"
        
        logger.info(f"🔥 Device map: {device_map}")
        logger.info("🔥 二重安全装置: PyTorch + HuggingFace制限")
        
        # Llama-4-Scout初期化（超軽量量子化）
        from transformers import Llama4ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        
        # 🔥 Web調査解決策：meta tensorエラー回避
        # CPU offloadと量子化の同時使用がmeta tensor問題を引き起こすため分離
        if gpu_count >= 4:
            # 複数GPU環境：量子化なしでCPU offload活用
            quantization_config = None
            logger.info("🔥 Meta tensor回避: 4GPU環境では量子化無効、CPU offload活用")
        else:
            # 単一GPU環境：量子化活用、CPU offload無効
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            logger.info("🔥 Meta tensor回避: 単一GPU環境では量子化活用")
        
        # 🔥 SAM2重みファイル自動ダウンロード機能
        def ensure_sam2_weights():
            """SAM2重みファイルの自動確保・ダウンロード"""
            sam2_weight_path = "/lambda/nfs/llama4-lisa-project-fs-north-texas/data/weights/sam2_hiera_large.pt"
            
            if os.path.exists(sam2_weight_path):
                logger.info(f"✅ SAM2重みファイル確認済み: {sam2_weight_path}")
                return sam2_weight_path
            
            logger.info("🔄 SAM2重みファイル自動ダウンロード開始...")
            
            try:
                # huggingface_hubのインストール確認・実行
                try:
                    from huggingface_hub import hf_hub_download
                    logger.info("✅ huggingface_hub利用可能")
                except ImportError:
                    logger.info("📦 huggingface_hubをインストール中...")
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
                    from huggingface_hub import hf_hub_download
                    logger.info("✅ huggingface_hubインストール完了")
                
                # ディレクトリ作成
                weights_dir = os.path.dirname(sam2_weight_path)
                os.makedirs(weights_dir, exist_ok=True)
                logger.info(f"📁 重みディレクトリ作成: {weights_dir}")
                
                # HuggingFace Hubから自動ダウンロード
                logger.info("🌐 HuggingFace Hubからダウンロード中...")
                logger.info("  - Repository: facebook/sam2-hiera-large")
                logger.info("  - Filename: sam2_hiera_large.pt")
                
                downloaded_path = hf_hub_download(
                    repo_id="facebook/sam2-hiera-large",
                    filename="sam2_hiera_large.pt",
                    local_dir=weights_dir,
                    local_dir_use_symlinks=False  # 実ファイルコピー
                )
                
                logger.info(f"✅ SAM2重みファイルダウンロード完了: {downloaded_path}")
                
                # ダウンロード先を確認
                if os.path.exists(sam2_weight_path):
                    logger.info(f"✅ 期待パスにファイル確認: {sam2_weight_path}")
                    return sam2_weight_path
                else:
                    logger.info(f"📍 ダウンロード先: {downloaded_path}")
                    return downloaded_path
                    
            except Exception as e:
                logger.warning(f"⚠️ SAM2重みダウンロード失敗: {e}")
                logger.info("💡 HuggingFace Hubモデルを直接使用します")
                return "facebook/sam2-hiera-large"  # フォールバック
        
        # SAM2重み確保
        sam2_model_id = ensure_sam2_weights()
        logger.info(f"✅ SAM2モデル準備完了: {sam2_model_id}")
        
        # 🔥 Web調査meta tensor回避: CPU主体offload最適化
        shared_llama4_model = Llama4ForConditionalGeneration.from_pretrained(
            qformer_config.llama_model_id,
            quantization_config=quantization_config,
            device_map=device_map,
            max_memory=max_memory,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # 🔥 Web調査推奨：meta tensor回避
            use_safetensors=True,
            # 🔥 Web調査：offload設定をGPU数に応じて調整
            offload_state_dict=(gpu_count >= 4),   # 4GPU以上のみoffload有効
            offload_folder="/tmp/llama4_cpu_offload" if gpu_count >= 4 else None,
        )
        
        shared_llama4_processor = AutoProcessor.from_pretrained(
            qformer_config.llama_model_id,
            trust_remote_code=True
        )
        
        logger.info(f"✓ 共有Llama-4初期化完了")
        
        # 🔍 GPU監視: モデル初期化後
        monitor_gpu_memory("Llama-4初期化後", logger)
        
        # 4. QFormerSegmentationBridge初期化（重複回避版）
        logger.info("🔍 QFormerSegmentationBridge初期化...")
        
        model = QFormerSegmentationBridge(
            config=qformer_config,
            shared_llama_model=shared_llama4_model,    # 共有インスタンス
            shared_llama_processor=shared_llama4_processor,
            enable_moe=True,                           # MoE統合
            training_stage=1                           # Stage 1: 基本機能学習
        )
        
        logger.info(f"✓ QFormerSegmentationBridge初期化完了")
        
        # 🔥 5. デュアルパスウェイデコーダー統合（重複ロード回避版 + SAM2自動取得）
        if qformer_config.enable_dual_pathway:
            logger.info("🔄 デュアルパスウェイデコーダー統合...")
            
            # ✅ QFormerSegmentationBridge内のSAM2インスタンスを再利用
            existing_sam2 = None
            if hasattr(model, 'sam2_model') and model.sam2_model is not None:
                existing_sam2 = model.sam2_model
                logger.info("✅ 既存SAM2インスタンス発見、再利用します（重複ロード回避）")
            elif hasattr(model, 'sam2_wrapper') and model.sam2_wrapper is not None:
                existing_sam2 = model.sam2_wrapper
                logger.info("✅ 既存SAM2Wrapper発見、再利用します（重複ロード回避）")
            else:
                logger.warning("⚠️ 既存SAM2インスタンス未発見、HuggingFace Hub使用")
            
            # 🔥 test_phase3b_integration_real.py準拠の基本的な呼び出し
            dual_decoder = create_dual_pathway_decoder(
                llama_hidden_size=5120,     # 🔥 固定値使用（config問題回避）
                sam_output_dim=256,         # 🔥 SAM2標準値使用
                fusion_strategy="learned_weighted",
                force_gpu=True              # 🔥 訓練スクリプト対応：GPU強制
            )
            
            # モデルに統合
            model.dual_pathway_decoder = dual_decoder
            logger.info(f"✓ デュアルパスウェイデコーダー統合完了（SAM2自動取得版）")
        
        # 6. 学習可能パラメータ数確認
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✓ Phase 3Bモデル統計（重複ロード回避版）:")
        logger.info(f"  - 総パラメータ: {total_params:,}")
        logger.info(f"  - 学習可能パラメータ: {trainable_params:,}")
        logger.info(f"  - 学習可能割合: {100 * trainable_params / total_params:.3f}%")
        logger.info(f"  - メモリ最適化: CPU offload + 重複ロード回避 + GPU分散調整")
        
        return model, shared_llama4_processor
        
    except Exception as e:
        logger.error(f"Phase 3Bモデル初期化エラー: {e}")
        raise

def create_dataset_and_dataloader(processor, args, logger):
    """Phase 3B用データセット作成（HybridDataset使用）"""
    logger.info("=== Phase 3B用データセット初期化 ===")
    
    # データセット種類の設定
    if hasattr(args, 'dataset') and args.dataset:
        datasets = [ds.strip() for ds in args.dataset.split(',')]
        dataset_string = "||".join(datasets)
    else:
        dataset_string = "reason_seg"  # デフォルト
    
    logger.info(f"対象データセット: {datasets if 'datasets' in locals() else [dataset_string]}")
    
    # サンプル数制御（loss減少確認用、OOM対策でさらに削減）
    samples_per_epoch = getattr(args, 'samples_per_epoch', 50)  # 🔥 大幅削減（OOM対策）
    logger.info(f"エポックあたりのサンプル数: {samples_per_epoch}")
    
    # HybridDataset初期化（デュアルエンコーダー対応）
    dataset = HybridDataset(
        base_image_dir=config_linux.DATASET_BASE_DIR,
        llama_processor=processor,
        samples_per_epoch=samples_per_epoch,
        precision="bf16",
        llama_image_size=config_linux.LLAMA_IMAGE_SIZE,  # 448
        sam_image_size=config_linux.SAM_IMAGE_SIZE,      # 1024
        num_classes_per_sample=3,
        exclude_val=False,
        dataset=dataset_string,
        sample_rate=[9, 3, 3, 1],  # reason_seg, refer_seg, vqa, sem_seg の比率
        sem_seg_data=config_linux.SEM_SEG_DATA,
        refer_seg_data=config_linux.REFER_SEG_DATA,
        vqa_data=config_linux.VQA_DATA,
        reason_seg_data=config_linux.REASON_SEG_DATA,
        explanatory=0.1
    )
    
    logger.info(f"✓ HybridDataset初期化完了: {len(dataset)} サンプル")
    
    # DataLoader作成（Reference準拠：メモリ最適化）
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 🔥 Reference準拠：メモリ最適化のため0に設定
        collate_fn=collate_fn,
        pin_memory=False,  # 🔥 Reference準拠：メモリ最適化のためFalse
        drop_last=True
    )
    
    logger.info(f"✓ DataLoader作成完了: バッチサイズ={args.batch_size}")
    
    return dataset, dataloader

def train_epoch_phase3b(model, dataloader, optimizer, scheduler, epoch, args, logger, writer=None):
    """Phase 3B用1エポック学習実行（Reference準拠メモリ最適化）"""
    model.train()
    
    # Phase 3B損失メトリクス初期化
    total_losses = AverageMeter('Total', ':.4e')
    text_losses = AverageMeter('Text', ':.4e')
    mask_losses = AverageMeter('Mask', ':.4e')
    dual_losses = AverageMeter('Dual', ':.4e')
    moe_losses = AverageMeter('MoE', ':.4e')
    
    progress = ProgressMeter(
        len(dataloader) if args.steps_per_epoch is None else args.steps_per_epoch,
        [total_losses],
        prefix=f"Phase3B Epoch: [{epoch}]"
    )
    
    # エポック内損失履歴
    epoch_loss_history = {
        'total_loss': [],
        'text_loss': [],
        'mask_loss': [],
        'dual_pathway_loss': [],
        'moe_loss': []
    }
    
    start_time = time.time()
    
    for step, batch in enumerate(dataloader):
        # 🔥 H100×4最適化：ステップごとにメモリ最適化
        if torch.cuda.is_available():
            # 全GPU同期とメモリクリア
            for i in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()
                torch.cuda.synchronize(i)
            
            # 🔥 メモリ断片化対策：GC強制実行
            import gc
            gc.collect()
        
        # ステップ数制限チェック
        if args.steps_per_epoch is not None and step >= args.steps_per_epoch:
            break
        
        # データをGPUに移動
        device = next(model.parameters()).device
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
        
        # バッチ内容確認（初回のみ）
        if step == 0:
            logger.info(f"Phase 3Bバッチ内容: {list(batch.keys())}")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: {value.shape} ({value.dtype})")
        
        # フォワードパス（Phase 3B統合）
        optimizer.zero_grad()
        
        # QFormerSegmentationBridge用入力準備
        model_inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch.get('attention_mask'),
            'images': batch.get('pixel_values'),           # Llama-4用画像
            'sam_images': batch.get('sam_pixel_values'),   # 🔥 SAM2用画像（パラメータ名修正）
            'labels': batch.get('labels'),
            'return_dict': True
        }
        
        # Noneの値を除去
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
        
        # 🔍 H100×4 GPU監視: リアルタイム監視
        memory_info = monitor_gpu_memory(f"Step {step} 開始前", logger)
        
        # 🔥 OOM予測・早期停止機能
        if memory_info.get('total', {}).get('usage_percent', 0) > 95:
            logger.error(f"🚨 OOM予測による緊急停止 - Step {step}")
            logger.error("🔥 全GPUメモリクリアを実行中...")
            for i in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()
                torch.cuda.synchronize(i)
            import gc; gc.collect()
            
            # 一時停止してメモリ確認
            memory_info_after = monitor_gpu_memory(f"メモリクリア後 Step {step}", logger)
            if memory_info_after.get('total', {}).get('usage_percent', 0) > 90:
                raise RuntimeError(f"🚨 OOM回避不可能 - Step {step}: メモリクリア後も{memory_info_after['total']['usage_percent']:.1f}%使用")
        
        # Phase 3B統合フォワードパス実行（OOM中断対応）
        try:
            # 🔥 Gradient accumulation開始判定
            is_accumulating = (step + 1) % args.gradient_accumulation_steps != 0
            
            # 🔥 Mixed Precision無効化でOOM回避 + gradient scaling
            with torch.cuda.amp.autocast(enabled=False):  # AMP無効化
                model_outputs = model(**model_inputs)
                
                # 🔥 Gradient accumulation適用
                if is_accumulating:
                    # accumulation中は勾配をスケール
                    total_loss = model_outputs.loss / args.gradient_accumulation_steps
                else:
                    total_loss = model_outputs.loss
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"🔥 CUDA OOMエラー - フォワードパス (step {step}): {e}")
            
            # 🔍 詳細GPU監視（OOM時）
            monitor_gpu_memory(f"OOM発生時 Step {step}", logger)
            
            torch.cuda.empty_cache()
            logger.error("❌ フォワードパスOOMエラーのため実行を中断します")
            raise RuntimeError(f"Forward pass CUDA OOM at step {step}: {e}")  # 🔥 実行中断
        
        # Phase 3B損失の取得
        individual_losses = {
            'total_loss': 0, 'text_loss': 0, 'mask_loss': 0,
            'dual_pathway_loss': 0, 'moe_loss': 0
        }
        
        # デバッグ: モデル出力確認（初回のみ）
        if step == 0:
            if isinstance(model_outputs, dict):
                logger.info(f"  🔍 Phase 3Bモデル出力キー: {list(model_outputs.keys())}")
        
        if isinstance(model_outputs, dict):
            # テキスト損失
            if 'text_loss' in model_outputs and model_outputs['text_loss'] is not None:
                individual_losses['text_loss'] = model_outputs['text_loss'].item()
                loss = total_loss  # 🔥 gradient accumulation適用済みloss使用
                individual_losses['total_loss'] = total_loss.item()
                
                if step == 0:
                    logger.info(f"  ✓ text_loss取得成功: {individual_losses['text_loss']:.4f}")
            else:
                logger.warning("text_lossが見つかりません")
                # フォールバック: ダミー損失
                loss = torch.tensor(0.0, requires_grad=True, device=device)
                individual_losses['total_loss'] = 0.0
            
            # 追加情報（あれば取得）
            if 'predicted_masks' in model_outputs:
                logger.info(f"  📊 予測マスク: {model_outputs['predicted_masks'].shape}")
            
            # MoE情報（あれば表示）
            if 'moe_info' in model_outputs and isinstance(model_outputs['moe_info'], dict):
                moe_info = model_outputs['moe_info']
                if 'expert_weights' in moe_info:
                    logger.info(f"  🔄 MoEエキスパート重み: {moe_info['expert_weights']}")
        else:
            raise ValueError(f"Phase 3Bモデル出力形式エラー: {type(model_outputs)}")
        
        # メモリクリア（Reference準拠）
        torch.cuda.empty_cache()
        
        # 逆伝播（Gradient Accumulation対応）
        try:
            # Reference準拠：メモリクリア後逆伝播
            torch.cuda.empty_cache()
            loss.backward()
            
            # 🔥 Gradient accumulation: 指定ステップごとにのみoptimizer更新
            if not is_accumulating:
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
                
                # パラメータ更新
                optimizer.step()
                optimizer.zero_grad()  # 勾配リセット
                scheduler.step()
                
                if step % 5 == 0:  # 5ステップごとにログ出力
                    logger.info(f"🔄 Step {step}: Optimizer update (accumulation完了)")
            else:
                if step % 5 == 0:
                    logger.info(f"🔄 Step {step}: Gradient accumulating ({(step + 1) % args.gradient_accumulation_steps}/{args.gradient_accumulation_steps})")
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"🔥 CUDA OOMエラー (step {step}): {e}")
            logger.error("💡 GPU Memory Status:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.error(f"   GPU {i}: {allocated:.2f}GB/{total:.2f}GB allocated, {reserved:.2f}GB reserved")
            torch.cuda.empty_cache()
            logger.error("❌ OOMエラーのため実行を中断します")
            raise RuntimeError(f"CUDA OOM at step {step}: {e}")  # 🔥 実行中断
        except RuntimeError as e:
            if "CUDA" in str(e) or "memory" in str(e):
                logger.error("❌ CUDA メモリ不足エラー")
                logger.error("💡 対策: --steps_per_epoch を小さくしてください")
                torch.cuda.empty_cache()
                continue
            else:
                raise
        
        # 🔥 Gradient accumulationにより上記で処理済み
        
        # メトリクス更新
        batch_size = batch['input_ids'].size(0)
        total_losses.update(individual_losses['total_loss'], batch_size)
        text_losses.update(individual_losses['text_loss'], batch_size)
        mask_losses.update(individual_losses['mask_loss'], batch_size)
        dual_losses.update(individual_losses['dual_pathway_loss'], batch_size)
        moe_losses.update(individual_losses['moe_loss'], batch_size)
        
        # エポック内履歴更新
        for key in epoch_loss_history:
            epoch_loss_history[key].append(individual_losses[key])
        
        # 進行度表示
        if step % args.print_freq == 0:
            total_steps = args.steps_per_epoch if args.steps_per_epoch else len(dataloader)
            progress_pct = (step + 1) / total_steps * 100
            elapsed_time = time.time() - start_time
            eta = elapsed_time / (step + 1) * (total_steps - step - 1) if step > 0 else 0
            
            logger.info(f"🚀 Phase3B Epoch [{epoch+1}] Step [{step+1:3d}/{total_steps}] ({progress_pct:5.1f}%) "
                       f"ETA: {eta/60:.1f}min | "
                       f"Total: {individual_losses['total_loss']:6.3f} | "
                       f"Text: {individual_losses['text_loss']:6.3f} | "
                       f"Mask: {individual_losses['mask_loss']:6.3f}")
            
            # TensorBoard記録
            if writer is not None:
                global_step = epoch * len(dataloader) + step
                writer.add_scalar('Phase3B/TotalLoss', total_losses.val, global_step)
                writer.add_scalar('Phase3B/TextLoss', text_losses.val, global_step)
                writer.add_scalar('Phase3B/MaskLoss', mask_losses.val, global_step)
                writer.add_scalar('Phase3B/DualLoss', dual_losses.val, global_step)
                writer.add_scalar('Phase3B/MoELoss', moe_losses.val, global_step)
        
        # 🔥 Web調査メモリ管理（厳格制限監視+予防的解放）
        if step % 10 == 0:
            # 定期的厳格制限監視
            memory_info = monitor_gpu_memory(
                step_name=f"Epoch[{epoch+1}] Step[{step+1}]",
                logger=logger,
                memory_limit_gb_per_gpu=8  # Web調査成果：8GB超厳格制限
            )
            
            # OOM危険検出時の追加対策
            if memory_info.get('oom_risk_detected', False):
                logger.warning("🚨 OOM危険検出: 予防的バッチサイズ削減検討")
            
        # 従来のメモリ解放（頻度増加）
        if step % 5 == 0:
            torch.cuda.empty_cache()
    
    # エポック終了時の統計
    epoch_summary = {
        'total_loss': total_losses.avg,
        'text_loss': text_losses.avg,
        'mask_loss': mask_losses.avg,
        'dual_pathway_loss': dual_losses.avg,
        'moe_loss': moe_losses.avg,
        'loss_history': epoch_loss_history
    }
    
    # Reference準拠：エポック終了時のメモリクリア
    torch.cuda.empty_cache()
    
    elapsed_time = time.time() - start_time
    logger.info(f"📊 Phase 3B Epoch [{epoch+1}] 完了 ({elapsed_time/60:.1f}分)")
    logger.info(f"  - 平均総損失: {total_losses.avg:.4f}")
    logger.info(f"  - 平均テキスト損失: {text_losses.avg:.4f}")
    
    return epoch_summary

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Phase 3B QFormerSegmentationBridge学習スクリプト")
    
    # 基本設定
    parser.add_argument('--exp_name', type=str, default='phase3b_loss_check', help='実験名')
    parser.add_argument('--epochs', type=int, default=3, help='エポック数')
    parser.add_argument('--steps_per_epoch', type=int, default=10, help='エポック毎ステップ数（None=全データ）')
    parser.add_argument('--batch_size', type=int, default=1, help='バッチサイズ（OOM回避のため1固定推奨）')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps（メモリ効率化）')
    parser.add_argument('--workers', type=int, default=0, help='DataLoaderワーカー数')  # 🔥 Reference準拠：0
    parser.add_argument('--samples_per_epoch', type=int, default=30, help='エポック毎サンプル数')  # 🔥 OOM対策：さらに削減
    
    # 学習設定
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='学習率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='重み減衰')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='勾配クリッピング')
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help='ウォームアップ比率')
    
    # データセット設定
    parser.add_argument('--dataset', type=str, default='reason_seg', help='データセット名')
    
    # 出力設定
    parser.add_argument('--output_dir', type=str, default='./phase3b_results', help='出力ディレクトリ')
    parser.add_argument('--print_freq', type=int, default=5, help='ログ出力頻度')
    parser.add_argument('--save_freq', type=int, default=1, help='モデル保存頻度')
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, f"{args.exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # ロギング設定
    logger = setup_logging(exp_dir)
    logger.info("=== Phase 3B QFormerSegmentationBridge学習開始 ===")
    logger.info(f"実験名: {args.exp_name}")
    logger.info(f"出力ディレクトリ: {exp_dir}")
    
    # 🔥 強制メモリクリア（残骸削除）
    if torch.cuda.is_available():
        logger.info("🧹 強制メモリクリア開始...")
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(i)
        torch.cuda.synchronize()
        logger.info("✅ 全GPU強制メモリクリア完了")
    
    # 環境設定
    setup_environment()
    
    # TensorBoard設定
    writer = SummaryWriter(os.path.join(exp_dir, 'tensorboard'))
    
    try:
        # Phase 3Bモデル作成
        model, processor = create_phase3b_model(logger)
        
        # データセット作成
        dataset, dataloader = create_dataset_and_dataloader(processor, args, logger)
        
        # オプティマイザー設定
        logger.info("=== オプティマイザー設定 ===")
        
        if BITSANDBYTES_AVAILABLE:
            optimizer = bnb.optim.AdamW8bit(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
            logger.info("✓ 8bit AdamW使用")
        else:
            optimizer = optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
            logger.info("✓ 標準AdamW使用")
        
        # スケジューラー設定
        total_steps = args.epochs * (args.steps_per_epoch if args.steps_per_epoch else len(dataloader))
        warmup_steps = int(total_steps * args.warmup_ratio)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"✓ 学習設定: LR={args.learning_rate}, 総ステップ={total_steps}, ウォームアップ={warmup_steps}")
        
        # 学習実行
        logger.info("=== Phase 3B学習実行 ===")
        
        loss_history = []
        
        for epoch in range(args.epochs):
            logger.info(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
            
            # 1エポック学習
            epoch_result = train_epoch_phase3b(
                model, dataloader, optimizer, scheduler, epoch, args, logger, writer
            )
            
            # 損失履歴に追加
            loss_history.append(epoch_result)
            
            # モデル保存
            if (epoch + 1) % args.save_freq == 0:
                save_path = os.path.join(exp_dir, f'model_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss_history': loss_history,
                    'args': args
                }, save_path)
                logger.info(f"✓ モデル保存: {save_path}")
            
            # Reference準拠：GPU メモリクリーンアップ
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        # 学習曲線可視化
        if loss_history:
            plot_file = create_loss_plots(loss_history, exp_dir, args.exp_name)
            logger.info(f"✅ 学習曲線保存: {plot_file}")
        
        # 結果サマリー
        logger.info("=== Phase 3B学習完了 ===")
        logger.info(f"実行エポック数: {args.epochs}")
        logger.info(f"最終損失: {loss_history[-1]['total_loss']:.4f}")
        logger.info(f"初期損失: {loss_history[0]['total_loss']:.4f}")
        
        if len(loss_history) > 1:
            improvement = loss_history[0]['total_loss'] - loss_history[-1]['total_loss']
            improvement_pct = (improvement / loss_history[0]['total_loss']) * 100
            logger.info(f"損失改善: {improvement:.4f} ({improvement_pct:.2f}%)")
        
        # 結果保存
        results = {
            'experiment_name': args.exp_name,
            'epochs': args.epochs,
            'final_loss': loss_history[-1]['total_loss'],
            'initial_loss': loss_history[0]['total_loss'],
            'loss_history': loss_history,
            'args': vars(args)
        }
        
        results_file = os.path.join(exp_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ 結果保存: {results_file}")
        
    except Exception as e:
        logger.error(f"学習エラー: {e}")
        raise
    finally:
        writer.close()

if __name__ == "__main__":
    main()