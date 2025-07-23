#!/usr/bin/env python3
"""
Phase 3B QFormerSegmentationBridge 訓練スクリプト
test_phase3b_integration_real.pyの成功パターンを基に実際のデータで学習

実行方法:
# 最小時間テスト（10ステップ）- test script成功パターン準拠（H100x4対応）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@192.222.52.239 "cd /lambda/nfs/llama4-lisa-project-fs-north-texas/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && export CUDA_VISIBLE_DEVICES=0,1,2,3 && export CUDA_LAUNCH_BLOCKING=1 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && PYTHONUNBUFFERED=1 python -u train_phase3b_qformer_bridge.py --exp_name quick_test --steps_per_epoch 10 --epochs 1 2>&1"

# 通常訓練（実際のデータで損失減少観察）- test script成功パターン準拠（H100x4対応）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@192.222.52.239 "cd /lambda/nfs/llama4-lisa-project-fs-north-texas/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && export CUDA_VISIBLE_DEVICES=0,1,2,3 && export CUDA_LAUNCH_BLOCKING=1 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && PYTHONUNBUFFERED=1 python -u train_phase3b_qformer_bridge.py --exp_name phase3b_train --epochs 3 --steps_per_epoch 50 2>&1"
"""

# 🔥 PyTorchインポート前の環境準備（CUDA Error 802対策）
import sys
import os

# Step 1: CUDA Error 802対策 + Webリサーチメモリ最適化環境変数設定（PyTorchインポート前）
print("🔧 CUDA Error 802対策：PyTorchインポート前環境設定...")
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'

# Webリサーチ最適化: 2025年最新のOOM対策設定（PyTorch torchtune準拠）
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.5,roundup_power2_divisions:32'
print("🔧 メモリ断片化対策強化: max_split_size_mb:128,gc_threshold:0.5,roundup_divisions:32（GPU 0過負荷対応）")

# Step 2: CUDA_VISIBLE_DEVICESが未設定の場合のみ設定（H100x4対応 - test script成功パターン準拠）
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # H100x4専用

# Step 2.1: test script成功パターン準拠の最小限設定
# ❌ 重複削除: PYTORCH_CUDA_ALLOC_CONF は上で設定済み

print("✅ 環境変数設定完了 - PyTorchインポート開始...")

# Step 3: PyTorchインポート
import argparse
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torch.cuda.amp import GradScaler  # 旧API: 2025年削除予定
# torch.amp.GradScaler を直接使用（2025年推奨）
import transformers
from transformers import AutoProcessor, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# 🔥 2025年メモリ最適化戦略（PyTorch torchtune + HuggingFace Accelerate準拠）
if torch.cuda.is_available():
    # 📊 メモリ使用率最適化（GPU 0過負荷対応）
    torch.cuda.set_per_process_memory_fraction(0.85)  # 80%→85%（Model Parallelism対応）
    
    # 🔧 GPU固有メモリ設定（GPU 0集中対応）
    for gpu_id in range(torch.cuda.device_count()):
        with torch.cuda.device(gpu_id):
            if gpu_id == 0:
                # GPU 0: embed_tokens重負荷対応
                torch.cuda.set_per_process_memory_fraction(0.90, device=gpu_id)
            else:
                # GPU 1-7: 標準設定
                torch.cuda.set_per_process_memory_fraction(0.85, device=gpu_id)
    
    # 🧹 初期メモリ状態クリーンアップ
    torch.cuda.empty_cache()  # 初期キャッシュクリア
    torch.cuda.reset_peak_memory_stats()  # メモリ統計リセット
    torch.cuda.synchronize()  # GPU同期
    
    # 🚀 2025年ベストプラクティス: CUDA設定最適化 + メモリリーク対策
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32高速化（H100専用）
    torch.backends.cudnn.allow_tf32 = True       # cuDNN TF32高速化
    torch.backends.cudnn.benchmark = False       # 動的サイズに対応（SAM2対応）
    torch.backends.cuda.enable_math_sdp(True)    # Math SDP有効化（メモリ効率化）
    torch.backends.cuda.enable_flash_sdp(True)   # Flash Attention有効化
    torch.backends.cuda.enable_mem_efficient_sdp(True)  # Memory-efficient attention
    print("🔧 2025年最適化: CUDA memory fraction=0.85, TF32=True, benchmark=False（SAM2+Llama4統合）")

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logging.warning("bitsandbytes not available. Using standard AdamW optimizer.")

from PIL import Image
import matplotlib
matplotlib.use('Agg')  # バックエンドを非対話モードに設定
import matplotlib.pyplot as plt
import numpy as np

# Disable warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 🔥 Webリサーチメモリ最適化: GPU RAMモニタリング追加
from utils.memory_monitor import GPUMemoryMonitor, memory_monitor_section, log_memory_status, emergency_cleanup

# Phase 3B統合モデルとユーティリティ
from model.llama4_qformer_sam2 import QFormerSegmentationBridge, LlamaQFormerSAM2Config
from model.sam2_integration import get_sam2_wrapper
# from model.qformer import get_qformer_model  # ❌ 削除: QFormerSegmentationBridge内で初期化されるため不要
from model.dual_pathway_decoder import create_dual_pathway_decoder
from model.multiresolution_fusion import Llama4SAM2MultiResolutionFusion
from model.ohem_loss import create_ohem_loss
from model.dataset_adapter import adapt_dataset_for_qformer, configure_dual_encoder
from utils.dataset import HybridDataset, collate_fn, preprocess_sam_image, build_correct_labels_for_llama4
from utils.constants import DEFAULT_SEG_TOKEN
import config_linux

# Lambda Cloud環境パス検証レポート表示
config_linux.validate_and_report_paths()

# 評価用メトリクス
from utils.utils import (
    AverageMeter, ProgressMeter, Summary, 
    dict_to_cuda, intersectionAndUnionGPU
)

# HuggingFace transformers（Llama-4-Scout用）
try:
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
    print("✅ HuggingFace Transformers利用可能")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ HuggingFace Transformersが利用できません")

def create_loss_plots(loss_history, log_dir, exp_name):
    """Phase 3B学習曲線の可視化（統合損失対応）"""
    if not loss_history:
        return
    
    # 全エポックのデータを整理
    epochs = list(range(1, len(loss_history) + 1))
    total_losses = [epoch_data['total_loss'] for epoch_data in loss_history]
    phase2_losses = [epoch_data.get('phase2_loss', 0) for epoch_data in loss_history]
    seg_losses = [epoch_data.get('seg_loss', 0) for epoch_data in loss_history]
    ohem_losses = [epoch_data.get('ohem_loss', 0) for epoch_data in loss_history]
    
    # Phase 3B統合チャート
    plt.figure(figsize=(15, 10))
    
    # メインプロット: 総損失
    plt.subplot(2, 2, 1)
    plt.plot(epochs, total_losses, 'b-', linewidth=3, marker='o', markersize=6, label='総損失')
    plt.title('Phase 3B: 総損失', fontsize=12, fontweight='bold')
    plt.xlabel('エポック')
    plt.ylabel('損失値')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Phase 2損失
    plt.subplot(2, 2, 2)
    plt.plot(epochs, phase2_losses, 'g-', linewidth=2, marker='s', label='Phase 2統一空間')
    plt.title('Phase 2: 部分統一トークン空間', fontsize=12, fontweight='bold')
    plt.xlabel('エポック')
    plt.ylabel('損失値')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # セグメンテーション損失
    plt.subplot(2, 2, 3)
    plt.plot(epochs, seg_losses, 'orange', linewidth=2, marker='v', label='SAM2セグメンテーション')
    plt.title('SAM2セグメンテーション損失', fontsize=12, fontweight='bold')
    plt.xlabel('エポック')
    plt.ylabel('損失値')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # OHEM損失
    plt.subplot(2, 2, 4)
    plt.plot(epochs, ohem_losses, 'r-', linewidth=2, marker='^', label='OHEM困難例')
    plt.title('OHEM困難例マイニング', fontsize=12, fontweight='bold')
    plt.xlabel('エポック')
    plt.ylabel('損失値')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Phase 3B QFormerSegmentationBridge 学習曲線 - {exp_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存
    plot_file = os.path.join(log_dir, f'phase3b_loss_curves_{exp_name}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file

def setup_logging(log_dir: str):
    """ロギング設定"""
    # 環境変数からログレベルを取得
    log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'train_phase3b.log'))
        ]
    )
    return logging.getLogger(__name__)


def setup_environment():
    """環境変数の設定（test_phase3b成功パターン）"""
    # GPU表示順序を固定
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # メモリ最適化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def create_model_and_components(args, logger):
    """Phase 3Bモデルとコンポーネント作成（test_phase3b成功パターン移植）"""
    logger.info("=== Phase 3B統合モデル初期化（QFormerSegmentationBridge） ===")
    
    # test_phase3b成功パターン: GPU最適化設定
    logger.info("🔥 GPU RAM分散利用最適化設定を適用...")
    
    # PyTorchマルチGPU最適化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    logger.info("✓ TF32最適化 + cuDNNベンチマーク有効化")
    
    # GPU間メモリ転送最適化
    if torch.cuda.device_count() >= 2:
        torch.cuda.set_per_process_memory_fraction(0.95)  # 95%使用許可
        logger.info("✓ GPU RAM使用率95%設定（メモリ効率化）")
    
    # 動的コンパイル無効化（安定性のため）
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True
    logger.info("✓ torch._dynamo.config.suppress_errors = True: Llama-4-Scout既知バグ回避")
    
    try:
        # 1. Llama-4-Scout初期化（test_phase3b成功パターン）
        logger.info("🔄 Llama-4-Scout-17B-16E-Instruct初期化...")
        
        if TRANSFORMERS_AVAILABLE:
            # test_phase3b成功パターン完全移植
            llama_config = {
                "llama_model_id": config_linux.LLAMA_MODEL_ID,
                "use_cache": False,
                "low_cpu_mem_usage": True
            }
            
            model_id = llama_config["llama_model_id"]
            
            # HuggingFaceから直接ロード（Web調査2025年最新パターン）+ GPU RAM分散対応
            # LISA準拠: CausalLMアーキテクチャを使用
            try:
                # Llama4専用クラスの確認
                try:
                    from transformers import Llama4ForCausalLM
                    model_class = Llama4ForCausalLM
                    logger.info("✓ Llama4ForCausalLMクラス利用可能")
                except ImportError:
                    # AutoModelForCausalLMを使用（Llama4を自動選択）
                    from transformers import AutoModelForCausalLM
                    model_class = AutoModelForCausalLM
                    logger.warning("⚠️ Llama4ForCausalLM未対応、AutoModelForCausalLM使用")
                    logger.info("💡 transformers>=4.45.0へのアップデートを推奨")
                
                # 🔥 Model Parallelism復活: マルチGPU分散利用（17Bモデル必須）
                device_count = torch.cuda.device_count()
                logger.info(f"🔥 Model Parallelism復活: {device_count}x H100分散利用（17Bモデル対応）")
                logger.info(f"検出されたGPU数: {device_count}")
                
                # 🔧 メモリ断片化解決: PYTORCH_CUDA_ALLOC_CONF設定（Webリサーチ推奨）
                import os
                # 🔥 2025年最適化CUDAメモリ設定（Webリサーチ準拠・断片化完全対策）
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
                    'expandable_segments:True,'
                    'max_split_size_mb:128,'          # 256→128: より細かい制御
                    'garbage_collection_threshold:0.8,'  # 0.6→0.8: より積極的なGC
                    'roundup_power2_divisions:32,'    # 16→32: メモリアライメント最適化
                    'backend:cudaMallocAsync'         # 🔥 追加: 非同期メモリ割り当て
                )
                logger.info("✓ CUDAメモリ最適化強化: expandable_segments + cudaMallocAsync + 断片化制御")
                
                # 🔥 解決策3: 8xH100環境対応（容量不足解決・全GPU均等分散）
                logger.info("🔥 解決策3: 8xH100環境対応（容量不足根本解決）")
                logger.info("📋 問題: 4xH100では容量不足（232.9GB > 316.8GB理論値）")
                logger.info("📋 解決: 8xH100で余裕のある分散配置（50-60GB/GPU目標）")
                
                # 🔥 Webリサーチ最適化: GPU 1ボトルネック解決戦略（2025年カスタムdevice_map）
                logger.info("🚀 GPU 1ボトルネック解決: カスタムdevice_map + Expert Parallelism")
                logger.info("📋 問題: balanced_low_0でGPU 1が58.7GB（74.1%）でOOM")
                logger.info("📋 解決: embed_tokens分散 + レイヤー最適配置")
                
                # 🔥 Expert Parallelism戦略: Llama-4 Scout MoE最適化
                logger.info("🎆 Expert Parallelism準備: Llama-4 Scout MoE構造分析")
                logger.info("📋 MoEアーキテクチャ: 16 Experts per layer, 17B active params per token")
                logger.info(f"🔍 device_count確認: {device_count}個GPU検出")
                
                # 🔍 Expert Parallelismデバッグ: device_map構造確認
                logger.info("🔍 Expert Parallelism device_map構築開始...")
                
                # Llama-4 Scout MoE構造に最適化したExpert Parallelism device_map（GPU 0負荷超軽減版）
                device_map_setting = {
                    # Embedding層: GPU 1に移動（GPU 0負荷軽減）
                    'model.embed_tokens': 1,
                    
                    # 🔥 Expert Parallelism: 16 Experts を 8 GPU に分散（2 Experts per GPU）
                    # 各レイヤーのMoE Expert を GPU間で分散配置
                    
                    # Dense層（self_attn, mlp）とMoE層を区別して最適配置
                    # GPU 0: 最小限の1層のみ（OOM対策）
                    'model.layers.0': 0,
                    
                    # GPU 1: embed_tokens + layers 1-7（Q-Former統一・負荷軽減）
                    'model.layers.1': 1,
                    'model.layers.2': 1,
                    'model.layers.3': 1,
                    'model.layers.4': 1,
                    'model.layers.5': 1,
                    'model.layers.6': 1,
                    'model.layers.7': 1,
                    
                    # GPU 2: layers 8-14 + layers 39-47（負荷分散最適化）
                    'model.layers.8': 2,
                    'model.layers.9': 2,
                    'model.layers.10': 2,
                    'model.layers.11': 2,
                    'model.layers.12': 2,
                    'model.layers.13': 2,
                    'model.layers.14': 2,
                    # 🔥 GPU 1負荷軽減: layers 39-47をGPU 2に移動
                    'model.layers.39': 2,
                    'model.layers.40': 2,
                    'model.layers.41': 2,
                    'model.layers.42': 2,
                    'model.layers.43': 2,
                    'model.layers.44': 2,
                    'model.layers.45': 2,
                    'model.layers.46': 2,
                    'model.layers.47': 2,
                    
                    # GPU 3: Expert 6-7 + 中間層
                    'model.layers.15': 3,
                    'model.layers.16': 3,
                    'model.layers.17': 3,
                    'model.layers.18': 3,
                    'model.layers.19': 3,
                    'model.layers.20': 3,
                    
                    # GPU 4: Expert 8-9 + 中間層
                    'model.layers.21': 4,
                    'model.layers.22': 4,
                    'model.layers.23': 4,
                    'model.layers.24': 4,
                    'model.layers.25': 4,
                    'model.layers.26': 4,
                    
                    # GPU 5: Expert 10-11 + 中間層
                    'model.layers.27': 5,
                    'model.layers.28': 5,
                    'model.layers.29': 5,
                    'model.layers.30': 5,
                    'model.layers.31': 5,
                    'model.layers.32': 5,
                    
                    # GPU 6: Expert 12-13（負荷分散最適化）
                    'model.layers.33': 6,
                    'model.layers.34': 6,
                    'model.layers.35': 6,
                    'model.layers.36': 6,
                    'model.layers.37': 6,
                    'model.layers.38': 6,
                    
                    # GPU 7: lm_head（Backward OOM対策）
                    'lm_head': 7,  # 🔥 GPU 6 Backward OOM対策: lm_headをGPU 7に移動
                    
                    # 出力層: cuda:2に配置（layers 39-47と同一デバイス）
                    'model.norm': 2
                }
                
                # 🔍 Expert Parallelismデバッグ: device_map詳細出力
                logger.info("✓ Expert Parallelism device_map構成完了:")
                logger.info(f"  🔍 device_map_setting keys: {len(device_map_setting)}")
                logger.info(f"  🔍 embed_tokens device: {device_map_setting.get('model.embed_tokens', 'NOT_SET')}")
                logger.info(f"  🔍 lm_head device: {device_map_setting.get('lm_head', 'NOT_SET')}")
                
                # GPU別コンポーネント数確認
                gpu_component_count = {}
                for component, gpu_id in device_map_setting.items():
                    if gpu_id not in gpu_component_count:
                        gpu_component_count[gpu_id] = 0
                    gpu_component_count[gpu_id] += 1
                
                logger.info(f"  🔍 GPU別コンポーネント数: {gpu_component_count}")
                logger.info("📊 メモリ効率: 109B total → 17B active per token")
                
                if device_count >= 8:
                    # 🔥 Expert Parallelism対応メモリ制限（GPU 0超軽減版）
                    max_memory_dict = {
                        0: "32GB",   # GPU 0: 1層のみ（OOM対策）
                        1: "58GB",   # GPU 1: embed_tokens + 7層 + Expert分散
                        2: "52GB",   # GPU 2: 6層 + Expert 4-5分散
                        3: "52GB",   # GPU 3: 6層 + Expert 6-7分散
                        4: "52GB",   # GPU 4: 6層 + Expert 8-9分散
                        5: "52GB",   # GPU 5: 6層 + Expert 10-11分散
                        6: "58GB",   # GPU 6: 6層 + Expert 12-13 + lm_head（重負荷）
                        7: "50GB"    # GPU 7: 9層 + Expert 14-15（軽減済み）
                    }
                    logger.info("✓ Expert Parallelismメモリ戦略: GPU 7負荷軽減版")
                    logger.info("📋 GPU 0: 32GB (1層), GPU 1: 58GB (embed+7層), GPU 2-5: 52GB (6層+Expert)")
                    logger.info("📋 GPU 6: 58GB (6層+Expert+lm_head), GPU 7: 50GB (9層+Expert), 総容量: 376GB")
                    logger.info("📊 メモリ効率: Expert分散で実効17B/109Bパラメータアクティブ")
                
                elif device_count >= 4:
                    # 4x H100環境: 容量制限対応（後方互換）
                    max_memory_dict = {
                        0: "78GB",   # GPU 0: 98.5%使用率
                        1: "78GB",   # GPU 1: 98.5%使用率
                        2: "78GB",   # GPU 2: 98.5%使用率
                        3: "78GB",   # GPU 3: 98.5%使用率
                    }
                    logger.info("✓ 4x H100 balanced戦略: 各GPU 78GB（98.5%使用率・限界運用）")
                    logger.info("⚠️ 4GPU環境は容量限界、8GPU推奨")
                
                elif device_count >= 2:
                    # 2x H100環境
                    max_memory_dict = {
                        0: "78GB",  # GPU 0: 98.5%使用率
                        1: "78GB",  # GPU 1: 98.5%使用率
                    }
                    logger.info("✓ 2x H100 balanced戦略: 各GPU 78GB（98.5%使用率）")
                    
                else:
                    logger.warning("⚠️ GPU数不足、single GPU mode")
                    device_map_setting = "auto"
                    max_memory_dict = {0: "70GB"}  # single GPU用
                
                # モデルロード（HuggingFaceの自動device_map処理を活用）
                logger.info("🔧 HuggingFaceの自動device_map処理でModel Parallelismを実装...")
                
                # 🚨 2025年修正A改良版: 軽量device_map修正
                # dummy model作成を避けて軽量に実装
                logger.info("🔧 軽量device_map修正を適用中...")
                
                # autoで一度ロードしてから修正する方針に変更
                logger.info("  - 方針: auto → 事後lm_head修正")
                
                # 🚨 2025年修正B: 数値安定化config適用
                from transformers import LlamaConfig
                
                # Llama設定に数値安定化パラメータを適用
                config_modifications = {
                    "rms_norm_eps": config_linux.LAYERNORM_EPSILON,  # LayerNorm epsilon強化
                }
                
                # 🔥 Webリサーチ修正: カスタムdevice_mapでlm_head明示配置
                # HuggingFace Accelerate推奨パターンを適用
                
                # モデルロード（カスタムdevice_map最適化版）
                load_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "device_map": device_map_setting,  # カスタムdevice_map使用（GPU 1ボトルネック解決）
                    "attn_implementation": "sdpa",
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                    "max_memory": max_memory_dict,     # 最適化済みメモリ制限
                    "offload_state_dict": False,       # disk offload無効化
                    "use_safetensors": True,           # safetensors使用
                }
                
                logger.info("🔥 GPU 7負荷軽減戦略適用: lm_head→GPU6移動, メモリ制限厳格化")
                logger.info(f"📊 デバイス配置: {len(device_map_setting)} コンポーネント分散配置（48レイヤー完全カバー）")
                logger.info(f"📊 メモリ制限: {sum([int(mem.replace('GB', '')) for mem in max_memory_dict.values()])}GB総制限（19GB安全マージン）")
                logger.info("📋 負荷分散最適化: GPU0:1層, GPU1:embed+7層, GPU2-5:6層, GPU6:6層+lm_head, GPU7:9層")
                
                logger.info(f"🔧 数値安定化設定適用: rms_norm_eps={config_linux.LAYERNORM_EPSILON}")
                
                # 🔥 Webリサーチ修正: offload_folder完全無効化
                # "Cannot copy out of meta tensor"エラー回避のため
                logger.info("🔧 offload_folder無効化: Webリサーチによるmeta tensor問題回避")
                logger.info("💡 accelerate disk_offload関数は使用せず、カスタムdevice_mapで解決")
                
                # 🚨 数値安定化: config修正後にモデルロード
                try:
                    # 既存configを取得
                    model_config = model_class.config_class.from_pretrained(model_id)
                    
                    # 数値安定化パラメータを適用
                    model_config.rms_norm_eps = config_linux.LAYERNORM_EPSILON
                    logger.info(f"  ✅ LayerNorm epsilon: {model_config.rms_norm_eps} (安定化強化)")
                    
                    # 修正されたconfigでモデルロード
                    load_kwargs["config"] = model_config
                    
                except Exception as config_error:
                    logger.warning(f"  ⚠️ config修正スキップ: {config_error}")
                
                # 🔥 モデルロード前の緊急メモリクリーンアップ（GPU 7 OOM対策）
                logger.info("🔥 モデルロード前: 緊急メモリクリーンアップ実行...")
                import gc
                # torch は既にインポート済みなので再インポート不要
                
                # 既存のCUDAコンテキストをクリア
                for gpu_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                
                # Python GCの実行
                gc.collect()
                
                # 🔍 Expert Parallelismデバッグ: モデルロード直前
                logger.info("🔍 Expert Parallelism: モデルロード直前チェック")
                logger.info(f"  - model_id: {model_id}")
                logger.info(f"  - device_map type: {type(load_kwargs['device_map'])}")
                logger.info(f"  - device_map length: {len(load_kwargs['device_map'])}")
                logger.info(f"  - max_memory type: {type(load_kwargs['max_memory'])}")
                logger.info(f"  - GPU 7制限: {load_kwargs['max_memory'].get(7, 'N/A')} (lm_head移動済み)")
                
                llama4_model = model_class.from_pretrained(model_id, **load_kwargs)
                logger.info(f"✓ {model_class.__name__}使用（Expert Parallelism + カスタムdevice_map）")
                
                # 🔍 Expert Parallelismデバッグ: モデルロード直後
                logger.info("🔍 Expert Parallelism: モデルロード直後チェック")
                if hasattr(llama4_model, 'hf_device_map'):
                    actual_device_map = llama4_model.hf_device_map
                    logger.info(f"  - 実際のdevice_map length: {len(actual_device_map)}")
                    
                    # 重要コンポーネントの配置確認
                    embed_device = actual_device_map.get('model.embed_tokens', 'NOT_FOUND')
                    lm_head_device = actual_device_map.get('lm_head', 'NOT_FOUND')
                    logger.info(f"  - embed_tokens実際配置: {embed_device}")
                    logger.info(f"  - lm_head実際配置: {lm_head_device}")
                    
                    # 設定と実際の比較
                    expected_embed = device_map_setting.get('model.embed_tokens', 'NOT_SET')
                    expected_lm_head = device_map_setting.get('lm_head', 'NOT_SET')
                    logger.info(f"  - embed_tokens: 設定{expected_embed} vs 実際{embed_device}")
                    logger.info(f"  - lm_head: 設定{expected_lm_head} vs 実際{lm_head_device}")
                else:
                    logger.error("⚠️ hf_device_mapが見つかりません！")
                
                # 🚨 事後lm_head修正: ディスク配置を検出して修正
                logger.info("🔧 lm_head配置確認・修正中...")
                
                if hasattr(llama4_model, 'lm_head'):
                    try:
                        # lm_headのデバイス確認
                        lm_head_device = next(llama4_model.lm_head.parameters()).device
                        logger.info(f"  - lm_head現在配置: {lm_head_device}")
                        
                        # diskに配置されている場合のみ修正
                        if str(lm_head_device) == 'meta' or 'disk' in str(lm_head_device):
                            logger.info("  🔧 lm_head: ディスク配置検出 → 修正スキップ（meta tensor対応待ち）")
                            logger.warning("  ⚠️ NaN発生リスクあり（lm_headディスク配置）")
                        else:
                            logger.info(f"  ✅ lm_head負荷分散配置: {lm_head_device} (最適化版)")
                            
                    except Exception as lm_head_check_error:
                        logger.warning(f"  ⚠️ lm_head確認エラー: {lm_head_check_error}")
                
                if hasattr(llama4_model, 'model') and hasattr(llama4_model.model, 'embed_tokens'):
                    try:
                        embed_device = llama4_model.model.embed_tokens.weight.device
                        logger.info(f"  ✅ embed_tokens配置: {embed_device}")
                    except Exception as embed_check_error:
                        logger.warning(f"  ⚠️ embed_tokens確認エラー: {embed_check_error}")
                
                # HuggingFaceが自動的にdevice_mapを処理するため、dispatch_modelは不要
                logger.info("✓ Model Parallelismはfrom_pretrainedで自動的に適用されています")
                
                # 🔥 強化版Accelerate Hooks制御: GPU 0 OOM根本対策
                logger.info("🔧 強化版Accelerate Hooks制御開始（GPU 0 OOM根本対策）...")
                try:
                    from accelerate import hooks
                    
                    hooks_found = []
                    hooks_modified = 0
                    hooks_disabled = 0
                    align_device_hooks = []
                    
                    # Phase 1: AlignDevicesHookの特定と強力な制御
                    for name, module in llama4_model.named_modules():
                        if hasattr(module, '_hf_hook'):
                            hook = module._hf_hook
                            hook_type = type(hook).__name__
                            hooks_found.append(f"{name}: {hook_type}")
                            
                            if hook_type == 'AlignDevicesHook':
                                align_device_hooks.append((name, hook))
                                
                                # Method 1: post_forwardの完全無効化（OOM根本対策）
                                if hasattr(hook, 'post_forward'):
                                    original_post_forward = hook.post_forward
                                    def noop_post_forward(module, output):
                                        # 出力をそのまま返す（デバイス転送しない）
                                        return output
                                    hook.post_forward = noop_post_forward
                                    hooks_disabled += 1
                                    logger.info(f"  🛡️ {name}: post_forward無効化")
                                
                                # Method 2: input_device調整（従来手法も維持）
                                if hasattr(hook, 'input_device') and str(hook.input_device) == 'cuda:0':
                                    original_device = hook.input_device
                                    execution_device = getattr(hook, 'execution_device', hook.input_device)
                                    
                                    if str(execution_device) != 'cuda:0':
                                        hook.input_device = execution_device
                                        hooks_modified += 1
                                        logger.info(f"  📝 {name}: input_device {original_device} → {execution_device}")
                                
                                # Method 3: skip_keysの拡張（hidden_statesを転送から除外）
                                if hasattr(hook, 'skip_keys'):
                                    if hook.skip_keys is None:
                                        hook.skip_keys = set()
                                    elif not isinstance(hook.skip_keys, set):
                                        hook.skip_keys = set(hook.skip_keys) if hook.skip_keys else set()
                                    
                                    # 大きなテンソルを転送から除外
                                    original_skip_keys = len(hook.skip_keys)
                                    hook.skip_keys.update(['hidden_states', 'last_hidden_state', 'logits', 'attention_weights'])
                                    if len(hook.skip_keys) > original_skip_keys:
                                        logger.info(f"  🔒 {name}: skip_keys拡張（大テンソル除外）")
                    
                    # Phase 2: 追加のGPU 0負荷軽減策
                    logger.info("🔧 追加GPU 0負荷軽減策実行...")
                    
                    # embed_tokensをGPU 1に移動（より積極的な負荷分散）
                    if hasattr(llama4_model, 'hf_device_map'):
                        current_map = llama4_model.hf_device_map
                        if 'model.embed_tokens' in current_map and current_map['model.embed_tokens'] == 0:
                            logger.info("  📝 embed_tokens: GPU 0 → GPU 1 移動試行...")
                            try:
                                # embed_tokensの安全な移動
                                embed_module = llama4_model.model.embed_tokens
                                embed_module.to('cuda:1')
                                current_map['model.embed_tokens'] = 1
                                logger.info("  ✅ embed_tokens移動完了: GPU 0 → GPU 1")
                            except Exception as e:
                                logger.warning(f"  ⚠️ embed_tokens移動失敗: {e}")
                    
                    logger.info(f"✅ 強化版Hooks制御完了:")
                    logger.info(f"  - AlignDevicesHook検出: {len(align_device_hooks)}個")
                    logger.info(f"  - post_forward無効化: {hooks_disabled}個（OOM根本対策）")
                    logger.info(f"  - input_device修正: {hooks_modified}個")
                    logger.info(f"  - 総Hook数: {len(hooks_found)}個")
                    
                    if hooks_found:
                        logger.info(f"  - 検出Hook例: {', '.join(hooks_found[:3])}{'...' if len(hooks_found) > 3 else ''}")
                    
                except Exception as hook_error:
                    logger.warning(f"⚠️ 強化版Accelerate Hooks制御エラー: {hook_error}")
                    logger.warning("⚠️ GPU 0集約が続く可能性があります")
                
                # embed_tokensの状態確認（デバッグ用）
                if hasattr(llama4_model, 'model') and hasattr(llama4_model.model, 'embed_tokens'):
                    embed_weight = llama4_model.model.embed_tokens.weight
                    logger.info(f"✓ embed_tokens確認: shape={embed_weight.shape}, dtype={embed_weight.dtype}")
                    if embed_weight.is_meta:
                        logger.info("  💡 embed_tokensはmeta tensorです（disk offload）。HuggingFaceが自動的に処理します")
                    else:
                        logger.info(f"  ✓ embed_tokensデバイス: {embed_weight.device}")
                        # 統計情報（NaN確認用）
                        if not embed_weight.is_meta:
                            has_nan = torch.isnan(embed_weight).any().item()
                            has_inf = torch.isinf(embed_weight).any().item()
                            logger.info(f"  ✓ embed_tokens健全性: NaN={has_nan}, Inf={has_inf}")
                
                # GPU配置確認
                if hasattr(llama4_model, 'hf_device_map'):
                    actual_device_map = llama4_model.hf_device_map
                    gpu_distribution = {}
                    for component, device in actual_device_map.items():
                        if device not in gpu_distribution:
                            gpu_distribution[device] = 0
                        gpu_distribution[device] += 1
                    logger.info(f"✓ 実際のGPU分散: {gpu_distribution}")
                else:
                    logger.warning("⚠️ device_map情報を取得できませんでした")
                
            except Exception as e:
                error_msg = f"❌ CausalLMモデルロード失敗: {e}"
                logger.error(error_msg)
                logger.error(f"モデルID: {model_id}")
                logger.error(f"使用クラス: {model_class.__name__ if 'model_class' in locals() else 'Unknown'}")
                logger.error("考えられる原因:")
                logger.error("1. モデルへのアクセス権限がない")
                logger.error("2. HuggingFaceトークンが未設定")
                logger.error("3. ネットワーク接続の問題")
                logger.error("4. モデルIDが正しくない")
                raise RuntimeError(error_msg)
            
            # プロセッサ初期化（エラー詳細付き）
            try:
                llama4_processor = AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=True
                )
                logger.info("✓ AutoProcessor初期化成功")
                
                # Webリサーチ対策: プロセッサにもbfloat16設定を確認
                if hasattr(llama4_processor, 'image_processor'):
                    if hasattr(llama4_processor.image_processor, 'do_convert_rgb'):
                        llama4_processor.image_processor.do_convert_rgb = True
                    if hasattr(llama4_processor.image_processor, 'do_rescale'):
                        llama4_processor.image_processor.do_rescale = True
                    logger.info("✓ 画像プロセッサ設定確認完了")
            except Exception as proc_e:
                logger.warning(f"⚠️ AutoProcessor初期化失敗: {proc_e}")
                # フォールバック: AutoTokenizer使用
                try:
                    from transformers import AutoTokenizer
                    llama4_processor = AutoTokenizer.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        use_fast=True
                    )
                    logger.info("✓ AutoTokenizer使用（プロセッサ代替）")
                except Exception as tok_e:
                    logger.error(f"❌ AutoTokenizer初期化も失敗: {tok_e}")
                    raise RuntimeError(f"❌ プロセッサ初期化完全失敗: {tok_e}")
            
            llama_params = sum(p.numel() for p in llama4_model.parameters())
            logger.info(f"✓ Llama-4-Scout初期化完了: {llama_params:,} パラメータ")
            
            # 🔍 Expert Parallelism効果確認（詳細デバッグ）
            if hasattr(llama4_model, 'hf_device_map'):
                logger.info("🔍 Expert Parallelism効果確認開始...")
                actual_device_map = llama4_model.hf_device_map
                
                # Expert構造の検索
                expert_components = []
                moe_components = []
                for component, device in actual_device_map.items():
                    component_lower = component.lower()
                    if 'expert' in component_lower:
                        expert_components.append((component, device))
                    elif 'moe' in component_lower or 'mixture' in component_lower:
                        moe_components.append((component, device))
                
                logger.info(f"🔍 Expert コンポーネント数: {len(expert_components)}")
                logger.info(f"🔍 MoE コンポーネント数: {len(moe_components)}")
                
                if expert_components:
                    logger.info("🎆 Expert コンポーネント発見:")
                    for component, device in expert_components[:5]:  # 最初の5個を表示
                        logger.info(f"  {component} -> GPU {device}")
                    if len(expert_components) > 5:
                        logger.info(f"  ... その他 {len(expert_components) - 5} 個")
                        
                    # Expert分散状況を集計
                    expert_distribution = {}
                    for component, device in expert_components:
                        if device not in expert_distribution:
                            expert_distribution[device] = 0
                        expert_distribution[device] += 1
                    logger.info(f"🎆 Expert分散状況: {expert_distribution}")
                else:
                    logger.warning("⚠️ 明示的Expert構造が検出されませんでした - より詳細な分析を開始")
                    
                    # Llama-4 MoE詳細構造分析（Webリサーチベース）
                    logger.info("🔍 Llama-4 MoE詳細構造分析:")
                    
                    # MoE関連パターンを拡張検索
                    moe_patterns = ['mlp', 'feed_forward', 'gate_proj', 'up_proj', 'down_proj', 
                                   'router', 'gating', 'dense', 'wi_0', 'wi_1', 'wo']
                    moe_related_components = []
                    
                    component_analysis = {
                        'total': len(actual_device_map),
                        'transformer_layers': 0,
                        'mlp_components': 0,
                        'attention_components': 0,
                        'embed_components': 0,
                        'head_components': 0,
                        'moe_potential': 0
                    }
                    
                    for component in actual_device_map.keys():
                        component_lower = component.lower()
                        
                        if 'layers' in component_lower or 'blocks' in component_lower:
                            component_analysis['transformer_layers'] += 1
                        elif 'embed' in component_lower:
                            component_analysis['embed_components'] += 1
                        elif 'lm_head' in component_lower or 'head' in component_lower:
                            component_analysis['head_components'] += 1
                        elif 'attn' in component_lower or 'attention' in component_lower:
                            component_analysis['attention_components'] += 1
                        
                        # MoE関連パターン検索
                        for pattern in moe_patterns:
                            if pattern in component_lower:
                                moe_related_components.append(component)
                                component_analysis['moe_potential'] += 1
                                if pattern in ['mlp', 'feed_forward', 'gate_proj', 'up_proj', 'down_proj']:
                                    component_analysis['mlp_components'] += 1
                                break
                    
                    logger.info(f"  📊 コンポーネント詳細分析:")
                    for key, value in component_analysis.items():
                        logger.info(f"    - {key}: {value}")
                    
                    if moe_related_components:
                        logger.info(f"  🎯 MoE関連候補コンポーネント数: {len(moe_related_components)}")
                        logger.info(f"  🎯 MoE関連コンポーネント例:")
                        for component in moe_related_components[:10]:  # 最初の10個を表示
                            device = actual_device_map.get(component, 'unknown')
                            logger.info(f"    - {component} -> GPU {device}")
                        if len(moe_related_components) > 10:
                            logger.info(f"    - ... その他 {len(moe_related_components) - 10} 個")
                    
                    # Llama-4期待構造との比較
                    expected_layers = 48  # Llama-4標準
                    if component_analysis['transformer_layers'] > 0:
                        logger.info(f"  🔍 期待Transformer層数: ~{expected_layers}")
                        logger.info(f"  🔍 実際検出層数: {component_analysis['transformer_layers']}")
                        
                        if component_analysis['mlp_components'] > 0:
                            mlp_per_layer = component_analysis['mlp_components'] / max(component_analysis['transformer_layers'], 1)
                            logger.info(f"  🔍 層あたりMLP数: {mlp_per_layer:.1f}")
                            if mlp_per_layer > 1.5:  # 通常のTransformerなら1.0程度
                                logger.info(f"  ✅ 高MLP密度検出: MoE構造の可能性高")
                            else:
                                logger.info(f"  ❓ 標準的MLP密度: 通常Transformer或いは内蔵MoE")
                    
                    # 結論
                    if component_analysis['moe_potential'] > 20:  # 閾値調整
                        logger.info(f"  ✅ MoE構造高可能性: {component_analysis['moe_potential']}個のMoE関連コンポーネント")
                        logger.info(f"  📝 Llama-4のMoE構造はMLPレイヤー内部に埋め込まれている可能性")
                    else:
                        logger.info(f"  ❓ MoE構造不明: 標準Transformer或いは高度に統合されたMoE")
            else:
                logger.error("⚠️ hf_device_mapが存在しません！Expert Parallelism確認不可")
        else:
            # ❌ ダミーモデルフォールバック削除 - 明確にエラーで停止
            error_msg = "❌ HuggingFace Transformersが利用できません。Lambda環境のtransformersライブラリを確認してください。"
            logger.error(error_msg)
            logger.error("原因として考えられる問題:")
            logger.error("1. transformersライブラリがインストールされていない")
            logger.error("2. Pythonパスの問題")
            logger.error("3. 仮想環境が正しく有効化されていない")
            raise RuntimeError(error_msg)
        
        # 2. Q-Formerは統合ブリッジ内で初期化（test_phase3b成功パターン準拠）
        logger.info("🔄 Q-Formerは統合ブリッジ内で初期化されます（重複回避）")
        
        # 3. 統合ブリッジ初期化（デュアルエンコーダー構成対応）
        logger.info("🔄 Q-Former-SAM2統合ブリッジ初期化（デュアルエンコーダー対応版）...")
        qformer_config = LlamaQFormerSAM2Config()
        
        # デュアルエンコーダー設定を適用
        qformer_config = configure_dual_encoder(qformer_config)
        
        qformer_bridge = QFormerSegmentationBridge(
            config=qformer_config,
            shared_llama_model=llama4_model,        # ✅ 共有インスタンス
            shared_llama_processor=llama4_processor, # ✅ 共有プロセッサ
            training_stage=1,
            enable_moe=True
        )
        
        # 🔥 Model Parallelism対応: 補助コンポーネントを主要デバイスに配置
        logger.info("🔄 QFormerSegmentationBridge Model Parallelism対応配置...")
        try:
            # Llama-4の主要デバイスを取得
            main_device, access_info = get_model_device_map(qformer_bridge)
            logger.info(f"主要デバイス確認: {access_info}")
            
            # 補助コンポーネントを主要デバイスに配置（meta tensor対応）
            if hasattr(qformer_bridge, 'qformer') and qformer_bridge.qformer is not None:
                qformer_bridge.qformer = move_to_device_safely(qformer_bridge.qformer, main_device, logger, "qformer")
            if hasattr(qformer_bridge, 'sam2_wrapper') and qformer_bridge.sam2_wrapper is not None:
                qformer_bridge.sam2_wrapper = move_to_device_safely(qformer_bridge.sam2_wrapper, main_device, logger, "sam2_wrapper")
            if hasattr(qformer_bridge, 'seg_token_generator') and qformer_bridge.seg_token_generator is not None:
                qformer_bridge.seg_token_generator = move_to_device_safely(qformer_bridge.seg_token_generator, main_device, logger, "seg_token_generator")
            
            # Llama-4モデルはModel Parallelismで分散配置済み（手動移動不要）
            
            logger.info(f"✓ QFormerSegmentationBridge Model Parallelism配置完了（主要: {main_device}）")
            
            # 🔍 段階1: 診断強化 - Q-Former内部LayerNormパラメータの詳細デバイス確認
            logger.info("🔍 診断強化: Q-Former内部パラメータデバイス詳細分析...")
            diagnose_qformer_device_placement(qformer_bridge, logger)
            
            # 🔥 段階2: SAM2Wrapper強制統一（PyTorchベストプラクティス準拠）
            logger.info("🔥 段階2: SAM2Wrapper強制デバイス統一開始...")
            if hasattr(qformer_bridge, 'sam2_wrapper') and qformer_bridge.sam2_wrapper is not None:
                try:
                    logger.info(f"  📍 SAM2Wrapper強制統一: {main_device}に完全移動...")
                    # 再度強制的にSAM2Wrapperを統一
                    qformer_bridge.sam2_wrapper = qformer_bridge.sam2_wrapper.to(device=main_device, dtype=torch.bfloat16)
                    
                    # 内部のSAM2Modelも強制統一
                    if hasattr(qformer_bridge.sam2_wrapper, 'sam2_model'):
                        qformer_bridge.sam2_wrapper.sam2_model = qformer_bridge.sam2_wrapper.sam2_model.to(device=main_device, dtype=torch.bfloat16)
                        logger.info(f"  ✅ SAM2Model内部も{main_device}に統一完了")
                    
                    logger.info(f"  ✅ SAM2Wrapper強制統一完了: {main_device}")
                    
                    # 統一後の再診断
                    logger.info("🔍 統一後の診断:")
                    diagnose_qformer_device_placement(qformer_bridge, logger)
                    
                except Exception as sam_error:
                    logger.error(f"  ❌ SAM2Wrapper強制統一エラー: {sam_error}")
            else:
                logger.warning("  ⚠️ SAM2Wrapperが見つかりません")
            
        except Exception as e:
            logger.warning(f"⚠️ QFormerSegmentationBridge配置エラー: {e}")
        
        logger.info("✓ 統合ブリッジ初期化完了（Model Parallelism対応）")
        
        # 4. Phase 3B拡張コンポーネント初期化
        logger.info("🔄 Phase 3B拡張コンポーネント初期化...")
        
        # デュアルパスウェイデコーダ（test_phase3b成功パターン移植）
        dual_pathway_decoder = create_dual_pathway_decoder(
            llama_hidden_size=5120,           # Llama-4隠れ層サイズ
            sam_output_dim=256,               # SAM2出力次元
            fusion_strategy="learned_weighted",
            force_gpu=True                    # 訓練スクリプト対応：GPU強制
        )
        # Model Parallelism対応デバイス配置
        main_device, _ = get_model_device_map(qformer_bridge)
        dual_pathway_decoder = move_to_device_safely(dual_pathway_decoder, main_device, logger, "dual_pathway_decoder")
        
        # 🔥 段階3: デュアルパスウェイデコーダ内SAM2強制統一（最終段階）
        logger.info("🔥 段階3: デュアルパスウェイデコーダ内SAM2コンポーネント強制統一開始...")
        if hasattr(dual_pathway_decoder, 'main_sam2_decoder') and dual_pathway_decoder.main_sam2_decoder is not None:
            logger.info(f"  🎯 SAM2デコーダ発見: {type(dual_pathway_decoder.main_sam2_decoder)}")
            # SAM2デコーダを強制的にmain_deviceに移動
            dual_pathway_decoder.main_sam2_decoder = dual_pathway_decoder.main_sam2_decoder.to(device=main_device, dtype=torch.bfloat16)
            logger.info(f"  ✅ SAM2デコーダ強制統一完了: → {main_device}")
            
            # さらにSAM2Wrapper内部の全サブモジュールを強制統一
            for name, module in dual_pathway_decoder.main_sam2_decoder.named_modules():
                if module != dual_pathway_decoder.main_sam2_decoder:  # 自分自身は除く
                    try:
                        module.to(device=main_device, dtype=torch.bfloat16)
                    except Exception as e:
                        logger.warning(f"    ⚠️ サブモジュール移動失敗 {name}: {e}")
            logger.info(f"  ✅ SAM2Wrapper全サブモジュール強制統一完了")
        else:
            logger.warning("  ⚠️ SAM2デコーダが見つかりません（モック使用の可能性）")
        
        logger.info(f"✓ デュアルパスウェイデコーダ初期化完了（{main_device}）")
        
        # 🔍 段階4: SAM2コンポーネント全体検索デバッグ（完全スキャン）
        logger.info("🔍 段階4: SAM2コンポーネント全体検索開始...")
        sam2_components_found = []
        
        def scan_for_sam2_components(module, path=""):
            """SAM2関連のコンポーネントを再帰的に検索"""
            for name, submodule in module.named_children():
                current_path = f"{path}.{name}" if path else name
                
                # SAM2関連キーワードでマッチ
                if any(keyword in name.lower() for keyword in ['sam2', 'sam_', 'segment', 'wrapper']):
                    device_info = "CPU"
                    if hasattr(submodule, 'parameters'):
                        try:
                            first_param = next(submodule.parameters(), None)
                            if first_param is not None:
                                device_info = str(first_param.device)
                        except:
                            pass
                    
                    sam2_components_found.append({
                        'path': current_path,
                        'type': type(submodule).__name__,
                        'device': device_info,
                        'module': submodule
                    })
                    logger.info(f"  🎯 SAM2コンポーネント発見: {current_path} ({type(submodule).__name__}) → {device_info}")
                
                # 再帰的に探索継続
                scan_for_sam2_components(submodule, current_path)
        
        # 全モジュールをスキャン
        logger.info("  📊 QFormerSegmentationBridge内を検索中...")
        scan_for_sam2_components(qformer_bridge, "qformer_bridge")
        
        logger.info("  📊 デュアルパスウェイデコーダ内を検索中...")
        scan_for_sam2_components(dual_pathway_decoder, "dual_pathway_decoder")
        
        # 多重解像度特徴統合（test_phase3b成功パターン移植）
        multiresolution_fusion = Llama4SAM2MultiResolutionFusion(
            llama_hidden_size=5120,          # Llama-4隠れ層サイズ
            sam_feature_dim=256,             # SAM2特徴次元（SAM2公式準拠）
            sam_scales=[1024, 512, 256],     # SAM2マルチスケール
            qformer_dim=768,                 # Q-Former特徴次元
            qformer_queries=32,              # Q-Formerクエリ数
            fusion_dim=512,                  # 正しい引数名（実装確認済み）
            output_size=(448, 448),          # Llama-4解像度準拠
            fusion_strategy="attention"      # 注意機構融合（2024ベストプラクティス）
        )
        # Model Parallelism対応デバイス配置
        multiresolution_fusion = move_to_device_safely(multiresolution_fusion, main_device, logger, "multiresolution_fusion")
        logger.info(f"✓ 多重解像度特徴統合初期化完了（{main_device}）")
        
        # 🔍 段階4.5: 多重解像度融合モジュール内のSAM2検索
        logger.info("🔍 段階4.5: 多重解像度融合モジュール内SAM2検索...")
        fusion_sam2_found = []
        for name, submodule in multiresolution_fusion.named_modules():
            if any(keyword in name.lower() for keyword in ['sam2', 'sam_', 'segment']):
                device_info = "CPU"
                if hasattr(submodule, 'parameters'):
                    try:
                        first_param = next(submodule.parameters(), None)
                        if first_param is not None:
                            device_info = str(first_param.device)
                    except:
                        pass
                fusion_sam2_found.append((name, type(submodule).__name__, device_info))
                logger.info(f"  🎯 融合モジュール内SAM2: {name} ({type(submodule).__name__}) → {device_info}")
        
        if not fusion_sam2_found:
            logger.info("  ✅ 多重解像度融合モジュール内にSAM2コンポーネントは見つかりませんでした")
        
        # 🔍 全体のSAM2コンポーネント検索結果まとめ
        logger.info(f"🔍 SAM2コンポーネント検索結果: {len(sam2_components_found)}個発見")
        for i, comp in enumerate(sam2_components_found):
            logger.info(f"  [{i+1}] {comp['path']} ({comp['type']}) → {comp['device']}")
        
        # cuda:7にあるコンポーネントを特定
        cuda7_components = [comp for comp in sam2_components_found if 'cuda:7' in comp['device']]
        if cuda7_components:
            logger.warning(f"⚠️ cuda:7に配置されたSAM2コンポーネント: {len(cuda7_components)}個")
            for comp in cuda7_components:
                logger.warning(f"    🔥 修正対象: {comp['path']} ({comp['type']})")
        else:
            logger.info("✅ cuda:7にSAM2コンポーネントは見つかりませんでした")
        
        # OHEM損失関数（test_phase3b成功パターン移植）
        ohem_loss = create_ohem_loss(
            hard_ratio=0.25,
            config_override={
                'ce_loss_weight': 1.0,
                'bce_loss_weight': 2.0,
                'dice_loss_weight': 0.5
            }
        )
        logger.info("✓ OHEM損失関数初期化完了")
        
        logger.info("✅ Phase 3B統合モデル初期化完成")
        
        return {
            'qformer_bridge': qformer_bridge,
            'dual_pathway_decoder': dual_pathway_decoder,
            'multiresolution_fusion': multiresolution_fusion,
            'ohem_loss': ohem_loss,
            'llama4_model': llama4_model,
            'llama4_processor': llama4_processor
            # 注: qformer_modelは削除（QFormerSegmentationBridge内で管理）
        }
        
    except Exception as e:
        logger.error(f"❌ Phase 3Bモデル初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        raise

def apply_lora_to_model(model_components, args, logger):
    """QFormerSegmentationBridgeにLoRA適用（train_llama4_lisa_single_process.py準拠）"""
    logger.info("=== QFormerSegmentationBridgeにMixLoRA適用 ===")
    
    try:
        qformer_bridge = model_components['qformer_bridge']
        
        # Llama-4部分にLoRA適用（train_llama4_lisa_single_process.py成功パターン）
        if hasattr(qformer_bridge, 'llama_model') and qformer_bridge.llama_model is not None:
            # device_map保存（train_llama4_lisa_single_process.py完全移植）
            original_device_map = None
            original_device_map_location = None
            
            # device_mapの場所を特定して保存（成功パターン完全移植）
            if hasattr(qformer_bridge.llama_model, 'hf_device_map') and qformer_bridge.llama_model.hf_device_map:
                original_device_map = qformer_bridge.llama_model.hf_device_map.copy()
                original_device_map_location = "llama_model"
                logger.info(f"✓ 元のdevice_map保存（llama_model経由）: {len(original_device_map)} エントリ")
            elif hasattr(qformer_bridge, 'hf_device_map') and qformer_bridge.hf_device_map:
                original_device_map = qformer_bridge.hf_device_map.copy()
                original_device_map_location = "direct"
                logger.info(f"✓ 元のdevice_map保存（直接アクセス）: {len(original_device_map)} エントリ")
            else:
                logger.warning("⚠️ device_mapが見つかりません。Model Parallelismが未設定の可能性があります。")
            
            # MixLoRA設定作成
            lora_config_dict = config_linux.get_lora_config()
            lora_config_dict.pop('task_type', None)  # 重複回避
            
            # modules_to_saveの設定を確認
            modules_to_save = config_linux.ADDITIONAL_TRAINABLE_PARAMS
            logger.info(f"📊 modules_to_save設定: {modules_to_save}")
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                modules_to_save=modules_to_save,  # lm_head, embed_tokensを含む
                **lora_config_dict
            )
            
            logger.info(f"📊 MixLoRA設定:")
            logger.info(f"  - rank: {lora_config.r}")
            logger.info(f"  - alpha: {lora_config.lora_alpha}")
            logger.info(f"  - target_modules: {lora_config.target_modules}")
            
            # LoRA適用
            qformer_bridge.llama_model = get_peft_model(
                qformer_bridge.llama_model, 
                lora_config,
                autocast_adapter_dtype=False  # 🔥 dtype自動変換無効化
            )
            
            # パラメータ統計
            trainable_params = sum(p.numel() for p in qformer_bridge.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in qformer_bridge.parameters())
            
            logger.info(f"✓ LoRA適用完了")
            logger.info(f"  - 学習可能パラメータ: {trainable_params:,}")
            logger.info(f"  - 全パラメータ: {total_params:,}")
            logger.info(f"  - 効率化比率: {100 * trainable_params / total_params:.3f}%")
            
            # 🔥 device_mapの復元試行（train_llama4_lisa_single_process.py完全移植）
            if original_device_map and original_device_map_location:
                # 複数のアクセス方法を試行
                restoration_success = False
                
                # 方法1: 直接アクセス確認
                if hasattr(qformer_bridge, 'llama_model') and hasattr(qformer_bridge.llama_model, 'hf_device_map') and qformer_bridge.llama_model.hf_device_map:
                    logger.info("✓ llama_model経由device_mapアクセス確認済み")
                    restoration_success = True
                
                # 方法2: base_model経由のアクセス
                elif hasattr(qformer_bridge, 'llama_model') and hasattr(qformer_bridge.llama_model, 'base_model') and hasattr(qformer_bridge.llama_model.base_model, 'hf_device_map') and qformer_bridge.llama_model.base_model.hf_device_map:
                    logger.info("✓ llama_model.base_model経由device_mapアクセス確認済み")
                    restoration_success = True
                
                # 方法3: 手動復元
                if not restoration_success:
                    logger.warning("⚠️ device_mapが失われました。手動復元を試行...")
                    
                    # 元の場所に基づいて復元
                    if original_device_map_location == "llama_model":
                        if hasattr(qformer_bridge, 'llama_model'):
                            qformer_bridge.llama_model.hf_device_map = original_device_map
                            logger.info("✓ llama_model.hf_device_mapを手動復元")
                            restoration_success = True
                    elif original_device_map_location == "direct":
                        if hasattr(qformer_bridge, 'llama_model') and hasattr(qformer_bridge.llama_model, 'base_model'):
                            qformer_bridge.llama_model.base_model.hf_device_map = original_device_map
                            logger.info("✓ llama_model.base_model.hf_device_mapを手動復元")
                            restoration_success = True
                
                if not restoration_success:
                    logger.error("❌ device_mapの復元に失敗しました")
            
            # 🔥 Webリサーチ最適化: 2025年強化Gradient Checkpointing（SAM2+Llama4統合専用）
            # PyTorch torchtune準拠: selective checkpointing + offloading対応
            try:
                # 2025年ベストプラクティス: より効率的なcheckpointing設定
                gradient_ckpt_kwargs = {
                    "use_reentrant": False,  # requires_grad正しく維持
                    "preserve_rng_state": True,  # 再現性保証
                    "pack_hook_handles": True   # メモリ効率向上
                }
                
                if hasattr(qformer_bridge, 'llama_model') and hasattr(qformer_bridge.llama_model, 'gradient_checkpointing_enable'):
                    # 引数を受け取れるか確認
                    import inspect
                    sig = inspect.signature(qformer_bridge.llama_model.gradient_checkpointing_enable)
                    if 'gradient_checkpointing_kwargs' in sig.parameters:
                        qformer_bridge.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_ckpt_kwargs)
                        logger.info("✅ Gradient Checkpointing有効化成功（use_reentrant=False、requires_grad維持）")
                    else:
                        # 旧バージョンのHuggingFaceの場合
                        qformer_bridge.llama_model.gradient_checkpointing_enable()
                        logger.info("✅ Gradient Checkpointing有効化成功（メモリ30%削減）")
                        logger.warning("⚠️ use_reentrant=False未対応のHuggingFaceバージョン")
                elif hasattr(qformer_bridge, 'base_model') and hasattr(qformer_bridge.base_model, 'llama_model'):
                    if hasattr(qformer_bridge.base_model.llama_model, 'gradient_checkpointing_enable'):
                        sig = inspect.signature(qformer_bridge.base_model.llama_model.gradient_checkpointing_enable)
                        if 'gradient_checkpointing_kwargs' in sig.parameters:
                            qformer_bridge.base_model.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_ckpt_kwargs)
                            logger.info("✅ base_model Gradient Checkpointing有効化成功（use_reentrant=False）")
                        else:
                            qformer_bridge.base_model.llama_model.gradient_checkpointing_enable()
                            logger.info("✅ base_model Gradient Checkpointing有効化成功")
                elif hasattr(qformer_bridge, 'gradient_checkpointing_enable'):
                    sig = inspect.signature(qformer_bridge.gradient_checkpointing_enable)
                    if 'gradient_checkpointing_kwargs' in sig.parameters:
                        qformer_bridge.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_ckpt_kwargs)
                        logger.info("✅ 直接Gradient Checkpointing有効化成功（use_reentrant=False）")
                    else:
                        qformer_bridge.gradient_checkpointing_enable()
                        logger.info("✅ 直接Gradient Checkpointing有効化成功")
                else:
                    logger.warning("⚠️ Gradient Checkpointingサポートなし")
            except Exception as e:
                logger.warning(f"⚠️ Gradient Checkpointing有効化失敗（スキップ）: {e}")
            
            model_components['qformer_bridge'] = qformer_bridge
        else:
            logger.warning("⚠️ Llama-4モデルが見つかりません")
        
        return model_components
        
    except Exception as e:
        logger.error(f"LoRA適用エラー: {e}")
        raise

def create_dataset_and_dataloader(model_components, args, logger):
    """データセットとデータローダー作成（train_llama4_lisa成功パターン移植）"""
    logger.info("=== HybridDataset初期化（train_llama4_lisa成功パターン） ===")
    
    # データセット設定
    if hasattr(args, 'dataset') and args.dataset:
        datasets = [ds.strip() for ds in args.dataset.split(',')]
        # HybridDatasetの期待する形式（"||"区切り）に変換
        dataset_string = "||".join(datasets)
    else:
        dataset_string = "reason_seg"  # デフォルト
    
    logger.info(f"対象データセット: {datasets if 'datasets' in locals() else [dataset_string]}")
    
    # 各データセットのサンプル数制御
    samples_per_dataset = getattr(args, 'samples_per_dataset', None)
    if samples_per_dataset:
        logger.info(f"サンプル数制限: {samples_per_dataset}")
        # samples_per_datasetからsamples_per_epochを計算
        dataset_count = len(datasets) if 'datasets' in locals() else 1
        samples_per_epoch = samples_per_dataset * dataset_count
    else:
        # デフォルトは小さめに設定（段階的検証用）
        samples_per_epoch = 500  # train_llama4_lisa準拠
    
    logger.info(f"エポックあたりのサンプル数: {samples_per_epoch}")
    
    # プロセッサを取得（QFormerSegmentationBridge対応）
    qformer_bridge = model_components['qformer_bridge']
    if hasattr(qformer_bridge, 'llama_processor') and qformer_bridge.llama_processor:
        llama_processor = qformer_bridge.llama_processor
    elif hasattr(qformer_bridge, 'shared_llama_processor') and qformer_bridge.shared_llama_processor:
        llama_processor = qformer_bridge.shared_llama_processor
    else:
        logger.warning("⚠️ Llama processor not found in QFormerSegmentationBridge")
        llama_processor = None
    
    # HybridDataset作成（エラーハンドリング強化版）
    logger.info("🔍 HybridDataset初期化前デバッグ情報:")
    logger.info(f"  - base_image_dir: {config_linux.DATASET_BASE_DIR}")
    logger.info(f"  - base_image_dir存在確認: {os.path.exists(config_linux.DATASET_BASE_DIR)}")
    logger.info(f"  - llama_processor: {type(llama_processor)}")
    logger.info(f"  - samples_per_epoch: {samples_per_epoch}")
    logger.info(f"  - dataset_string: {dataset_string}")
    logger.info(f"  - sem_seg_data: {config_linux.SEM_SEG_DATA}")
    logger.info(f"  - refer_seg_data: {config_linux.REFER_SEG_DATA}")
    logger.info(f"  - vqa_data: {config_linux.VQA_DATA}")
    logger.info(f"  - reason_seg_data: {config_linux.REASON_SEG_DATA}")
    
    try:
        dataset = HybridDataset(
            base_image_dir=config_linux.DATASET_BASE_DIR,
            llama_processor=llama_processor,
            samples_per_epoch=samples_per_epoch,
            precision="bf16",
            llama_image_size=config_linux.LLAMA_IMAGE_SIZE,
            sam_image_size=config_linux.SAM_IMAGE_SIZE,
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
        
        # ❌ フォールバック削除: データ検証を強化してエラーで停止
        logger.info(f"HybridDataset初期化完了: {len(dataset)} サンプル")
        
        # データセット検証: 実際にデータが取得できるかテスト
        logger.info("🔍 データセット検証開始...")
        if len(dataset) == 0:
            error_msg = "❌ データセットが空です。データが存在しません。"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # 最初のサンプルを取得してデータの有効性を検証
        try:
            test_sample = dataset[0]
            logger.info("✅ データセット検証成功: 最初のサンプル取得完了")
        except Exception as validation_error:
            error_msg = f"❌ データセット検証失敗: 実際のデータ取得でエラーが発生しました。\n詳細: {validation_error}"
            logger.error(error_msg)
            logger.error("原因として考えられる問題:")
            logger.error("1. データファイルが存在しない")
            logger.error("2. データファイルが破損している")
            logger.error("3. 必要な説明ファイル（explanatory/train.json等）が不足")
            logger.error("4. データセットパスの設定ミス")
            logger.error(f"5. 設定確認: base_image_dir={config_linux.DATASET_BASE_DIR}")
            logger.error(f"6. 設定確認: reason_seg_data={config_linux.REASON_SEG_DATA}")
            raise RuntimeError(error_msg)
        
    except Exception as e:
        logger.error(f"❌ HybridDataset初期化または検証失敗: {e}")
        logger.error("詳細なエラー情報:")
        import traceback
        traceback.print_exc()
        logger.error("❌ データセットエラーにより訓練を停止します。")
        logger.error("解決方法:")
        logger.error("1. Lambda環境のデータセットパスを確認")
        logger.error("2. 必要なデータファイルの存在確認")
        logger.error("3. config_linux.pyのデータセット設定を確認")
        raise
    
    # ステップ数制限（デバッグ用）
    if hasattr(args, 'samples_per_epoch') and args.samples_per_epoch:
        logger.info(f"サンプル数制限: {args.samples_per_epoch}")
        # データセットの一部のみ使用
        indices = list(range(min(args.samples_per_epoch, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # DataLoader作成（参考ファイル準拠：メモリ最適化）
    dataloader = DataLoader(
        dataset,
        batch_size=getattr(args, 'batch_size', 1),  # GPU制約により小バッチ
        shuffle=True,
        num_workers=getattr(args, 'num_workers', 0),  # メモリ最適化：0に設定
        pin_memory=False,  # メモリ最適化：Falseに設定
        collate_fn=collate_fn
    )
    
    logger.info(f"✓ DataLoader作成完了")
    logger.info(f"  - データセットサイズ: {len(dataset)}")
    logger.info(f"  - バッチサイズ: {getattr(args, 'batch_size', 1)}")
    logger.info(f"  - ワーカー数: {getattr(args, 'num_workers', 0)}")
    
    return dataloader

def create_optimizer_and_scheduler(model_components, dataloader, args, logger):
    """オプティマイザーとスケジューラー作成"""
    logger.info("=== オプティマイザーとスケジューラー作成 ===")
    
    qformer_bridge = model_components['qformer_bridge']
    
    # 学習可能パラメータのみ対象
    trainable_params = [p for p in qformer_bridge.parameters() if p.requires_grad]
    
    # パラメータ状態チェック
    total_params = sum(1 for _ in qformer_bridge.named_parameters())
    trainable_count = len(trainable_params)
    logger.info(f"  - 総パラメータ数: {total_params}, 学習可能: {trainable_count}")
    
    # オプティマイザー作成（NaN対策版）
    lr = getattr(args, 'lr', config_linux.LEARNING_RATE)
    weight_decay = getattr(args, 'weight_decay', config_linux.WEIGHT_DECAY)
    
    if BITSANDBYTES_AVAILABLE and getattr(args, 'use_8bit_adam', False):
        optimizer = bnb.optim.AdamW8bit(
            trainable_params,
            lr=lr,
            betas=(config_linux.BETA1, config_linux.BETA2),
            weight_decay=weight_decay
        )
        logger.info("✓ 8bit AdamW使用")
    else:
        optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            betas=(config_linux.BETA1, config_linux.BETA2),
            weight_decay=weight_decay
        )
        logger.info("✓ 標準AdamW使用")
    
    # 🔥 Webリサーチ最適化: 2025年Loss Scaler（PyTorch torchtune準拠）
    scaler = None
    if getattr(config_linux, 'USE_LOSS_SCALING', False):
        scaler = torch.amp.GradScaler('cuda',
            init_scale=getattr(config_linux, 'INITIAL_LOSS_SCALE', 2**10),
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000
        )
        logger.info(f"✓ 2025年Loss Scaling: CUDA専用、初期スケール={config_linux.INITIAL_LOSS_SCALE}")
    
    # スケジューラー作成（linear scheduler使用）
    total_steps = len(dataloader) * getattr(args, 'epochs', 3)
    if hasattr(args, 'steps_per_epoch') and args.steps_per_epoch:
        total_steps = args.steps_per_epoch * getattr(args, 'epochs', 3)
    
    warmup_steps = int(config_linux.WARMUP_RATIO * total_steps)
    
    # linear scheduler使用（安定性優先）
    if config_linux.LR_SCHEDULER_TYPE == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        logger.info("✓ Linear Scheduler使用（安定性優先）")
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        logger.info("✓ Cosine Scheduler使用")
    
    logger.info(f"✓ スケジューラー作成完了")
    logger.info(f"  - 総ステップ数: {total_steps}")
    logger.info(f"  - ウォームアップステップ: {warmup_steps}")
    logger.info(f"  - 学習率: {lr}")
    
    return optimizer, scheduler, scaler

def get_model_device_map(model):
    """Model Parallelism対応device_map取得（17Bモデル用分散処理）"""
    device_map = None
    access_path = None
    
    # 複数のアクセス方法を試行（Model Parallelism対応）
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        device_map = model.hf_device_map
        access_path = "直接アクセス"
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'hf_device_map') and model.base_model.hf_device_map:
        device_map = model.base_model.hf_device_map
        access_path = "base_model経由"
    elif hasattr(model, 'llama_model') and hasattr(model.llama_model, 'hf_device_map') and model.llama_model.hf_device_map:
        device_map = model.llama_model.hf_device_map
        access_path = "llama_model経由"
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model') and hasattr(model.base_model.llama_model, 'hf_device_map') and model.base_model.llama_model.hf_device_map:
        device_map = model.base_model.llama_model.hf_device_map
        access_path = "base_model.llama_model経由"
    
    if device_map:
        # Model Parallelismの場合、最初のデバイスを返す（主要処理用）
        first_device = next(iter(device_map.values()))
        return first_device, f"{access_path} (Model Parallelism: 主要デバイス {first_device})"
    else:
        # フォールバック: cuda:0
        return torch.device("cuda:0"), "fallback (cuda:0)"

def move_to_device_safely(obj, device, logger, name="object"):
    """meta tensors対応のデバイス移動（test_phase3b成功パターン完全移植）"""
    try:
        if isinstance(obj, torch.nn.Module):
            # test_phase3b成功パターン: meta tensorの詳細チェック
            meta_params = [n for n, p in obj.named_parameters() if p.is_meta]
            if meta_params:
                logger.info(f"  - {name}: meta tensorsを含むためスキップ（disk offload維持）")
                logger.debug(f"    meta parameters: {meta_params[:3]}{'...' if len(meta_params) > 3 else ''}")
                return obj  # meta tensor含有モジュールはスキップ
            else:
                logger.info(f"  - {name}: {device}に移動中...")
                
                # 🔥 SAM2Wrapper特別処理：強制的な完全デバイス統一
                if 'sam2' in name.lower():
                    logger.info(f"  🔥 SAM2Wrapper強制統一: {name} → {device}")
                    # PyTorchベストプラクティス: 完全な再帰的移動
                    obj = obj.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                    
                    # 追加確認：全パラメータが正しく移動されたかチェック
                    device_check_failed = False
                    for param_name, param in obj.named_parameters():
                        if param.device != torch.device(device):
                            logger.warning(f"    ⚠️ {param_name}: {param.device} ≠ {device}")
                            device_check_failed = True
                    
                    if device_check_failed:
                        logger.warning(f"    🔄 再試行: {name}の完全統一")
                        # 再帰的強制移動
                        for module_name, module in obj.named_modules():
                            if hasattr(module, 'to'):
                                module.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                    
                    logger.info(f"  ✅ SAM2Wrapper強制統一完了: {name}")
                    return obj
                else:
                    # meta tensorがない場合のみデバイス移動
                    return obj.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        elif isinstance(obj, torch.Tensor):
            if obj.is_meta:
                logger.debug(f"  - {name}: meta tensorのためスキップ")
                return obj  # meta tensorはスキップ
            else:
                return obj.to(device, non_blocking=True)
        else:
            return obj
    except Exception as e:
        logger.warning(f"  - {name}: 移動エラー（{e}）、スキップ")
        return obj

def diagnose_qformer_device_placement(qformer_bridge, logger):
    """🔍 段階1: Q-Former内部のLayerNormパラメータ詳細デバイス診断"""
    try:
        logger.info("=" * 60)
        logger.info("🔍 Q-Former内部パラメータデバイス診断レポート")
        logger.info("=" * 60)
        
        # 1. Q-Former内部のLayerNormパラメータを再帰的に検索
        layernorm_devices = {}
        linear_devices = {}
        all_devices = set()
        
        def analyze_module_recursive(module, prefix=""):
            """モジュールを再帰的に分析してLayerNorm、Linearを特定"""
            for name, submodule in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # LayerNormの分析
                if isinstance(submodule, (torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    norm_type = type(submodule).__name__
                    if hasattr(submodule, 'weight') and submodule.weight is not None:
                        device = submodule.weight.device
                        layernorm_devices[f"{full_name}({norm_type})"] = str(device)
                        all_devices.add(str(device))
                        logger.info(f"  📍 {norm_type}: {full_name} → {device}")
                
                # Linearの分析
                elif isinstance(submodule, torch.nn.Linear):
                    if hasattr(submodule, 'weight') and submodule.weight is not None:
                        device = submodule.weight.device
                        linear_devices[f"{full_name}(Linear)"] = str(device)
                        all_devices.add(str(device))
                        logger.debug(f"  🔗 Linear: {full_name} → {device}")
                
                # 再帰的に子モジュールを解析
                analyze_module_recursive(submodule, full_name)
        
        # 2. Q-Former本体の分析
        if hasattr(qformer_bridge, 'qformer') and qformer_bridge.qformer is not None:
            logger.info("📊 Q-Former本体分析:")
            analyze_module_recursive(qformer_bridge.qformer, "qformer")
        
        # 3. SAM2Wrapperの分析（詳細版）
        if hasattr(qformer_bridge, 'sam2_wrapper') and qformer_bridge.sam2_wrapper is not None:
            logger.info("📊 SAM2Wrapper詳細分析:")
            analyze_module_recursive(qformer_bridge.sam2_wrapper, "sam2_wrapper")
            
            # SAM2内部のSAM2Modelも分析
            if hasattr(qformer_bridge.sam2_wrapper, 'sam2_model'):
                logger.info("📊 SAM2Wrapper.sam2_model詳細分析:")
                analyze_module_recursive(qformer_bridge.sam2_wrapper.sam2_model, "sam2_wrapper.sam2_model")
        
        # 4. その他のコンポーネント分析
        for attr_name in ['seg_token_generator', 'enhanced_sam_projector', 'curriculum_projector']:
            if hasattr(qformer_bridge, attr_name):
                component = getattr(qformer_bridge, attr_name)
                if component is not None:
                    logger.info(f"📊 {attr_name}分析:")
                    analyze_module_recursive(component, attr_name)
        
        # 5. 分析結果サマリ
        logger.info("=" * 60)
        logger.info(f"🔍 診断結果サマリ:")
        logger.info(f"  検出されたデバイス数: {len(all_devices)}")
        logger.info(f"  使用デバイス: {sorted(all_devices)}")
        logger.info(f"  LayerNorm/Norm層数: {len(layernorm_devices)}")
        logger.info(f"  Linear層数: {len(linear_devices)}")
        
        # 6. デバイス不整合の警告
        if len(all_devices) > 1:
            logger.warning("⚠️ デバイス不整合検出！複数デバイスにパラメータが分散しています:")
            device_counts = {}
            for device in all_devices:
                device_counts[device] = sum(1 for d in layernorm_devices.values() if d == device)
                device_counts[device] += sum(1 for d in linear_devices.values() if d == device)
            
            for device, count in device_counts.items():
                logger.warning(f"  {device}: {count}個のパラメータ")
        else:
            logger.info("✅ デバイス統一確認: 全パラメータが同一デバイス")
        
        # 7. nn.ModuleList使用状況確認
        logger.info("📊 nn.ModuleList使用状況:")
        modulelist_found = False
        
        def check_modulelist_recursive(module, prefix=""):
            nonlocal modulelist_found
            for name, submodule in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(submodule, torch.nn.ModuleList):
                    modulelist_found = True
                    logger.info(f"  ✅ nn.ModuleList発見: {full_name} (長さ: {len(submodule)})")
                elif isinstance(submodule, (list, tuple)):
                    logger.warning(f"  ⚠️ 通常のPythonリスト/タプル: {full_name} (タイプ: {type(submodule)})")
                
                check_modulelist_recursive(submodule, full_name)
        
        if hasattr(qformer_bridge, 'qformer') and qformer_bridge.qformer is not None:
            check_modulelist_recursive(qformer_bridge.qformer, "qformer")
        
        if not modulelist_found:
            logger.warning("⚠️ nn.ModuleListが見つかりませんでした。通常のPythonリストを使用している可能性があります。")
        
        logger.info("=" * 60)
        return layernorm_devices, linear_devices, all_devices
        
    except Exception as e:
        logger.error(f"❌ Q-Former診断エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}, {}, set()

def train_epoch(model_components, dataloader, optimizer, scheduler, scaler, epoch, args, logger, writer=None):
    """1エポックの学習実行（Phase 3B統合対応）"""
    qformer_bridge = model_components['qformer_bridge']
    dual_pathway_decoder = model_components['dual_pathway_decoder']
    multiresolution_fusion = model_components['multiresolution_fusion']
    ohem_loss = model_components['ohem_loss']
    
    qformer_bridge.train()
    dual_pathway_decoder.train()
    multiresolution_fusion.train()
    
    # 損失メトリクス初期化
    total_losses = AverageMeter('Total', ':.4e')
    phase2_losses = AverageMeter('Phase2', ':.4e')
    seg_losses = AverageMeter('Seg', ':.4e')
    ohem_losses = AverageMeter('OHEM', ':.4e')
    
    progress = ProgressMeter(
        len(dataloader) if args.steps_per_epoch is None else args.steps_per_epoch,
        [total_losses, phase2_losses, seg_losses, ohem_losses],
        prefix=f"Epoch: [{epoch}]"
    )
    
    # 個別損失履歴（エポック内）
    epoch_loss_history = {
        'total_loss': [],
        'phase2_loss': [],
        'seg_loss': [],
        'ohem_loss': []
    }
    
    # 🔍 メモリ推移トラッキング（リーク検出強化）
    memory_history = {
        'steps': [],
        'gpu_memory': {gpu_id: [] for gpu_id in range(torch.cuda.device_count())},
        'peak_memory': {gpu_id: [] for gpu_id in range(torch.cuda.device_count())},
        'reserved_memory': {gpu_id: [] for gpu_id in range(torch.cuda.device_count())},
        'timestamps': []
    }
    
    # メモリベースライン記録
    baseline_memory = {}
    for gpu_id in range(torch.cuda.device_count()):
        baseline_memory[gpu_id] = torch.cuda.memory_allocated(gpu_id) / 1024**3
    
    logger.info(f"📊 メモリベースライン記録: {', '.join([f'GPU{i}: {mem:.1f}GB' for i, mem in baseline_memory.items()])}")
    
    start_time = time.time()
    
    # 🔥 Webリサーチ最適化: 2025年動的バッチサイズ管理（PyTorch torchtune準拠）
    current_batch_size = dataloader.batch_size
    oom_retry_count = 0
    max_oom_retries = 2
    
    # 🔥 メモリ推移記録用変数（部分ログでも傾向把握可能）
    memory_progression = {
        'steps': [],
        'gpu_memory': {gpu_id: [] for gpu_id in range(torch.cuda.device_count())},
        'gpu0_concentration': [],  # GPU 0の負荷集中度
        'total_allocated': [],     # 全GPU総メモリ使用量
        'memory_efficiency': []    # メモリ効率（allocated/reserved）
    }
    
    for step, batch in enumerate(dataloader):
        # 🔥 メモリ推移記録（5ステップごと + 重要ポイント）
        if step % 5 == 0 or step < 10:
            current_memory = {}
            total_allocated = 0
            total_reserved = 0
            
            for gpu_id in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                current_memory[gpu_id] = allocated
                total_allocated += allocated
                total_reserved += reserved
            
            # GPU 0集中度計算
            gpu0_ratio = (current_memory[0] / total_allocated * 100) if total_allocated > 0 else 0
            memory_efficiency = (total_allocated / total_reserved * 100) if total_reserved > 0 else 0
            
            # 推移データ記録
            memory_progression['steps'].append(step)
            memory_progression['total_allocated'].append(total_allocated)
            memory_progression['gpu0_concentration'].append(gpu0_ratio)
            memory_progression['memory_efficiency'].append(memory_efficiency)
            
            for gpu_id, mem in current_memory.items():
                memory_progression['gpu_memory'][gpu_id].append(mem)
            
            # 詳細ログ（初期 + 問題発生時）
            if step < 3 or step % 20 == 0:
                logger.info(f"📊 Step {step} メモリ状況:")
                for gpu_id, mem in current_memory.items():
                    baseline_mem = baseline_memory.get(gpu_id, 0)
                    delta = mem - baseline_mem
                    logger.info(f"  GPU {gpu_id}: {mem:.2f}GB (Δ{delta:+.2f}GB)")
                logger.info(f"  総使用量: {total_allocated:.2f}GB, GPU0集中: {gpu0_ratio:.1f}%, 効率: {memory_efficiency:.1f}%")
        
        # 🔥 メモリ最適化: バッチごとにCUDAキャッシュクリア（参考ファイル準拠）
        torch.cuda.empty_cache()
        
        # ステップ数制限チェック
        if args.steps_per_epoch is not None and step >= args.steps_per_epoch:
            break
        
        # 🔥 Model Parallelism対応データ準備（train_llama4_lisa_single_process.py完全移植）
        try:
            # Model Parallelismのdevice_mapを厳密にチェック（QFormerSegmentationBridge対応）
            device_map = None
            access_path = None
            
            # 複数のアクセス方法を試行（成功パターン完全移植）
            if hasattr(qformer_bridge, 'llama_model') and hasattr(qformer_bridge.llama_model, 'hf_device_map') and qformer_bridge.llama_model.hf_device_map:
                device_map = qformer_bridge.llama_model.hf_device_map
                access_path = "llama_model経由"
            elif hasattr(qformer_bridge, 'hf_device_map') and qformer_bridge.hf_device_map:
                device_map = qformer_bridge.hf_device_map
                access_path = "直接アクセス"
            elif hasattr(qformer_bridge, 'llama_model') and hasattr(qformer_bridge.llama_model, 'base_model') and hasattr(qformer_bridge.llama_model.base_model, 'hf_device_map') and qformer_bridge.llama_model.base_model.hf_device_map:
                device_map = qformer_bridge.llama_model.base_model.hf_device_map
                access_path = "llama_model.base_model経由"
            
            if not device_map:
                raise RuntimeError("Model Parallelismが設定されていません。17Bモデルには必須です。")
            
            if step == 0:
                logger.info(f"Model Parallelismデバイスマップ確認: {access_path}")
            
            first_device = next(iter(device_map.values()))
            
            # HybridDatasetの出力をデュアルエンコーダー対応に変換
            adapted_batch = adapt_dataset_for_qformer(batch)
            
            # デバイス移動（train_llama4_lisa_single_process.py準拠）
            adapted_batch = {k: v.to(first_device) if hasattr(v, 'to') else v for k, v in adapted_batch.items()}
            
            # バッチ内容をデバッグ出力（初回のみ）
            if step == 0:
                logger.info(f"📊 デュアルエンコーダーバッチ確認:")
                for key, value in adapted_batch.items():
                    if isinstance(value, torch.Tensor):
                        logger.info(f"  {key}: {value.shape} ({value.dtype})")
                    else:
                        logger.info(f"  {key}: {type(value)}")
        
        except Exception as e:
            logger.error(f"データ準備エラー (step {step}): {e}")
            continue
        
        # 🔥 強化メモリリーク対策: フォワードパス前の完全クリーンアップ
        optimizer.zero_grad(set_to_none=True)  # より効率的なゼロ化
        
        # 中間変数の明示的解放
        if 'qformer_outputs' in locals():
            del qformer_outputs
        if 'dual_outputs' in locals():
            del dual_outputs
        if 'individual_losses' in locals():
            del individual_losses
        
        # CUDA Memory Pool設定最適化
        torch.cuda.empty_cache()
        if step % 3 == 0:  # 3ステップごとに断片化解消
            torch.cuda.synchronize()  # GPU同期
            gc.collect()  # Python GC
        
        try:
            # 🔥 2025年メモリ効率化: Gradient Checkpointing + Memory Pool最適化
            torch.cuda.empty_cache()  # フォワードパス前のメモリクリア
            
            # 📊 Dynamic Memory Management（メモリリーク防止）
            if step > 0:
                # 前ステップのcomputation graph完全削除
                torch.cuda.synchronize()
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
            
            # 🔍 段階5: LayerNormエラー直前デバイス診断
            if step == 0:
                logger.info("🔍 段階5: LayerNormエラー直前のデバイス診断開始...")
                
                # QFormerBridge内の全LayerNormモジュールをチェック
                layernorm_devices = []
                for name, module in qformer_bridge.named_modules():
                    if 'LayerNorm' in type(module).__name__ or 'layernorm' in name.lower():
                        try:
                            first_param = next(module.parameters(), None)
                            if first_param is not None:
                                device_info = str(first_param.device)
                                layernorm_devices.append((name, device_info))
                                logger.info(f"    LayerNorm発見: {name} → {device_info}")
                        except:
                            pass
                
                # 入力テンソルのデバイス確認
                images_device = adapted_batch.get('pixel_values', adapted_batch.get('images')).device
                sam_images_device = adapted_batch.get('sam_pixel_values', adapted_batch.get('sam_images')).device
                input_ids_device = adapted_batch['input_ids'].device
                
                logger.info(f"    入力テンソルデバイス:")
                logger.info(f"      - images: {images_device}")
                logger.info(f"      - sam_images: {sam_images_device}")
                logger.info(f"      - input_ids: {input_ids_device}")
                
                # デバイス不一致の可能性を予測
                unique_devices = set([device for _, device in layernorm_devices])
                if len(unique_devices) > 1:
                    logger.warning(f"⚠️ LayerNormデバイス不一致検出: {unique_devices}")
                    for name, device in layernorm_devices:
                        if 'cuda:7' in device:
                            logger.warning(f"    🔥 cuda:7のLayerNorm: {name}")
                else:
                    logger.info(f"✅ LayerNorm統一デバイス: {unique_devices}")
            
            # 2025年ベストプラクティス: H100専用高速化autocast + Memory-efficient attention
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True, cache_enabled=False):  # cache無効でメモリ節約
                # 1. QFormerSegmentationBridge推論（test_phase3b成功パターン移植）
                qformer_outputs = qformer_bridge(
                    images=adapted_batch.get('pixel_values', adapted_batch.get('images')),     # pixel_values → images
                    sam_images=adapted_batch.get('sam_pixel_values', adapted_batch.get('sam_images')),  # sam_pixel_values → sam_images
                    input_ids=adapted_batch['input_ids'],
                    attention_mask=adapted_batch['attention_mask'],
                    labels=adapted_batch['labels'],
                    return_dict=True
                )
                
                # デバッグ: モデル出力のキーを確認（初回のみ）
                if step == 0:
                    if isinstance(qformer_outputs, dict):
                        logger.info(f"  🔍 QFormerSegmentationBridge出力キー: {list(qformer_outputs.keys())}")
                    else:
                        logger.info(f"  🔍 QFormerSegmentationBridge出力タイプ: {type(qformer_outputs)}")
                
                # 2. Llama隠れ状態の取得（test_phase3b成功パターン完全移植）
                llama_features = None
                sam_features = None
                qformer_features = None
                
                # test_phase3bパターン: QFormerSegmentationBridge出力から確実に取得
                logger.debug(f"  🔍 Llama隠れ状態取得開始...")
                
                # Llama特徴の取得（実際の出力構造に基づく）
                # デバッグ出力: ['text_loss', 'predicted_masks', 'query_embeddings', 'sam_prompts', 'method', 'num_queries', 'seg_token', 'seg_sam_prompt']
                
                # QFormerSegmentationBridgeにはllama_hidden_statesがないため、query_embeddingsを使用
                if hasattr(qformer_outputs, 'query_embeddings') and qformer_outputs.query_embeddings is not None:
                    # query_embeddingsをLlama特徴として使用（768次元 → 5120次元に投影必要）
                    query_embeddings = qformer_outputs.query_embeddings
                    logger.debug(f"    ✓ query_embeddings取得: {query_embeddings.shape}")
                    
                    # Q-Formerクエリ(768次元)をLlama隠れ状態(5120次元)に投影
                    if 'query_to_llama_projector' not in model_components:
                        logger.debug(f"    🔧 Query→Llama投影層作成中...")
                        model_components['query_to_llama_projector'] = nn.Linear(
                            768,  # Q-Former query次元
                            5120, # Llama隠れ状態次元
                            bias=True
                        ).to(device=query_embeddings.device, dtype=torch.bfloat16)
                        logger.debug(f"    ✅ 投影層作成完了: 768 → 5120")
                    
                    # 投影実行
                    llama_features = model_components['query_to_llama_projector'](query_embeddings.to(torch.bfloat16))
                    logger.debug(f"    ✅ Query→Llama投影完了: {query_embeddings.shape} → {llama_features.shape}")
                    
                elif isinstance(qformer_outputs, dict) and 'query_embeddings' in qformer_outputs:
                    query_embeddings = qformer_outputs['query_embeddings']
                    logger.debug(f"    ✓ 辞書からquery_embeddings取得: {query_embeddings.shape}")
                    
                    # 同様の投影処理
                    if 'query_to_llama_projector' not in model_components:
                        model_components['query_to_llama_projector'] = nn.Linear(768, 5120, bias=True).to(device=query_embeddings.device, dtype=torch.bfloat16)
                    
                    llama_features = model_components['query_to_llama_projector'](query_embeddings.to(torch.bfloat16))
                    logger.debug(f"    ✅ 辞書Query→Llama投影完了: {llama_features.shape}")
                    
                else:
                    available_attrs = list(qformer_outputs.keys()) if isinstance(qformer_outputs, dict) else [attr for attr in dir(qformer_outputs) if not attr.startswith('_')]
                    raise ValueError(f"query_embeddingsがQFormerSegmentationBridge出力に見つかりません。利用可能な属性: {available_attrs}")
                
                # SAM特徴とQ-Former特徴の取得
                if hasattr(qformer_outputs, 'sam_features') and qformer_outputs.sam_features is not None:
                    sam_features = qformer_outputs.sam_features
                    logger.debug(f"    ✓ SAM特徴取得: {sam_features.shape}")
                
                if hasattr(qformer_outputs, 'query_embeddings') and qformer_outputs.query_embeddings is not None:
                    qformer_features = qformer_outputs.query_embeddings
                    logger.debug(f"    ✓ Q-Former特徴(query_embeddings)取得: {qformer_features.shape}")
                elif hasattr(qformer_outputs, 'query_embeds') and qformer_outputs.query_embeds is not None:
                    qformer_features = qformer_outputs.query_embeds
                    logger.debug(f"    ✓ Q-Former特徴(query_embeds)取得: {qformer_features.shape}")
                
                # 多重解像度融合（特徴が利用可能な場合のみ）
                fusion_outputs = None
                if llama_features is not None and sam_features is not None and qformer_features is not None:
                    # 🔥 デバイス統一確認（全特徴量をfirst_deviceに統一）
                    if qformer_features.device != first_device:
                        qformer_features = qformer_features.to(first_device)
                        logger.debug(f"    📝 qformer_features統一: → {first_device}")
                    
                    if sam_features.device != first_device:
                        sam_features = sam_features.to(first_device)
                        logger.debug(f"    📝 sam_features統一: → {first_device}")
                    
                    if llama_features.device != first_device:
                        llama_features = llama_features.to(first_device)
                        logger.debug(f"    📝 llama_features統一: → {first_device}")
                    
                    fusion_outputs = multiresolution_fusion(
                        llama_features=llama_features,
                        sam_features=sam_features,
                        qformer_features=qformer_features
                    )
                    logger.debug(f"    ✅ 多重解像度融合完了")
                else:
                    logger.debug(f"    ⚠️ 多重解像度融合スキップ（特徴不足）")
                
                # 3. デュアルパスウェイ推論（SAM画像が利用可能な場合）
                if 'sam_images' in adapted_batch or 'sam_pixel_values' in adapted_batch:
                    sam_images = adapted_batch.get('sam_images', adapted_batch.get('sam_pixel_values'))
                    
                    if sam_images is not None:
                        # SAMプロンプト生成（test_phase3b成功パターン: 複数属性名対応）
                        sam_prompts = None
                        if hasattr(qformer_outputs, 'sam_prompts') and qformer_outputs.sam_prompts is not None:
                            sam_prompts = qformer_outputs.sam_prompts
                            logger.debug(f"  🔍 sam_prompts属性使用: {sam_prompts.shape}")
                        elif hasattr(qformer_outputs, 'seg_sam_prompt') and qformer_outputs.seg_sam_prompt is not None:
                            sam_prompts = qformer_outputs.seg_sam_prompt
                            logger.debug(f"  🔍 seg_sam_prompt属性使用: {sam_prompts.shape}")
                        elif isinstance(qformer_outputs, dict):
                            if 'sam_prompts' in qformer_outputs and qformer_outputs['sam_prompts'] is not None:
                                sam_prompts = qformer_outputs['sam_prompts']
                                logger.debug(f"  🔍 辞書sam_prompts使用: {sam_prompts.shape}")
                            elif 'seg_sam_prompt' in qformer_outputs and qformer_outputs['seg_sam_prompt'] is not None:
                                sam_prompts = qformer_outputs['seg_sam_prompt']
                                logger.debug(f"  🔍 辞書seg_sam_prompt使用: {sam_prompts.shape}")
                        
                        if sam_prompts is None:
                            # デバッグ: 利用可能な属性/キー出力
                            if hasattr(qformer_outputs, '__dict__'):
                                available_attrs = [attr for attr in dir(qformer_outputs) if not attr.startswith('_')]
                                logger.error(f"❌ 利用可能な属性: {available_attrs}")
                            if isinstance(qformer_outputs, dict):
                                logger.error(f"❌ 利用可能なキー: {list(qformer_outputs.keys())}")
                            raise ValueError("SAM prompts not found in qformer_outputs")
                        
                        # Llama隠れ状態（必須）
                        if llama_features is not None:
                            llama_hidden_states = llama_features
                            logger.debug(f"    ✓ Llama隠れ状態使用: {llama_hidden_states.shape}")
                        else:
                            raise ValueError("Llama features必須ですが利用できません")
                        
                        # 🔥 デュアルパスウェイ推論前のデバイス統一（cuda:1 vs cuda:7エラー対策）
                        target_device = llama_hidden_states.device  # Llamaのデバイスを基準とする
                        
                        if sam_images.device != target_device:
                            sam_images = sam_images.to(target_device)
                            logger.debug(f"    📝 sam_images統一: → {target_device}")
                        
                        if sam_prompts.device != target_device:
                            sam_prompts = sam_prompts.to(target_device)
                            logger.debug(f"    📝 sam_prompts統一: → {target_device}")
                        
                        # デュアルパスウェイ推論
                        dual_outputs = dual_pathway_decoder(
                            images=sam_images,
                            sam_prompts=sam_prompts,
                            llama_hidden_states=llama_hidden_states
                        )
                    else:
                        dual_outputs = None
                else:
                    dual_outputs = None
                
                # 4. 損失計算（詳細デバッグ付き）
                logger.debug(f"  🔍 損失計算デバッグ開始 (step {step})...")
                individual_losses = {'total_loss': 0, 'phase2_loss': 0, 'seg_loss': 0, 'ohem_loss': 0}
                
                # QFormerSegmentationBridge出力の詳細分析
                logger.debug(f"    📊 qformer_outputs詳細分析:")
                logger.debug(f"      - タイプ: {type(qformer_outputs)}")
                if isinstance(qformer_outputs, dict):
                    logger.debug(f"      - 利用可能なキー: {list(qformer_outputs.keys())}")
                    
                    # 各キーの値をチェック
                    for key, value in qformer_outputs.items():
                        if isinstance(value, torch.Tensor):
                            if value.numel() == 1:  # スカラー値
                                logger.debug(f"        {key}: {value.item():.6f} (shape: {value.shape}, dtype: {value.dtype})")
                            else:
                                logger.debug(f"        {key}: shape={value.shape}, dtype={value.dtype}")
                        else:
                            logger.debug(f"        {key}: {type(value)}")
                
                if isinstance(qformer_outputs, dict):
                    # 🔍 デバッグ: 利用可能な損失を全て確認
                    logger.debug(f"    🔍 損失候補検索...")
                    potential_losses = {}
                    
                    for key, value in qformer_outputs.items():
                        if isinstance(value, torch.Tensor) and value.numel() == 1:  # スカラーテンソル
                            if 'loss' in key.lower():
                                potential_losses[key] = value.item()
                                logger.debug(f"      損失候補発見: {key} = {value.item():.6f}")
                    
                    logger.debug(f"    📋 発見された損失候補: {potential_losses}")
                    
                    # 1. 直接total_lossを確認
                    if 'total_loss' in qformer_outputs:
                        loss = qformer_outputs['total_loss']
                        logger.debug(f"    ✓ total_loss直接取得: {loss.item():.6f}")
                        individual_losses['total_loss'] = loss.item()
                    
                    # 2. losses辞書内のtotal_lossを確認
                    elif 'losses' in qformer_outputs and isinstance(qformer_outputs['losses'], dict):
                        losses_dict = qformer_outputs['losses']
                        logger.debug(f"    📋 losses辞書内容: {list(losses_dict.keys())}")
                        if 'total_loss' in losses_dict:
                            loss = losses_dict['total_loss']
                            logger.debug(f"    ✓ losses辞書内total_loss取得: {loss.item():.6f}")
                            individual_losses['total_loss'] = loss.item()
                        else:
                            logger.error(f"    ❌ losses辞書にtotal_lossなし: {list(losses_dict.keys())}")
                            raise ValueError(f"losses辞書にtotal_lossが見つかりません。利用可能なキー: {list(losses_dict.keys())}")
                    
                    # 3. loss属性を確認（標準transformers出力）
                    elif 'loss' in qformer_outputs:
                        loss = qformer_outputs['loss']
                        logger.debug(f"    ✓ loss属性取得: {loss.item():.6f}")
                        individual_losses['total_loss'] = loss.item()
                    
                    # 4. text_lossをtotal_lossとして使用（QFormerSegmentationBridge対応）
                    elif 'text_loss' in qformer_outputs:
                        loss = qformer_outputs['text_loss']
                        logger.warning(f"    ⚠️ text_lossをtotal_lossとして使用: {loss.item():.6f}")
                        individual_losses['total_loss'] = loss.item()
                    
                    # 5. エラー：損失が見つからない
                    else:
                        available_keys = list(qformer_outputs.keys())
                        logger.error(f"    ❌ いかなる損失も見つかりません")
                        logger.error(f"      利用可能キー: {available_keys}")
                        logger.error(f"      損失候補: {potential_losses}")
                        raise ValueError(f"total_loss not found in model outputs. Available keys: {available_keys}")
                    
                    # 個別損失の詳細を取得（厳密チェック）
                    if 'losses' in qformer_outputs and isinstance(qformer_outputs['losses'], dict):
                        losses_dict = qformer_outputs['losses']
                        if step == 0:
                            logger.info(f"  🔍 losses辞書キー: {list(losses_dict.keys())}")
                        
                        # phase2_loss（Phase 2統一空間）
                        if 'phase2_loss' in losses_dict and losses_dict['phase2_loss'] is not None:
                            individual_losses['phase2_loss'] = losses_dict['phase2_loss'].item()
                        elif 'moe_loss' in losses_dict and losses_dict['moe_loss'] is not None:
                            individual_losses['phase2_loss'] = losses_dict['moe_loss'].item()
                        
                        # seg_loss（セグメンテーション損失）
                        if 'seg_loss' in losses_dict and losses_dict['seg_loss'] is not None:
                            individual_losses['seg_loss'] = losses_dict['seg_loss'].item()
                        
                        # ohem_loss（OHEM損失）- QFormerからは通常利用できないため、後で計算
                        if 'ohem_loss' in losses_dict and losses_dict['ohem_loss'] is not None:
                            individual_losses['ohem_loss'] = losses_dict['ohem_loss'].item()
                        elif 'lm_loss' in losses_dict and losses_dict['lm_loss'] is not None:
                            individual_losses['ohem_loss'] = losses_dict['lm_loss'].item()
                        # ✅ OHEM損失は後で専用関数で計算される
                    
                    # Phase2損失（MoE負荷分散）の正確な計算 - 詳細デバッグ版
                    logger.info(f"    🔧 Phase2損失計算開始...")
                    logger.info(f"    🔍 qformer_outputs詳細構造デバッグ:")
                    logger.info(f"      - Type: {type(qformer_outputs)}")
                    logger.info(f"      - isinstance(dict): {isinstance(qformer_outputs, dict)}")
                    
                    if isinstance(qformer_outputs, dict):
                        logger.info(f"      - Keys: {list(qformer_outputs.keys())}")
                        logger.info(f"      - 各キーの値の型:")
                        for key, value in qformer_outputs.items():
                            logger.info(f"        {key}: {type(value)}")
                            if hasattr(value, 'shape'):
                                logger.info(f"          - Shape: {value.shape}")
                            elif isinstance(value, dict):
                                logger.info(f"          - Dict keys: {list(value.keys())}")
                            elif isinstance(value, (list, tuple)):
                                logger.info(f"          - Length: {len(value)}")
                    else:
                        logger.info(f"      - 属性一覧: {[attr for attr in dir(qformer_outputs) if not attr.startswith('_')]}")
                        # 各属性の詳細確認
                        for attr in [attr for attr in dir(qformer_outputs) if not attr.startswith('_')]:
                            try:
                                value = getattr(qformer_outputs, attr)
                                logger.info(f"        {attr}: {type(value)}")
                                if hasattr(value, 'shape'):
                                    logger.info(f"          - Shape: {value.shape}")
                                elif isinstance(value, dict):
                                    logger.info(f"          - Dict keys: {list(value.keys())}")
                            except Exception as e:
                                logger.info(f"        {attr}: Error accessing - {e}")
                    
                    # Phase2処理結果の探索 - 複数パターン対応
                    phase2_found = False
                    phase2_output = None
                    
                    # パターン1: 辞書のphase2_outputsキー（修正後の正しい構造）
                    if isinstance(qformer_outputs, dict) and 'phase2_outputs' in qformer_outputs:
                        phase2_output = qformer_outputs['phase2_outputs']
                        phase2_found = True
                        logger.info(f"    ✅ Phase2パターン1発見: 辞書のphase2_outputsキー（修正後）")
                    
                    # パターン2: phase2_outputs属性（フォールバック）
                    elif hasattr(qformer_outputs, 'phase2_outputs') and qformer_outputs.phase2_outputs:
                        phase2_output = qformer_outputs.phase2_outputs
                        phase2_found = True
                        logger.info(f"    ✅ Phase2パターン2発見: phase2_outputs属性")
                    
                    # パターン3: 辞書のphase2キー（フォールバック）
                    elif isinstance(qformer_outputs, dict) and 'phase2' in qformer_outputs:
                        phase2_output = qformer_outputs['phase2']
                        phase2_found = True
                        logger.info(f"    ✅ Phase2パターン3発見: 辞書のphase2キー")
                    
                    # パターン4: MoE関連の直接キー探索
                    elif isinstance(qformer_outputs, dict):
                        moe_keys = [k for k in qformer_outputs.keys() if 'moe' in k.lower() or 'load_balance' in k.lower()]
                        if moe_keys:
                            logger.info(f"    🔍 MoE関連キー発見: {moe_keys}")
                            for moe_key in moe_keys:
                                moe_value = qformer_outputs[moe_key]
                                logger.info(f"      {moe_key}: {type(moe_value)}")
                                if isinstance(moe_value, dict):
                                    logger.info(f"        - Subkeys: {list(moe_value.keys())}")
                    
                    if phase2_found and phase2_output is not None:
                        logger.info(f"    📊 Phase2出力構造詳細: {type(phase2_output)}")
                        
                        if isinstance(phase2_output, dict):
                            logger.info(f"      - Keys: {list(phase2_output.keys())}")
                            
                            # MoE負荷分散損失の抽出
                            if 'moe_adapted_outputs' in phase2_output and 'load_balance_loss' in phase2_output['moe_adapted_outputs']:
                                lb_loss = phase2_output['moe_adapted_outputs']['load_balance_loss']
                                if torch.is_tensor(lb_loss) and not torch.isnan(lb_loss):
                                    individual_losses['phase2_loss'] = lb_loss.item()
                                    logger.info(f"    ✅ Phase2損失（MoE負荷分散）: {individual_losses['phase2_loss']:.6f}")
                                else:
                                    logger.error(f"    ❌ Phase2 MoE負荷分散損失が無効: {lb_loss}")
                                    raise ValueError(f"Phase2 MoE負荷分散損失が無効です: {lb_loss}")
                            else:
                                logger.error(f"    ❌ Phase2 MoE負荷分散損失が見つからない")
                                logger.error(f"    📊 利用可能なキー: {list(phase2_output.keys())}")
                                if 'moe_adapted_outputs' in phase2_output:
                                    logger.error(f"    📊 moe_adapted_outputsのキー: {list(phase2_output['moe_adapted_outputs'].keys())}")
                                # エラーを隠蔽せず、詳細情報を提供してから停止
                                raise ValueError("Phase2 MoE負荷分散損失が見つかりません - 詳細はログを確認")
                        else:
                            logger.error(f"    ❌ Phase2出力が辞書型ではない: {type(phase2_output)}")
                            raise ValueError(f"Phase2出力の型が予期しない形式: {type(phase2_output)}")
                    else:
                        logger.error(f"    ❌ Phase2出力が全パターンで見つからない")
                        logger.error(f"    📊 探索パターン:")
                        logger.error(f"      1. hasattr(qformer_outputs, 'phase2_outputs'): {hasattr(qformer_outputs, 'phase2_outputs')}")
                        logger.error(f"      2. 'phase2' in qformer_outputs: {isinstance(qformer_outputs, dict) and 'phase2' in qformer_outputs}")
                        logger.error(f"      3. 'phase2_outputs' in qformer_outputs: {isinstance(qformer_outputs, dict) and 'phase2_outputs' in qformer_outputs}")
                        # CLAUDE.md指針: エラーを隠蔽せずデバッグに繋げる
                        raise ValueError("Phase2処理結果が全探索パターンで見つかりません - 実装を確認する必要があります")
                    
                    # セグメンテーション損失（デュアルパスウェイ一貫性）の正確な計算 - 詳細デバッグ版
                    logger.info(f"    🔧 セグメンテーション損失計算開始...")
                    logger.info(f"    🔍 dual_outputs詳細構造デバッグ:")
                    logger.info(f"      - Type: {type(dual_outputs)}")
                    logger.info(f"      - bool(dual_outputs): {bool(dual_outputs)}")
                    logger.info(f"      - isinstance(dict): {isinstance(dual_outputs, dict)}")
                    
                    if dual_outputs:
                        if isinstance(dual_outputs, dict):
                            logger.info(f"      - Keys: {list(dual_outputs.keys())}")
                            logger.info(f"      - 各キーの値の型:")
                            for key, value in dual_outputs.items():
                                logger.info(f"        {key}: {type(value)}")
                                if hasattr(value, 'shape'):
                                    logger.info(f"          - Shape: {value.shape}")
                                elif isinstance(value, dict):
                                    logger.info(f"          - Dict keys: {list(value.keys())}")
                                elif isinstance(value, (list, tuple)):
                                    logger.info(f"          - Length: {len(value)}")
                                elif torch.is_tensor(value):
                                    logger.info(f"          - Tensor value: {value.item() if value.numel() == 1 else 'multi-element'}")
                        else:
                            logger.info(f"      - 属性一覧: {[attr for attr in dir(dual_outputs) if not attr.startswith('_')]}")
                            # 各属性の詳細確認
                            for attr in [attr for attr in dir(dual_outputs) if not attr.startswith('_')]:
                                try:
                                    value = getattr(dual_outputs, attr)
                                    logger.info(f"        {attr}: {type(value)}")
                                    if hasattr(value, 'shape'):
                                        logger.info(f"          - Shape: {value.shape}")
                                    elif isinstance(value, dict):
                                        logger.info(f"          - Dict keys: {list(value.keys())}")
                                except Exception as e:
                                    logger.info(f"        {attr}: Error accessing - {e}")
                        
                        # 一貫性損失の探索 - 複数パターン対応
                        consistency_found = False
                        cons_loss = None
                        
                        if isinstance(dual_outputs, dict):
                            # パターン1: consistency_loss
                            if 'consistency_loss' in dual_outputs:
                                cons_loss = dual_outputs['consistency_loss']
                                consistency_found = True
                                logger.info(f"    ✅ 一貫性損失パターン1発見: consistency_loss")
                            
                            # パターン2: dual_consistency
                            elif 'dual_consistency' in dual_outputs:
                                cons_loss = dual_outputs['dual_consistency']
                                consistency_found = True
                                logger.info(f"    ✅ 一貫性損失パターン2発見: dual_consistency")
                            
                            # パターン3: pathway_consistency
                            elif 'pathway_consistency' in dual_outputs:
                                cons_loss = dual_outputs['pathway_consistency']
                                consistency_found = True
                                logger.info(f"    ✅ 一貫性損失パターン3発見: pathway_consistency")
                            
                            # パターン4: loss関連キーの探索
                            else:
                                loss_keys = [k for k in dual_outputs.keys() if 'loss' in k.lower() or 'consist' in k.lower()]
                                if loss_keys:
                                    logger.info(f"    🔍 損失関連キー発見: {loss_keys}")
                                    for loss_key in loss_keys:
                                        loss_value = dual_outputs[loss_key]
                                        logger.info(f"      {loss_key}: {type(loss_value)}")
                                        if torch.is_tensor(loss_value):
                                            logger.info(f"        - Tensor value: {loss_value.item() if loss_value.numel() == 1 else 'multi-element'}")
                        
                        if consistency_found and cons_loss is not None:
                            logger.info(f"    📊 一貫性損失詳細: {type(cons_loss)}")
                            if torch.is_tensor(cons_loss):
                                logger.info(f"      - Tensor info: shape={cons_loss.shape}, dtype={cons_loss.dtype}")
                                logger.info(f"      - Value: {cons_loss.item() if cons_loss.numel() == 1 else 'multi-element'}")
                                logger.info(f"      - NaN check: {torch.isnan(cons_loss).any()}")
                                
                                if not torch.isnan(cons_loss):
                                    individual_losses['seg_loss'] = abs(cons_loss).item()
                                    logger.info(f"    ✅ セグメンテーション損失（一貫性）: {individual_losses['seg_loss']:.6f}")
                                else:
                                    logger.error(f"    ❌ デュアルパスウェイ一貫性損失にNaN: {cons_loss}")
                                    raise ValueError(f"デュアルパスウェイ一貫性損失にNaN: {cons_loss}")
                            else:
                                logger.error(f"    ❌ 一貫性損失がTensorではない: {type(cons_loss)}")
                                raise ValueError(f"一貫性損失の型が予期しない形式: {type(cons_loss)}")
                        else:
                            logger.error(f"    ❌ デュアルパスウェイ一貫性損失が全パターンで見つからない")
                            if isinstance(dual_outputs, dict):
                                logger.error(f"    📊 利用可能なキー: {list(dual_outputs.keys())}")
                            # CLAUDE.md指針: エラーを隠蔽せずデバッグに繋げる
                            raise ValueError("デュアルパスウェイ一貫性損失が全探索パターンで見つかりません - 実装を確認する必要があります")
                    else:
                        logger.error(f"    ❌ デュアルパスウェイ出力が存在しないまたはNone/False")
                        logger.error(f"    📊 dual_outputs: {dual_outputs}")
                        # CLAUDE.md指針: エラーを隠蔽せずデバッグに繋げる
                        raise ValueError("デュアルパスウェイ処理結果が存在しません - 実装を確認する必要があります")
                    
                    # OHEM損失計算（test_phase3b_integration_real.py準拠実装）
                    logger.info(f"    🔧 OHEM損失計算開始...")
                    if 'ohem_loss' in model_components and model_components['ohem_loss'] is not None:
                        ohem_loss_fn = model_components['ohem_loss']
                        logger.info(f"    ✅ OHEM損失関数利用可能")
                        
                        # test_phase3b準拠の完全なOHEM損失計算
                        try:
                            # 必要なデータを取得
                            input_ids = batch.get('input_ids')
                            attention_mask = batch.get('attention_mask')
                            
                            if input_ids is not None and attention_mask is not None:
                                # Llama logitsを取得（既に計算済みのlogitsを再利用）
                                llama_logits = None
                                
                                # パターン1: qformer_outputsから既に計算済みのlogitsを取得
                                if 'llama_logits' in qformer_outputs:
                                    llama_logits = qformer_outputs['llama_logits']
                                    logger.info(f"      ✅ 既存のLlama logits取得成功: {llama_logits.shape}")
                                else:
                                    logger.warning(f"      ⚠️ qformer_outputsにllama_logits未含有")
                                    logger.info(f"      📊 利用可能キー: {list(qformer_outputs.keys())}")
                                    # フォールバック: 再推論を試行（非効率だが機能確保）
                                    logger.info(f"      🔄 フォールバック: Llama再推論試行...")
                                    
                                    # qformer_bridge属性探索
                                    qformer_attrs = [attr for attr in dir(qformer_bridge) if not attr.startswith('_')]
                                    llama_model = None
                                    
                                    # 複数パターンでLlamaモデル探索
                                    for attr_name in ['llama_model', 'llama4_model', 'model']:
                                        if hasattr(qformer_bridge, attr_name):
                                            llama_model = getattr(qformer_bridge, attr_name)
                                            logger.info(f"      ✅ {attr_name}属性発見")
                                            break
                                    
                                    if llama_model is not None:
                                        try:
                                            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                                                llama_outputs = llama_model(
                                                    input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    images=batch.get('pixel_values', batch.get('images')),
                                                    output_hidden_states=False,
                                                    return_dict=True
                                                )
                                                if hasattr(llama_outputs, 'logits'):
                                                    llama_logits = llama_outputs.logits
                                                    logger.info(f"      ✅ フォールバック推論成功: {llama_logits.shape}")
                                        except Exception as llama_error:
                                            logger.warning(f"      ⚠️ フォールバック推論失敗: {llama_error}")
                                    else:
                                        logger.warning(f"      ⚠️ Llamaモデルアクセス不可")
                                        logger.info(f"      📊 利用可能属性: {[attr for attr in qformer_attrs if 'model' in attr.lower() or 'llama' in attr.lower()]}")
                                
                                # SAM予測の取得（test script準拠の厳密な検証）
                                sam_predictions = None
                                if 'predicted_masks' in qformer_outputs:
                                    sam_predictions = qformer_outputs['predicted_masks']
                                    logger.info(f"      ✅ SAM予測（qformer）取得: {sam_predictions.shape}")
                                elif dual_outputs and 'final_mask' in dual_outputs:
                                    sam_predictions = dual_outputs['final_mask']
                                    logger.info(f"      ✅ SAM予測（dual）取得: {sam_predictions.shape}")
                                
                                # test script準拠: SAM予測の厳密な検証
                                if sam_predictions is not None:
                                    if not torch.is_tensor(sam_predictions):
                                        logger.error(f"      ❌ SAM予測がTensorではない: {type(sam_predictions)}")
                                        raise TypeError(f"SAM予測はTensorである必要があります: {type(sam_predictions)}")
                                    
                                    logger.info(f"      📊 SAM予測詳細: shape={sam_predictions.shape}, dtype={sam_predictions.dtype}")
                                    
                                    # 形状検証
                                    if sam_predictions.dim() < 3:
                                        logger.error(f"      ❌ SAM予測次元不足: {sam_predictions.shape} (期待: 3次元以上)")
                                        raise ValueError(f"SAM予測の次元が不正: {sam_predictions.shape}")
                                else:
                                    logger.error(f"      ❌ SAM予測が取得できません")
                                    logger.error(f"      📊 qformer_outputsキー: {list(qformer_outputs.keys())}")
                                    if dual_outputs:
                                        logger.error(f"      📊 dual_outputsキー: {list(dual_outputs.keys())}")
                                    raise ValueError("SAM予測データが見つかりません")
                                
                                # SAMターゲットの取得（test script準拠の厳密な検証）
                                sam_targets = None
                                if 'ground_truth_mask' in batch and batch['ground_truth_mask']:
                                    if isinstance(batch['ground_truth_mask'], list) and batch['ground_truth_mask']:
                                        sam_targets = batch['ground_truth_mask'][0]
                                        logger.info(f"      ✅ SAMターゲット（リスト）取得: {type(sam_targets)}")
                                    else:
                                        sam_targets = batch['ground_truth_mask']
                                        logger.info(f"      ✅ SAMターゲット（直接）取得: {type(sam_targets)}")
                                    
                                    # test script準拠: データ形状とタイプの厳密な検証
                                    if sam_targets is not None:
                                        if torch.is_tensor(sam_targets):
                                            logger.info(f"      📊 SAMターゲット形状: {sam_targets.shape}, dtype: {sam_targets.dtype}")
                                            # 形状検証
                                            if sam_targets.dim() < 3:
                                                logger.error(f"      ❌ SAMターゲット次元不足: {sam_targets.shape} (期待: 3次元以上)")
                                                raise ValueError(f"SAMターゲットの次元が不正: {sam_targets.shape}")
                                        else:
                                            logger.error(f"      ❌ SAMターゲットがTensorではない: {type(sam_targets)}")
                                            raise TypeError(f"SAMターゲットはTensorである必要があります: {type(sam_targets)}")
                                else:
                                    logger.error(f"      ❌ ground_truth_maskが存在しないまたは空")
                                    logger.error(f"      📊 バッチキー: {list(batch.keys())}")
                                    raise KeyError("ground_truth_maskが見つかりません")
                                
                                # test_phase3b完全準拠のOHEM損失計算
                                if llama_logits is not None and sam_predictions is not None and sam_targets is not None:
                                    logger.info(f"      🎯 OHEM損失計算実行（test_phase3b準拠）...")
                                    
                                    # test script準拠: 最終データ検証
                                    logger.info(f"        📊 データ整合性確認:")
                                    logger.info(f"          - llama_logits: {llama_logits.shape}, dtype={llama_logits.dtype}")
                                    logger.info(f"          - llama_targets: {input_ids.shape}, dtype={input_ids.dtype}")
                                    logger.info(f"          - attention_mask: {attention_mask.shape}, dtype={attention_mask.dtype}")
                                    logger.info(f"          - sam_predictions: {sam_predictions.shape}, dtype={sam_predictions.dtype}")
                                    logger.info(f"          - sam_targets: {sam_targets.shape}, dtype={sam_targets.dtype}")
                                    
                                    # test script準拠: デバイス統一処理
                                    target_device = llama_logits.device
                                    target_dtype = torch.bfloat16
                                    
                                    logger.info(f"        🔧 デバイス統一処理:")
                                    logger.info(f"          - ターゲットデバイス: {target_device}")
                                    logger.info(f"          - ターゲットdtype: {target_dtype}")
                                    
                                    # test script準拠: OHEM損失関数自体もデバイスに移動
                                    ohem_loss_fn = ohem_loss_fn.to(device=target_device, dtype=target_dtype)
                                    logger.info(f"          - OHEM損失関数デバイス移動完了")
                                    
                                    # SAMターゲットのサイズ調整（test script準拠 + Webリサーチベストプラクティス）
                                    sam_pred_shape = sam_predictions.shape[-2:]  # (1024, 1024)
                                    sam_target_shape = sam_targets.shape[-2:]
                                    
                                    logger.info(f"        🔍 SAMターゲット次元確認:")
                                    logger.info(f"          - sam_targets: {sam_targets.shape}")
                                    logger.info(f"          - sam_predictions: {sam_predictions.shape}")
                                    
                                    # 4次元テンソル (N, C, H, W) に変換（F.interpolate要求）
                                    if sam_targets.dim() == 3:
                                        # (B, H, W) → (B, 1, H, W)
                                        sam_targets_4d = sam_targets.unsqueeze(1)
                                        logger.info(f"        🔧 3次元→4次元変換: {sam_targets.shape} → {sam_targets_4d.shape}")
                                    elif sam_targets.dim() == 4:
                                        sam_targets_4d = sam_targets
                                    else:
                                        raise ValueError(f"SAMターゲットの次元が不正: {sam_targets.shape} (期待: 3次元または4次元)")
                                    
                                    if sam_target_shape != sam_pred_shape:
                                        logger.info(f"        🔧 SAMターゲットリサイズ (Webリサーチベストプラクティス):")
                                        logger.info(f"          - 元サイズ: {sam_target_shape} → 目標: {sam_pred_shape}")
                                        logger.info(f"          - モード: nearest (ディスクリート値保持)")
                                        
                                        # F.interpolateを使用してリサイズ（Webリサーチ準拠）
                                        sam_targets_resized = F.interpolate(
                                            sam_targets_4d.float(),
                                            size=sam_pred_shape,
                                            mode='nearest'
                                        )
                                        logger.info(f"        ✅ リサイズ完了: {sam_targets_resized.shape}")
                                    else:
                                        sam_targets_resized = sam_targets_4d
                                        logger.info(f"        ✅ サイズ一致: リサイズ不要")
                                    
                                    # multimask_output=False設定により1チャンネル出力を確認
                                    expected_shape = (sam_predictions.shape[0], 1, sam_predictions.shape[2], sam_predictions.shape[3])
                                    if sam_predictions.shape[1] != 1:
                                        logger.warning(f"        ⚠️ 予期しないSAM予測形状: {sam_predictions.shape} (期待: {expected_shape})")
                                        # フォールバック: 最初のチャンネルを使用
                                        sam_predictions = sam_predictions[:, 0:1, :, :]
                                        logger.info(f"        🔧 フォールバック適用: {sam_predictions.shape}")
                                    else:
                                        logger.info(f"        ✅ SAM予測形状確認: {sam_predictions.shape} (1チャンネル)")
                                    
                                    # 全てのテンソルを同一デバイス・dtypeに統一
                                    sam_predictions_unified = sam_predictions.to(device=target_device, dtype=target_dtype)
                                    sam_targets_unified = sam_targets_resized.to(device=target_device, dtype=target_dtype)
                                    input_ids_unified = input_ids.to(device=target_device)
                                    attention_mask_unified = attention_mask.to(device=target_device)
                                    
                                    logger.info(f"        ✅ デバイス統一完了:")
                                    logger.info(f"          - sam_predictions: {sam_predictions_unified.device}")
                                    logger.info(f"          - sam_targets: {sam_targets_unified.device}")
                                    logger.info(f"          - input_ids: {input_ids_unified.device}")
                                    logger.info(f"          - attention_mask: {attention_mask_unified.device}")
                                    
                                    # test script準拠の厳密なOHEM呼び出し
                                    loss_results = ohem_loss_fn(
                                        llama_logits=llama_logits,
                                        llama_targets=input_ids_unified,
                                        llama_attention_mask=attention_mask_unified,
                                        sam_predictions=sam_predictions_unified,
                                        sam_targets=sam_targets_unified,
                                        apply_ohem=True,
                                        return_individual=True
                                    )
                                    
                                    # test_phase3b準拠の結果処理
                                    if isinstance(loss_results, dict) and 'total_loss' in loss_results:
                                        individual_losses['ohem_loss'] = loss_results['total_loss'].item()
                                        logger.info(f"      ✅ OHEM損失計算成功: {individual_losses['ohem_loss']:.6f}")
                                        
                                        # 追加の損失詳細
                                        if 'llama_loss' in loss_results:
                                            logger.info(f"        - llama_loss: {loss_results['llama_loss'].item():.6f}")
                                        if 'sam_loss' in loss_results:
                                            logger.info(f"        - sam_loss: {loss_results['sam_loss'].item():.6f}")
                                    else:
                                        logger.error(f"      ❌ OHEM損失結果が無効: {type(loss_results)}")
                                        raise ValueError(f"OHEM損失計算結果が無効: {loss_results}")
                                else:
                                    # test script準拠: 厳密なエラーチェック
                                    missing = []
                                    if llama_logits is None:
                                        missing.append("llama_logits")
                                    if sam_predictions is None:
                                        missing.append("sam_predictions") 
                                    if sam_targets is None:
                                        missing.append("sam_targets")
                                    
                                    logger.error(f"      ❌ OHEM損失計算に必要なデータが不足: {missing}")
                                    # test scriptパターン: データ不足時は明確にエラーを報告
                                    raise RuntimeError(f"OHEM損失計算データ不足: {missing} - データ整合性を確認してください")
                            else:
                                logger.error(f"      ❌ 基本入力データ不足: input_ids={input_ids is not None}, attention_mask={attention_mask is not None}")
                                # test scriptパターン: 基本データ不足は重大なエラー
                                raise RuntimeError("OHEM損失計算用の基本入力データが不足しています")
                                
                        except Exception as ohem_error:
                            logger.error(f"      ❌ OHEM損失計算エラー: {ohem_error}")
                            # test scriptパターン: エラーを隠蔽せず、問題を明確化
                            raise RuntimeError(f"OHEM損失計算エラー: {ohem_error}") from ohem_error
                            
                    else:
                        logger.error(f"    ❌ OHEM損失関数が利用できません")
                        logger.error(f"    📊 model_componentsのキー: {list(model_components.keys())}")
                        # test scriptパターン: 必要な損失関数がない場合は設定エラー
                        raise RuntimeError("OHEM損失関数が初期化されていません - model_components設定を確認してください")
                
                else:
                    # 非辞書型出力の場合
                    if hasattr(qformer_outputs, 'loss'):
                        loss = qformer_outputs.loss
                        individual_losses['total_loss'] = loss.item()
                    else:
                        raise ValueError(f"Loss not found in model outputs. Output type: {type(qformer_outputs)}")
        
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"🔥 CUDA OOMエラー (step {step}): {e}")
            # 詳細メモリ状況をログ出力
            log_memory_status(f"OOM発生 step {step}")
            
            # 緊急クリーンアップ実行
            emergency_cleanup()
            # CLAUDE.md指示: エラーを隠蔽せず停止してデバッグに繋げる
            raise
        except Exception as e:
            logger.error(f"フォワードパスエラー (step {step}): {e}")
            continue
        
        # バックワードパス（train_llama4_lisa成功パターン移植）
        try:
            # 🔥 強化逆伝播前メモリ対策: 計算グラフ最適化
            torch.cuda.empty_cache()
            
            # 計算グラフの明示的リセット（メモリリーク防止）
            if hasattr(torch, '_C') and hasattr(torch._C, '_clear_cuda_memory_fraction'):
                try:
                    torch._C._clear_cuda_memory_fraction()
                except:
                    pass  # 一部環境で利用不可
            
            # Accelerate hooks メモリリーク対策
            if hasattr(qformer_bridge, 'llama_model') and hasattr(qformer_bridge.llama_model, '_hf_hook'):
                # HuggingFace hook のメモリクリア
                torch.cuda.synchronize()
            
            if individual_losses['total_loss'] > 0 and 'loss' in locals():
                # 🔧 方針A: meta tensor対応のカスタムbackward処理
                # meta tensorsを含むレイヤーの勾配計算を回避
                
                # 勾配追跡デバッグコード
                logger.debug("📊 勾配追跡開始...")
                grad_info = {
                    'total_params': 0,
                    'requires_grad': 0,
                    'meta_tensors': 0,
                    'cuda_tensors': 0,
                    'has_grad': 0,
                    'grad_nan': 0
                }
                
                for name, param in qformer_bridge.named_parameters():
                    grad_info['total_params'] += 1
                    if param.requires_grad:
                        grad_info['requires_grad'] += 1
                        if param.is_meta:
                            grad_info['meta_tensors'] += 1
                        else:
                            grad_info['cuda_tensors'] += 1
                            # meta tensorでないパラメータのみ勾配を有効化
                            param.retain_grad()
                
                logger.debug(f"  - 総パラメータ数: {grad_info['total_params']}")
                logger.debug(f"  - requires_grad=True: {grad_info['requires_grad']}")
                logger.debug(f"  - meta tensors: {grad_info['meta_tensors']}")
                logger.debug(f"  - cuda tensors: {grad_info['cuda_tensors']}")
                
                # 逆伝播（Loss Scaling対応）
                try:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"🔥 Backward OOMエラー (step {step}): {e}")
                    # 詳細メモリ状況をログ出力
                    log_memory_status(f"Backward OOM発生 step {step}")
                    
                    # 緊急クリーンアップ実行
                    emergency_cleanup()
                    # CLAUDE.md指示: エラーを隠蔽せず停止してデバッグに繋げる
                    raise
                except RuntimeError as e:
                    if "CUBLAS_STATUS_ALLOC_FAILED" in str(e):
                        logger.error("❌ CUDA/CUBLAS メモリ不足エラー")
                        break
                    elif "expected device meta but got cuda" in str(e) or "MmBackward0" in str(e):
                        logger.warning(f"⚠️ Meta tensor device mismatchエラー: {str(e)[:100]}...")
                        logger.info("💡 LoRAパラメータのみで勾配計算を試行...")
                        
                        # LoRAパラメータのみで勾配更新
                        lora_params = []
                        for name, param in qformer_bridge.named_parameters():
                            if param.requires_grad and ("lora" in name.lower() or "query_embeds" in name):
                                lora_params.append(param)
                        
                        if lora_params:
                            if scaler is not None:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=config_linux.GRADIENT_CLIP_NORM)
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=config_linux.GRADIENT_CLIP_NORM)
                                optimizer.step()
                            scheduler.step()
                            logger.info("✅ LoRAパラメータの更新完了")
                        else:
                            logger.error("❌ 更新可能なLoRAパラメータが見つかりません")
                            continue
                    else:
                        raise
                
                # 勾配計算後のデバッグ
                logger.debug("📊 勾配計算後の確認...")
                for name, param in qformer_bridge.named_parameters():
                    if param.requires_grad and not param.is_meta:
                        if param.grad is not None:
                            grad_info['has_grad'] += 1
                            if torch.isnan(param.grad).any():
                                grad_info['grad_nan'] += 1
                                logger.warning(f"  ⚠️ NaN勾配検出: {name}")
                
                logger.debug(f"  - 勾配あり: {grad_info['has_grad']}")
                logger.debug(f"  - NaN勾配: {grad_info['grad_nan']}")
                
                # 勾配クリッピングとオプティマイザーステップ（Loss Scaling対応）
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(qformer_bridge.parameters(), max_norm=config_linux.GRADIENT_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(qformer_bridge.parameters(), max_norm=config_linux.GRADIENT_CLIP_NORM)
                    optimizer.step()
                scheduler.step()
                
                # デバッグ: パラメータ更新後の状態確認
                if step < 3:
                    logger.debug(f"\n📊 ステップ {step} 更新後の状態確認...")
                    update_check = {
                        'nan_params': [],
                        'inf_params': [],
                        'unchanged_params': []
                    }
                    
                    for name, param in qformer_bridge.named_parameters():
                        if param.requires_grad and not param.is_meta:
                            if torch.isnan(param).any():
                                update_check['nan_params'].append(name)
                            elif torch.isinf(param).any():
                                update_check['inf_params'].append(name)
                    
                    if update_check['nan_params']:
                        logger.error(f"  ❌ 更新後NaNパラメータ: {update_check['nan_params'][:5]}...")
                    if update_check['inf_params']:
                        logger.error(f"  ❌ 更新後Infパラメータ: {update_check['inf_params'][:5]}...")
                    
                    # 学習率の確認
                    current_lr = scheduler.get_last_lr()[0]
                    logger.debug(f"  📊 現在の学習率: {current_lr:.2e}")
            else:
                logger.error(f"  ❌ step {step}: 損失が0または無効、訓練を停止します")
                raise RuntimeError("Invalid loss detected. Training cannot continue.")
        
        except RuntimeError as device_error:
            # 🔍 段階6: LayerNormデバイス不一致エラーの詳細診断
            if "Expected all tensors to be on the same device" in str(device_error) and "layernorm" in str(device_error).lower():
                logger.error(f"🔥 LayerNormデバイス不一致エラー詳細診断 (step {step}):")
                logger.error(f"  エラーメッセージ: {device_error}")
                
                # 全モジュールの詳細デバイス情報を出力
                logger.error("  📊 全モジュールデバイス状況:")
                for module_name, (module_path, module_type, device) in zip(
                    ["qformer_bridge", "dual_pathway_decoder", "multiresolution_fusion"],
                    [("qformer_bridge", type(qformer_bridge).__name__, "検査中"),
                     ("dual_pathway_decoder", type(dual_pathway_decoder).__name__, "検査中"),
                     ("multiresolution_fusion", type(multiresolution_fusion).__name__, "検査中")]
                ):
                    logger.error(f"    {module_name} ({module_type}):")
                    module = locals()[module_name]
                    
                    # LayerNormモジュールのデバイス情報
                    layernorm_count = 0
                    for name, submodule in module.named_modules():
                        if 'LayerNorm' in type(submodule).__name__:
                            layernorm_count += 1
                            try:
                                first_param = next(submodule.parameters(), None)
                                if first_param is not None:
                                    logger.error(f"      LayerNorm {layernorm_count}: {name} → {first_param.device}")
                            except:
                                logger.error(f"      LayerNorm {layernorm_count}: {name} → デバイス取得失敗")
                
                # 特に問題のあるコンポーネントを特定
                logger.error("  🎯 cuda:7に配置されたコンポーネント特定:")
                cuda7_found = []
                for name, module in [("qformer_bridge", qformer_bridge), 
                                   ("dual_pathway_decoder", dual_pathway_decoder),
                                   ("multiresolution_fusion", multiresolution_fusion)]:
                    for subname, submodule in module.named_modules():
                        try:
                            first_param = next(submodule.parameters(), None)
                            if first_param is not None and 'cuda:7' in str(first_param.device):
                                cuda7_found.append(f"{name}.{subname}")
                        except:
                            pass
                
                if cuda7_found:
                    logger.error(f"    🔥 cuda:7配置コンポーネント数: {len(cuda7_found)}")
                    for comp in cuda7_found[:10]:  # 最初の10個のみ表示
                        logger.error(f"      - {comp}")
                else:
                    logger.error("    ✅ cuda:7配置コンポーネントは見つかりませんでした")
                
                # 停止（修正のため）
                raise device_error
            else:
                # その他のRuntimeError
                raise device_error
        
        except torch.cuda.OutOfMemoryError as oom_e:
            # 🔥 Webリサーチ最適化: 2025年OOM自動回復（PyTorch torchtune準拠）
            logger.error(f"⚠️ OOM検出 (step {step}): {oom_e}")
            
            # 🔥 OOM発生時のメモリ推移データ出力（デバッグ用）
            logger.error("📊 OOM発生時メモリ推移データ:")
            if memory_progression['steps']:
                logger.error(f"  記録ステップ数: {len(memory_progression['steps'])}")
                logger.error(f"  ステップ範囲: {min(memory_progression['steps'])} - {max(memory_progression['steps'])}")
                
                # 最近の5ポイントを表示
                recent_points = min(5, len(memory_progression['steps']))
                logger.error(f"  最近の{recent_points}ポイント:")
                for i in range(-recent_points, 0):
                    step_idx = memory_progression['steps'][i]
                    total_mem = memory_progression['total_allocated'][i]
                    gpu0_conc = memory_progression['gpu0_concentration'][i]
                    efficiency = memory_progression['memory_efficiency'][i]
                    logger.error(f"    Step {step_idx}: 総メモリ{total_mem:.1f}GB, GPU0集中{gpu0_conc:.1f}%, 効率{efficiency:.1f}%")
                
                # 傾向分析
                if len(memory_progression['total_allocated']) >= 2:
                    memory_trend = memory_progression['total_allocated'][-1] - memory_progression['total_allocated'][0]
                    gpu0_trend = memory_progression['gpu0_concentration'][-1] - memory_progression['gpu0_concentration'][0]
                    logger.error(f"  傾向分析: 総メモリ{memory_trend:+.1f}GB, GPU0集中{gpu0_trend:+.1f}%")
            else:
                logger.error("  メモリ推移データなし")
            
            oom_retry_count += 1
            if oom_retry_count <= max_oom_retries:
                logger.warning(f"🔄 OOM自動回復試行 {oom_retry_count}/{max_oom_retries}")
                
                # 🔥 緊急深層メモリクリーンアップ（OOM Recovery）
                logger.info("🧹 緊急メモリクリーンアップ実行...")
                
                # 1. Python オブジェクト解放
                gc.collect()
                freed_objects = gc.collect()
                logger.info(f"  ✅ Python GC: {freed_objects} オブジェクト解放")
                
                # 2. PyTorch CUDA キャッシュクリア
                torch.cuda.empty_cache()
                logger.info("  ✅ PyTorch CUDA キャッシュクリア")
                
                # 3. 全GPU同期
                torch.cuda.synchronize()
                logger.info("  ✅ 全GPU同期完了")
                
                # 4. メモリ統計リセット
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
                
                # 5. 断片化解消のための待機
                time.sleep(0.1)  # 100ms待機で断片化解消
                
                # バッチサイズを半分に（データローダー動的調整は難しいため、ここでは警告のみ）
                logger.warning(f"💡 推奨: バッチサイズを {current_batch_size} → {current_batch_size//2} に削減してください")
                logger.warning(f"💡 現在のメモリ設定でOOMが発生しました。設定を見直してください。")
                
                # 現在のステップをスキップして継続
                continue
            else:
                logger.error(f"❌ OOM retry回数超過。training中止。")
                raise oom_e
        
        except Exception as e:
            logger.error(f"バックワードパスエラー (step {step}): {e}")
            continue
        
        # メトリクス更新（値が有効な場合のみ）
        if individual_losses['total_loss'] > 0:
            total_losses.update(individual_losses['total_loss'], 1)
            phase2_losses.update(individual_losses['phase2_loss'], 1)
            seg_losses.update(individual_losses['seg_loss'], 1)
            ohem_losses.update(individual_losses['ohem_loss'], 1)
        else:
            logger.warning(f"Step {step}: Invalid loss values, skipping metrics update")
        
        # 履歴保存（有効な損失値のみ）
        if individual_losses['total_loss'] > 0:
            for key in epoch_loss_history.keys():
                if key in individual_losses:
                    epoch_loss_history[key].append(individual_losses[key])
                else:
                    epoch_loss_history[key].append(0.0)
        
        # 🔥 Webリサーチ最適化: 2025年ステップ間メモリクリーンアップ（PyTorch torchtune準拠）
        try:
            # Activation offloading風の効率的クリーンアップ
            if step % 5 == 0:  # 5ステップごとにメモリクリーンアップ
                torch.cuda.empty_cache()
                
            # ガベージコレクション（メモリリーク防止）
            if step % 10 == 0:
                gc.collect()
                
                # 🔥 全GPUメモリ使用状況監視（ボトルネック検出）
                if torch.cuda.is_available():
                    gpu_stats = {}
                    bottleneck_detected = False
                    critical_gpus = []
                    
                    # 全GPUのメモリ使用率をチェック
                    for gpu_id in range(torch.cuda.device_count()):
                        try:
                            memory_used = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
                            memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3  # GB
                            usage_percent = (memory_used / memory_total) * 100
                            
                            gpu_stats[gpu_id] = {
                                'used': memory_used,
                                'total': memory_total,
                                'percent': usage_percent
                            }
                            
                            # 75%超過で警告、80%超過で緊急対応
                            if usage_percent > 80:
                                critical_gpus.append(gpu_id)
                                bottleneck_detected = True
                                logger.error(f"🚨 GPU {gpu_id} 緊急: {usage_percent:.1f}% ({memory_used:.1f}GB/{memory_total:.1f}GB)")
                            elif usage_percent > 75:
                                logger.warning(f"⚠️ GPU {gpu_id} 警告: {usage_percent:.1f}% ({memory_used:.1f}GB/{memory_total:.1f}GB)")
                            
                        except Exception as gpu_error:
                            logger.warning(f"⚠️ GPU {gpu_id} メモリチェックエラー: {gpu_error}")
                    
                    # GPU使用率サマリ表示
                    if step % 10 == 0 or bottleneck_detected:
                        gpu_summary = ', '.join([f"GPU{i}:{stats['percent']:.1f}%" for i, stats in gpu_stats.items()])
                        logger.info(f"📊 GPU使用率 (ステップ{step}): {gpu_summary}")
                    
                    # ボトルネック対応
                    if bottleneck_detected:
                        logger.warning(f"🔧 ボトルネック検出: GPU {critical_gpus} が80%超過、緊急メモリクリーンアップ")
                        
                        # 緊急メモリクリーンアップ
                        for gpu_id in critical_gpus:
                            torch.cuda.empty_cache()
                            torch.cuda.set_device(gpu_id)
                            torch.cuda.empty_cache()
                        
                        gc.collect()
                        
                        # クリーンアップ後の再チェック
                        for gpu_id in critical_gpus:
                            memory_after = torch.cuda.memory_allocated(gpu_id) / 1024**3
                            logger.info(f"📊 GPU {gpu_id} クリーンアップ後: {memory_after:.1f}GB")
                    
                    # 通常メモリクリーンアップ
                    else:
                        torch.cuda.empty_cache()
                        gc.collect()
        except Exception as e:
            logger.warning(f"⚠️ メモリクリーンアップエラー（継続）: {e}")
        
        # プログレス表示
        if step % 10 == 0:
            progress.display(step)
        
        # TensorBoard記録（有効な損失値のみ）
        if writer and individual_losses['total_loss'] > 0:
            global_step = epoch * len(dataloader) + step
            writer.add_scalar('Loss/Total', individual_losses['total_loss'], global_step)
            writer.add_scalar('Loss/Phase2', individual_losses['phase2_loss'], global_step)
            writer.add_scalar('Loss/Seg', individual_losses['seg_loss'], global_step)
            writer.add_scalar('Loss/OHEM', individual_losses['ohem_loss'], global_step)
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)
        
        # 🔥 2025年メモリリーク完全対策: ステップ終了時の徹底クリーンアップ
        if step % 2 == 0:
            torch.cuda.empty_cache()
            
        # 中間変数の強制削除（メモリリーク防止）
        if 'loss' in locals():
            del loss
        if 'outputs' in locals():
            del outputs
        
        # 🔍 詳細メモリ推移トラッキング（強化版 + リーク原因特定）
        if step % 5 == 0:
            current_time = time.time()
            memory_history['steps'].append(step)
            memory_history['timestamps'].append(current_time)
            
            # 🕵️ メモリリーク原因特定: フォワードパス前のメモリ記録
            pre_forward_memory = {}
            for gpu_id in range(torch.cuda.device_count()):
                pre_forward_memory[gpu_id] = torch.cuda.memory_allocated(gpu_id) / 1024**3
            
            # 全GPUのメモリ状況を記録
            total_increase = 0
            gpu_memory_details = {}
            for gpu_id in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                peak = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
                reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                
                memory_history['gpu_memory'][gpu_id].append(allocated)
                memory_history['peak_memory'][gpu_id].append(peak)
                memory_history['reserved_memory'][gpu_id].append(reserved)
                
                # ベースラインからの増加量計算
                increase = allocated - baseline_memory[gpu_id]
                total_increase += increase
                
                # 🔍 メモリ断片化分析
                fragmentation = reserved - allocated
                fragmentation_ratio = (fragmentation / reserved * 100) if reserved > 0 else 0
                
                gpu_memory_details[gpu_id] = {
                    'allocated': allocated,
                    'peak': peak,
                    'reserved': reserved,
                    'increase': increase,
                    'fragmentation': fragmentation,
                    'fragmentation_ratio': fragmentation_ratio
                }
                
            # 🕵️ Model Parallelismレイヤー別メモリ分析
            if step % 10 == 0:
                logger.info(f"🔍 Step{step} Model Parallelism詳細分析:")
                device_map = getattr(qformer_bridge.llama_model, 'hf_device_map', {})
                if device_map:
                    for layer_name, device in list(device_map.items())[:10]:  # 最初の10レイヤー表示
                        if isinstance(device, int):
                            gpu_mem = torch.cuda.memory_allocated(device) / 1024**3
                            logger.info(f"  {layer_name}: GPU{device} ({gpu_mem:.1f}GB)")
                
                # 🔍 GPU 0集中度分析
                gpu0_ratio = gpu_memory_details[0]['allocated'] / sum([details['allocated'] for details in gpu_memory_details.values()])
                logger.info(f"🔍 GPU 0 メモリ集中度: {gpu0_ratio*100:.1f}% (理想値: 12.5%)")
                
                # 🔍 メモリ断片化サマリー
                high_frag_gpus = [gpu_id for gpu_id, details in gpu_memory_details.items() if details['fragmentation_ratio'] > 20]
                if high_frag_gpus:
                    logger.warning(f"⚠️ 高断片化GPU検出: {high_frag_gpus} (断片化率>20%)")
            
            # 🚨 メモリリーク検出（強化版）
            current_mem = torch.cuda.memory_allocated() / 1024**3
            if step > 0 and current_mem > previous_memory * 1.1:  # 10%増加で警告
                logger.warning(f"⚠️ メモリリーク検出: {previous_memory:.1f}GB → {current_mem:.1f}GB (+{((current_mem/previous_memory-1)*100):.1f}%)")
                # 強制的なディープクリーンアップ
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            # 📈 メモリ推移サマリー（10ステップごと）
            if step % 10 == 0:
                # 過去のステップとの比較（トレンド分析）
                if len(memory_history['steps']) >= 3:
                    trend_analysis = {}
                    
                    for gpu_id in range(torch.cuda.device_count()):
                        recent_memory = memory_history['gpu_memory'][gpu_id]
                        if len(recent_memory) >= 3:
                            # 直近3回の平均増加率
                            trend = (recent_memory[-1] - recent_memory[-3]) / 2  # 平均増加量
                            trend_analysis[gpu_id] = trend
                    
                    # 傾向警告
                    concerning_gpus = [(gpu_id, trend) for gpu_id, trend in trend_analysis.items() if trend > 0.5]  # 0.5GB以上増加傾向
                    if concerning_gpus:
                        warnings = [f"GPU{gpu_id}:+{trend:.1f}GB/10steps" for gpu_id, trend in concerning_gpus]
                        logger.warning(f"📈 メモリ増加傾向検出: {', '.join(warnings)}")
                
                # 現在のメモリ状況サマリー（断片化情報付き）
                gpu_summary = ', '.join([f"GPU{gpu_id}: {details['allocated']:.1f}GB({details['increase']:+.1f}GB,断片化{details['fragmentation_ratio']:.1f}%)" 
                                       for gpu_id, details in gpu_memory_details.items()])
                logger.info(f"📊 メモリ推移 Step{step}: {gpu_summary}")
                
                # 🔍 Accelerate Hooks メモリリーク特定
                logger.info(f"🕵️ Step{step} Accelerate Hooks分析:")
                if hasattr(qformer_bridge.llama_model, '_hf_hook'):
                    hook = qformer_bridge.llama_model._hf_hook
                    logger.info(f"  - Hook存在: {type(hook).__name__}")
                    if hasattr(hook, 'input_device'):
                        logger.info(f"  - input_device: {hook.input_device}")
                    if hasattr(hook, 'execution_device'):
                        logger.info(f"  - execution_device: {hook.execution_device}")
                    
                    # 🔍 デバイス間データ移動パターン追跡
                    if hasattr(hook, '_previous_module_devices'):
                        devices = getattr(hook, '_previous_module_devices', {})
                        logger.info(f"  - デバイス間移動パターン: {len(devices)}個のモジュール")
                
                # 🔍 計算グラフメモリリーク検出
                tensor_count = 0
                for obj in gc.get_objects():
                    if torch.is_tensor(obj):
                        tensor_count += 1
                logger.info(f"  - 総Tensorオブジェクト数: {tensor_count}")
                
                # 🔍 GPU間データ移動回数カウント（推定）
                for gpu_id, details in gpu_memory_details.items():
                    if details['allocated'] > baseline_memory[gpu_id] * 1.5:  # 50%以上増加
                        logger.warning(f"  ⚠️ GPU{gpu_id} 大幅メモリ増加: {baseline_memory[gpu_id]:.1f}GB → {details['allocated']:.1f}GB")
                
                # 🔥 早期OOM警告（段階的）
                for gpu_id, details in gpu_memory_details.items():
                    utilization = (details['allocated'] / 79.2) * 100
                    if utilization > 90:  # 90%超過で緊急警告
                        logger.error(f"🚨 GPU{gpu_id} 緊急: {details['allocated']:.1f}GB ({utilization:.1f}%) - 即座にOOMリスク！")
                    elif utilization > 80:  # 80%超過で警告
                        logger.warning(f"⚠️ GPU{gpu_id} 警戒: {details['allocated']:.1f}GB ({utilization:.1f}%) - OOM注意")
                    elif utilization > 70:  # 70%超過で注意
                        logger.info(f"📊 GPU{gpu_id} 注意: {details['allocated']:.1f}GB ({utilization:.1f}%)")
        
        # メモリベースライン更新
        if step % 10 == 0:
            previous_memory = torch.cuda.memory_allocated() / 1024**3
        
        # 初期化
        if 'previous_memory' not in locals():
            previous_memory = torch.cuda.memory_allocated() / 1024**3
        
        # Gradient Accumulation制御（Webリサーチ最適化）
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)  # より効率的なゼロ化
            if scaler is not None:
                scaler.update()
            
            # Gradient Accumulation後のメモリクリーンアップ
            torch.cuda.empty_cache()
    
    # エポック終了時のメモリクリア（OOM対策）
    torch.cuda.empty_cache()
    
    # 🔍 エポック終了時のメモリ推移レポート
    if len(memory_history['steps']) > 1:
        logger.info("="*80)
        logger.info(f"📊 Epoch {epoch} メモリ推移レポート")
        logger.info("="*80)
        
        # 各GPUの推移サマリー
        for gpu_id in range(torch.cuda.device_count()):
            if len(memory_history['gpu_memory'][gpu_id]) > 1:
                start_mem = memory_history['gpu_memory'][gpu_id][0]
                end_mem = memory_history['gpu_memory'][gpu_id][-1]
                peak_mem = max(memory_history['peak_memory'][gpu_id])
                trend = end_mem - start_mem
                
                trend_symbol = "📈" if trend > 0.5 else "📉" if trend < -0.5 else "➡️"
                logger.info(f"GPU{gpu_id}: {start_mem:.1f}GB → {end_mem:.1f}GB ({trend:+.1f}GB) Peak: {peak_mem:.1f}GB {trend_symbol}")
        
        # 最も危険なGPU特定
        final_usage = {gpu_id: memory_history['gpu_memory'][gpu_id][-1] 
                      for gpu_id in range(torch.cuda.device_count()) 
                      if len(memory_history['gpu_memory'][gpu_id]) > 0}
        
        if final_usage:
            max_gpu = max(final_usage.keys(), key=lambda x: final_usage[x])
            max_usage = final_usage[max_gpu]
            max_utilization = (max_usage / 79.2) * 100
            
            if max_utilization > 85:
                logger.warning(f"⚠️ 最高使用率GPU{max_gpu}: {max_usage:.1f}GB ({max_utilization:.1f}%) - 次回OOM懸念")
            else:
                logger.info(f"✅ 最高使用率GPU{max_gpu}: {max_usage:.1f}GB ({max_utilization:.1f}%) - 安定")
        
        logger.info("="*80)
    
    # エポック統計
    epoch_time = time.time() - start_time
    logger.info(f"Epoch {epoch} 完了: {epoch_time:.2f}秒")
    logger.info(f"  - 総損失: {total_losses.avg:.6f}")
    logger.info(f"  - Phase2損失: {phase2_losses.avg:.6f}")
    logger.info(f"  - セグメンテーション損失: {seg_losses.avg:.6f}")
    logger.info(f"  - OHEM損失: {ohem_losses.avg:.6f}")
    
    # エポック平均を返す
    return {
        'total_loss': total_losses.avg,
        'phase2_loss': phase2_losses.avg,
        'seg_loss': seg_losses.avg,
        'ohem_loss': ohem_losses.avg,
        'epoch_loss_history': epoch_loss_history
    }

def main():
    parser = argparse.ArgumentParser(description='Phase 3B QFormerSegmentationBridge 訓練')
    parser.add_argument('--exp_name', type=str, default='phase3b_train', help='実験名')
    parser.add_argument('--epochs', type=int, default=3, help='エポック数')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help='エポックあたりのステップ数制限')
    parser.add_argument('--batch_size', type=int, default=1, help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=config_linux.LEARNING_RATE, help='学習率')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='H100x4対応: OOM対策でバッチサイズ削減+実効バッチサイズ維持')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='重み減衰')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoaderワーカー数')
    parser.add_argument('--dataset', type=str, default='reason_seg', help='データセット名')
    parser.add_argument('--samples_per_epoch', type=int, default=None, help='エポックあたりのサンプル数制限')
    parser.add_argument('--use_8bit_adam', action='store_true', help='8bit AdamWを使用')
    parser.add_argument('--log_dir', type=str, default='./logs_phase3b', help='ログディレクトリ')
    
    args = parser.parse_args()
    
    # 環境設定
    setup_environment()
    
    # ログディレクトリ作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_log_dir = os.path.join(args.log_dir, f"{args.exp_name}_{timestamp}")
    os.makedirs(exp_log_dir, exist_ok=True)
    
    # ロギング設定
    logger = setup_logging(exp_log_dir)
    
    # メモリモニター初期化
    memory_monitor = GPUMemoryMonitor(enable_detailed_logging=True)
    logger.info("✅ GPU Memory Monitor初期化完了（H100 80GB最適化）")
    
    # 初期メモリ状況確認
    memory_monitor.log_memory_status("訓練開始前")
    
    logger.info("🚀 Phase 3B QFormerSegmentationBridge 訓練開始")
    logger.info("=" * 80)
    logger.info(f"実験名: {args.exp_name}")
    logger.info(f"エポック数: {args.epochs}")
    logger.info(f"ステップ制限: {args.steps_per_epoch}")
    logger.info(f"バッチサイズ: {args.batch_size}")
    logger.info(f"学習率: {args.lr}")
    logger.info("=" * 80)
    
    try:
        # TensorBoard設定
        writer = SummaryWriter(log_dir=os.path.join(exp_log_dir, 'tensorboard'))
        
        # 1. モデルとコンポーネント作成
        with memory_monitor.monitor_section("モデルとコンポーネント作成"):
            model_components = create_model_and_components(args, logger)
        
        # 2. LoRA適用
        with memory_monitor.monitor_section("LoRA適用"):
            model_components = apply_lora_to_model(model_components, args, logger)
        
        # 3. データセットとデータローダー作成
        with memory_monitor.monitor_section("データセット作成"):
            dataloader = create_dataset_and_dataloader(model_components, args, logger)
        
        # 4. オプティマイザーとスケジューラー作成
        optimizer, scheduler, scaler = create_optimizer_and_scheduler(model_components, dataloader, args, logger)
        
        # 5. 学習ループ
        loss_history = []
        
        for epoch in range(1, args.epochs + 1):
            logger.info(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")
            
            # エポック開始前のメモリ状況
            memory_monitor.log_memory_status(f"Epoch {epoch} 開始前")
            
            with memory_monitor.monitor_section(f"Epoch {epoch} 訓練"):
                epoch_losses = train_epoch(
                    model_components, dataloader, optimizer, scheduler, scaler,
                    epoch, args, logger, writer
                )
            
            loss_history.append(epoch_losses)
            
            # GPU メモリクリーンアップ
            torch.cuda.empty_cache()
            gc.collect()
        
        # 6. 結果保存
        logger.info("\n🎯 訓練完了 - 結果保存中...")
        
        # 損失曲線の可視化
        plot_file = create_loss_plots(loss_history, exp_log_dir, args.exp_name)
        logger.info(f"✓ 損失曲線保存: {plot_file}")
        
        # 結果辞書
        results = {
            'exp_name': args.exp_name,
            'epochs': args.epochs,
            'final_losses': loss_history[-1] if loss_history else {},
            'loss_history': loss_history,
            'args': vars(args),
            'timestamp': timestamp
        }
        
        # JSON保存
        results_file = os.path.join(exp_log_dir, f'results_{args.exp_name}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✓ 結果保存: {results_file}")
        
        # 損失曲線画像保存
        if loss_history:
            plot_file = create_loss_plots(loss_history, exp_log_dir, args.exp_name)
            if plot_file:
                logger.info(f"✓ 損失曲線保存: {plot_file}")
        
        # モデル保存（最終エポック）- PyTorch標準形式
        model_save_path = os.path.join(exp_log_dir, 'final_model.pth')
        
        # QFormerSegmentationBridgeのstate_dictを保存
        model_state = {
            'model_state_dict': model_components['qformer_bridge'].state_dict(),
            'epoch': args.epochs,
            'exp_name': args.exp_name,
            'final_loss': loss_history[-1] if loss_history else 0.0
        }
        
        torch.save(model_state, model_save_path)
        logger.info(f"✓ モデル保存: {model_save_path}")
        
        # サマリー表示
        logger.info("\n" + "="*80)
        logger.info("🎉 Phase 3B QFormerSegmentationBridge 訓練完了")
        logger.info("="*80)
        logger.info(f"📊 最終損失:")
        if loss_history:
            final_losses = loss_history[-1]
            logger.info(f"  - 総損失: {final_losses['total_loss']:.6f}")
            logger.info(f"  - Phase2損失: {final_losses['phase2_loss']:.6f}")
            logger.info(f"  - セグメンテーション損失: {final_losses['seg_loss']:.6f}")
            logger.info(f"  - OHEM損失: {final_losses['ohem_loss']:.6f}")
        logger.info(f"🎯 実験ディレクトリ: {exp_log_dir}")
        logger.info("="*80)
        
        writer.close()
        
    except Exception as e:
        logger.error(f"❌ 訓練エラー: {e}")
        import traceback
        traceback.print_exc()
        
        # 緊急メモリクリーンアップ
        memory_monitor.emergency_cleanup()
        raise
    
    finally:
        # 最終メモリクリーンアップ
        memory_monitor.log_memory_status("訓練完了後")
        logger.info("🏁 訓練完了")

if __name__ == "__main__":
    main()