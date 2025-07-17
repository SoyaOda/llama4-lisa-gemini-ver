#!/bin/bash
# Phase 3C統合テスト実行スクリプト（実際のLlama-4モデル使用）

echo "=== Phase 3C統合テスト（実モデル版）開始 ==="
echo "⚠️ 注意: このテストは実際のLlama-4-Scout-17B-16E-Instructをロードします"
echo "   必要メモリ: 約80GB（2x H100推奨）"
echo ""

echo "1. Lambda Cloudへファイル転送..."

# ファイル転送
rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='lambda_results' \
    --exclude='verification_output' \
    --exclude='vis_output' \
    --exclude='.gitignore' \
    -e "ssh -i ~/.ssh/lambda_cloud_key" \
    ./ ubuntu@192.222.55.99:/lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux/

echo ""
echo "2. Lambda Cloud上でテスト実行（実モデル）..."

# テスト実行（OOM回避強化版）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@192.222.55.99 \
    "cd /lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux && \
     source ../../venvs/lisa_gemma_venv/bin/activate && \
     export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512' && \
     export CUDA_VISIBLE_DEVICES=0,1 && \
     export CUDA_LAUNCH_BLOCKING=1 && \
     export TORCH_CUDA_ALLOC_SYNC=1 && \
     export HF_HOME=/lambda/nfs/cache/huggingface && \
     export TRANSFORMERS_CACHE=/lambda/nfs/cache/transformers && \
     echo '🔧 OOM回避設定:' && \
     echo '  - バッチサイズ: 1 (8から削減)' && \
     echo '  - GPUメモリ制限: 35GB/GPU (40GBから削減)' && \
     echo '  - CPU offload: 有効' && \
     echo '  - メモリ分割サイズ: 512MB' && \
     echo '' && \
     PYTHONUNBUFFERED=1 python -u test_phase3c_integration.py 2>&1"

echo ""
echo "=== Phase 3C統合テスト（実モデル版）完了 ==="