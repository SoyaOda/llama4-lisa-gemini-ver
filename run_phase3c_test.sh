#!/bin/bash
# Phase 3C統合テスト実行スクリプト

echo "=== Phase 3C統合テスト開始 ==="
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
echo "2. Lambda Cloud上でテスト実行..."

# テスト実行
ssh -i ~/.ssh/lambda_cloud_key ubuntu@192.222.55.99 \
    "cd /lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux && \
     source ../../venvs/lisa_gemma_venv/bin/activate && \
     CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     PYTHONUNBUFFERED=1 python -u test_phase3c_integration.py 2>&1"

echo ""
echo "=== Phase 3C統合テスト完了 ==="