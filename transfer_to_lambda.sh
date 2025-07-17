#!/bin/bash
# Lambda Cloudへのファイル転送スクリプト

echo "Lambda Cloudへファイル転送を開始..."
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='verification_output' --exclude='vis_output' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@192.222.55.99:/lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux/

echo "転送完了"