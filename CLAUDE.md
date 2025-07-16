# CLAUDE.md

日本語で応答すること！
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

md_files/phase3_moe_optimization_strategy.mdに沿って実装している。今後の方針はmd_files/phase3c_implementation_specification.md, md_files/phase3d_advanced_optimization_strategy.mdにまとめてある。



## Development Rules & Guidelines

### Lambda Cloud Development Workflow
Lambda Cloud環境での実行を行うので、ローカルファイルの修正を行うたびに、以下のコマンド例を参考に、lambda上に転送し、lambda上で実行すること

```bash
# File transfer to Lambda Cloud
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='verification_output' --exclude='vis_output' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@<ip address>:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

# Remote execution
ssh -i ~/.ssh/lambda_cloud_key ubuntu@<ip address> "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1, ... PYTHONUNBUFFERED=1 python -u <script>.py 2>&1"

or 

ssh -i ~/.ssh/lambda_cloud_key ubuntu@<ip address> "cd /lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 python -u <script>.py 2>&1"
```
