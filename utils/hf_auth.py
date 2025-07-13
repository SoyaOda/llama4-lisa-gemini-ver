#!/usr/bin/env python3
"""
HuggingFace自動認証ユーティリティ
トークンファイルまたは環境変数からHuggingFaceに自動ログインします
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def get_hf_token() -> Optional[str]:
    """HuggingFaceトークンを取得"""
    # 1. 環境変数から取得
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if token:
        return token
    
    # 2. プロジェクトルートのhf_token.txtから取得
    project_root = Path(__file__).parent.parent
    token_file = project_root / "hf_token.txt"
    
    if token_file.exists():
        try:
            with open(token_file, 'r', encoding='utf-8') as f:
                token = f.read().strip()
                if token:
                    return token
        except Exception as e:
            print(f"警告: トークンファイル読み込みエラー: {e}")
    
    # 3. ホームディレクトリの.hf_tokenから取得
    home_token_file = Path.home() / ".hf_token"
    if home_token_file.exists():
        try:
            with open(home_token_file, 'r', encoding='utf-8') as f:
                token = f.read().strip()
                if token:
                    return token
        except Exception as e:
            print(f"警告: ホームトークンファイル読み込みエラー: {e}")
    
    return None


def is_hf_logged_in() -> bool:
    """HuggingFaceにログイン済みかチェック"""
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def login_to_hf(token: str) -> bool:
    """HuggingFaceにログイン"""
    try:
        # 環境変数でトークンを設定
        env = os.environ.copy()
        env['HF_TOKEN'] = token
        
        # huggingface-cli loginを実行
        result = subprocess.run(
            ["huggingface-cli", "login", "--token", token],
            capture_output=True,
            text=True,
            env=env,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ HuggingFaceログイン成功")
            return True
        else:
            print(f"❌ HuggingFaceログイン失敗: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ HuggingFaceログインエラー: {e}")
        return False


def ensure_hf_login() -> bool:
    """HuggingFaceログインを確保（メイン関数）"""
    print("HuggingFace認証状況を確認中...")
    
    # 既にログイン済みかチェック
    if is_hf_logged_in():
        print("✅ HuggingFaceに既にログイン済みです")
        return True
    
    print("HuggingFaceにログインしていません。自動ログインを試行します...")
    
    # トークンを取得
    token = get_hf_token()
    if not token:
        print("❌ HuggingFaceトークンが見つかりません")
        print("以下のいずれかの方法でトークンを設定してください:")
        print("  1. 環境変数: export HF_TOKEN=your_token")
        print("  2. プロジェクトルート: hf_token.txt ファイル")
        print("  3. ホームディレクトリ: ~/.hf_token ファイル")
        print("  4. 手動ログイン: huggingface-cli login")
        return False
    
    # ログイン実行
    if login_to_hf(token):
        # 環境変数にも設定（セッション中の後続処理のため）
        os.environ['HF_TOKEN'] = token
        return True
    else:
        return False


def main():
    """スタンドアロン実行用"""
    success = ensure_hf_login()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()