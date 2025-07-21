#!/usr/bin/env python3
"""
重複ロード修正版 train_phase3b_qformer_bridge.py

問題: create_dual_pathway_decoder()がSAM2を重複ロード
解決: 既存のSAM2インスタンスを再利用
"""

def create_phase3b_model_fixed(logger):
    """重複ロード修正版Phase 3Bモデル作成"""
    logger.info("=== Phase 3B統合モデル初期化（重複ロード修正版） ===")
    
    try:
        # 設定取得
        lisa_config = config_linux.get_lisa_model_config()
        mle_config = config_linux.get_mle_config()
        qformer_config = LlamaQFormerSAM2Config()
        qformer_config.enable_moe = True
        qformer_config.enable_dual_pathway = True
        
        # GPU分散設定（メモリ最適化版）
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 2:
            max_memory = {
                0: "32GiB",   # 🔥 GPU0負荷軽減
                1: "45GiB",   # 🔥 GPU1活用増
            }
            device_map = "balanced"
        else:
            max_memory = {0: "70GiB"}
            device_map = "auto"
        
        # 量子化設定
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_threshold=6.0,
        )
        
        # 🔥 修正1: 共有Llama-4インスタンス初期化（1回のみ）
        logger.info("🧠 共有Llama-4-Scout初期化...")
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
        
        shared_llama4_processor = AutoProcessor.from_pretrained(
            qformer_config.llama_model_id,
            trust_remote_code=True
        )
        logger.info(f"✓ 共有Llama-4初期化完了")
        
        # 🔥 修正2: QFormerSegmentationBridge初期化（共有インスタンス使用）
        logger.info("🔍 QFormerSegmentationBridge初期化...")
        model = QFormerSegmentationBridge(
            config=qformer_config,
            shared_llama_model=shared_llama4_model,    # 共有インスタンス
            shared_llama_processor=shared_llama4_processor,
            enable_moe=True,
            training_stage=1
        )
        logger.info(f"✓ QFormerSegmentationBridge初期化完了")
        
        # 🔥 修正3: 既存SAM2インスタンスを取得（重複回避）
        if qformer_config.enable_dual_pathway:
            logger.info("🔄 デュアルパスウェイデコーダー統合...")
            
            # ✅ QFormerSegmentationBridge内のSAM2インスタンスを再利用
            existing_sam2 = None
            if hasattr(model, 'sam2_model') and model.sam2_model is not None:
                existing_sam2 = model.sam2_model
                logger.info("✅ 既存SAM2インスタンス発見、再利用します")
            else:
                logger.warning("⚠️ 既存SAM2インスタンス未発見")
            
            # 🔥 修正4: create_dual_pathway_decoder_shared()を使用
            dual_decoder = create_dual_pathway_decoder_shared(
                llama_hidden_size=qformer_config.llama_config['hidden_size'],
                sam_embed_dim=qformer_config.sam_config['prompt_embed_dim'],
                output_size=(512, 512),
                existing_sam2_model=existing_sam2  # 🆕 既存SAM2を渡す
            )
            
            # モデルに統合
            model.dual_pathway_decoder = dual_decoder
            logger.info(f"✓ デュアルパスウェイデコーダー統合完了（重複回避）")
        
        # パラメータ統計
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✓ Phase 3Bモデル統計（重複ロード修正版）:")
        logger.info(f"  - 総パラメータ: {total_params:,}")
        logger.info(f"  - 学習可能パラメータ: {trainable_params:,}")
        logger.info(f"  - 学習可能割合: {100 * trainable_params / total_params:.3f}%")
        logger.info(f"  - メモリ最適化: CPU offload + 重複ロード回避")
        
        return model, shared_llama4_processor
        
    except Exception as e:
        logger.error(f"Phase 3Bモデル初期化エラー: {e}")
        raise

def create_dual_pathway_decoder_shared(
    llama_hidden_size: int = 5120,
    sam_embed_dim: int = 256,
    output_size: tuple = (512, 512),
    existing_sam2_model = None,  # 🆕 既存SAM2モデル
    fusion_strategy: str = "learned_weighted"
):
    """
    重複ロード回避版デュアルパスウェイデコーダ作成
    
    Args:
        existing_sam2_model: 既存のSAM2インスタンス（重複回避用）
    """
    from model.dual_pathway_decoder import Llama4SAM2DualPathwayDecoder
    
    print("🔄 デュアルパスウェイデコーダ作成中（重複回避版）...")
    
    # GPU確認
    if not torch.cuda.is_available():
        raise RuntimeError("GPU環境が必須です")
    
    # 🔥 修正: 既存SAM2インスタンスを渡す
    decoder = Llama4SAM2DualPathwayDecoder(
        llama_hidden_size=llama_hidden_size,
        sam_output_dim=sam_embed_dim,
        fusion_strategy=fusion_strategy
    )
    
    # 🔥 修正: 既存SAM2インスタンスを設定（新規ロード回避）
    if existing_sam2_model is not None:
        decoder.set_sam2_decoder(existing_sam2_model)
        print(f"✅ 既存SAM2インスタンス設定完了（重複ロード回避）")
    else:
        print(f"⚠️ 新規SAM2ロードが必要です")
    
    return decoder

# 🚀 使用例の修正
def main_fixed():
    """修正版メイン関数"""
    
    # 既存のコードをcreate_phase3b_model_fixed()に置き換え
    model, processor = create_phase3b_model_fixed(logger)
    
    # 残りの処理は同じ
    dataset, dataloader = create_dataset_and_dataloader(processor, args, logger)
    # ...

if __name__ == "__main__":
    main_fixed()