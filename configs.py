from typing import Dict, List, Any

SAE_MODELS: Dict[str, Dict[str, Any]] = {
    # 2B model config
    'sae_2b': {
        'hf_model': 'google/gemma-2-2b',
        'sae_checkpoint': 'gemma-scope-2b-pt-res-canonical',
        'latent_dim': 16384,
        'layers': [5, 12, 20],
    },
    # 9B model config, 16k features
    'sae_9b_16k': {
        'hf_model': 'google/gemma-2-9b',
        'sae_checkpoint': 'gemma-scope-9b-pt-res-canonical',
        'latent_dim': 16384,
        'layers': [9, 20, 31],
    },
    # 9B model, 131k features
    'sae_9b_131k': {
        'hf_model': 'google/gemma-2-9b',
        'sae_checkpoint': 'gemma-scope-9b-pt-res-canonical-131k',
        'latent_dim': 131072,
        'layers': [9, 20, 31],
    },
}

CLS_MODELS: Dict[str, str] = {
    'cls_gemma_2b': 'google/gemma-2-2b',
    'cls_gemma_9b': 'google/gemma-2-9b',
    'cls_qwen_4b':  'qwen/Qwen-4B',
    'cls_llama_3b': 'meta-llama/Llama-3-2.3b',
}