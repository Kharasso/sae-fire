import torch
import numpy as np
from typing import Dict
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from configs import SAE_MODELS, CLS_MODELS

class ActivationHook:
    """
    Context manager to grab activations from a specific layer.
    """
    def __init__(self, model: torch.nn.Module, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.activations = None
        self.handle = None

    def _hook(self, module, input, output):
        self.activations = output.detach().cpu().numpy()

    def __enter__(self):
        layer = self.model.transformer.h[self.layer_idx].mlp
        self.handle = layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.handle.remove()

    def get(self) -> np.ndarray:
        return self.activations

class SAEExtractor:
    def __init__(self, sae_id: str, device: str = 'cpu'):
        cfg = SAE_MODELS[sae_id]
        self.hf_model = cfg['hf_model']
        self.sae_checkpoint = cfg['sae_checkpoint']
        self.latent_dim = cfg['latent_dim']
        self.layers = cfg['layers']
        self.device = device

        from torch import load
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_model).to(device)
        self.sae = load(self.sae_checkpoint, map_location=device)
        self.sae.eval()

    def encode(self, text: str, layer_idx: int) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True).to(self.device)
        with ActivationHook(self.model, layer_idx=layer_idx) as hook:
            _ = self.model(**inputs)
            acts = hook.get()[0]  # (seq_len, hidden)
        sae_out = self.sae(torch.from_numpy(acts).to(self.device)).detach().cpu().numpy()
        return sae_out  # (seq_len, latent_dim)

    def stats(self, acts: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            'sum': np.sum(acts, axis=0),
            'mean': np.mean(acts, axis=0),
            'max': np.max(acts, axis=0),
            'last': acts[-1],
        }

class ClsExtractor:
    def __init__(self, cls_id: str, device: str = 'cpu'):
        self.hf_model = CLS_MODELS[cls_id]
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
        self.model = AutoModel.from_pretrained(self.hf_model).to(device)
        self.d = self.model.config.hidden_size

    def encode(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True).to(self.device)
        out = self.model(**inputs)
        h = out.last_hidden_state.detach().cpu().numpy()[0]
        return h  # (seq_len, hidden)

    def stats(self, h: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            'cls': h[0],
            'mean': np.mean(h, axis=0),
        }