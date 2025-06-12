import os
import numpy as np
import pandas as pd
from typing import Any

class BaseExperiment:
    def __init__(self, extractor: Any, dataset: Any, output_dir: str, flush_every: int = 100):
        self.ext = extractor
        self.ds = dataset
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.flush_every = flush_every

    def run(self):
        texts = self.ds.load_components()
        meta = self.ds.load_metadata()
        # for SAE, loop over configured layers; for CLS, only one
        for layer in getattr(self.ext, 'layers', [None]):
            buffers = {}
            ids = []
            for idx, (tid, txt) in enumerate(texts.items(), start=1):
                acts = self.ext.encode(txt, layer) if layer is not None else self.ext.encode(txt)
                feats = self.ext.stats(acts)
                for name, vec in feats.items():
                    buffers.setdefault(name, []).append(vec)
                ids.append(tid)
                if idx % self.flush_every == 0:
                    self._flush(buffers, ids, meta, layer)
                    buffers, ids = {}, []
            if ids:
                self._flush(buffers, ids, meta, layer)

    def _flush(self, buffers: dict, ids: list, meta: pd.DataFrame, layer: int = None):
        layer_suffix = f"_layer{layer}" if layer is not None else ''
        for name, vecs in buffers.items():
            arr = np.stack(vecs)
            fname = f"{name}{layer_suffix}_{ids[0]}_{ids[-1]}.npz"
            np.savez_compressed(os.path.join(self.output_dir, fname), arr=arr)
        subset = meta.loc[ids]
        subset.to_csv(os.path.join(self.output_dir, f"meta{layer_suffix}_{ids[0]}_{ids[-1]}.csv"))

class SaeExperiment(BaseExperiment):
    pass

class ClsExperiment(BaseExperiment):
    pass