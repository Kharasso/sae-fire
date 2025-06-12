import json
import pandas as pd
from typing import Dict

class TranscriptDataset:
    """
    Loads transcript component texts from JSONL and metadata from CSV.
    """
    def __init__(self, jsonl_path: str, meta_csv_path: str):
        self.jsonl_path = jsonl_path
        self.meta_csv_path = meta_csv_path

    def load_components(self) -> Dict[str, str]:
        """
        Returns a dict mapping transcript_id to concatenated component text.
        """
        comps = {}
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                tid = obj['transcript_id']
                text = obj.get('component_text', '')
                comps[tid] = comps.get(tid, '') + ' ' + text
        return comps

    def load_metadata(self) -> pd.DataFrame:
        """
        Returns metadata DataFrame indexed by transcript_id.
        """
        df = pd.read_csv(self.meta_csv_path)
        df = df.set_index('transcript_id')
        return df