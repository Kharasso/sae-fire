import os
import csv
import json
import pandas as pd
from typing import List, Dict

def load_raw_transcripts(input_dir: str) -> List[Dict]:
    """Read all JSONL transcript component files in a directory."""
    records = []
    for fname in os.listdir(input_dir):
        if not fname.endswith('.jsonl'):
            continue
        path = os.path.join(input_dir, fname)
        with open(path, 'r') as f:
            for line in f:
                records.append(json.loads(line))
    return records

def build_metadata(records: List[Dict]) -> pd.DataFrame:
    """
    From raw transcript records, extract one row per transcript:
    transcript_id, company_id, date, num_components, total_tokens, etc.
    """
    rows = []
    for rec in records:
        tid = rec['transcript_id']
        cid = rec.get('company_id')
        date = rec.get('date')
        comps = rec.get('component_texts', [])
        num_comps = len(comps)
        total_tokens = sum(len(text.split()) for text in comps)
        rows.append({
            'transcript_id': tid,
            'company_id': cid,
            'date': date,
            'num_components': num_comps,
            'total_tokens': total_tokens
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(['company_id','date'])
    return df

def save_metadata(df: pd.DataFrame, out_csv: str):
    """Write DataFrame to CSV."""
    df.to_csv(out_csv, index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Construct transcript metadata CSV")
    parser.add_argument('--input-dir', required=True,
                        help='Directory of transcript JSONL files')
    parser.add_argument('--output-csv', required=True,
                        help='Path to write transcript_metadata.csv')
    args = parser.parse_args()

    recs = load_raw_transcripts(args.input_dir)
    meta = build_metadata(recs)
    save_metadata(meta, args.output_csv)