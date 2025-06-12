import argparse
from data_io import TranscriptDataset
from models import ClsExtractor
from experiments import ClsExperiment
from configs import CLS_MODELS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--meta',  required=True)
    parser.add_argument('--out',   required=True)
    parser.add_argument('--cls-id', required=True,
                        choices=list(CLS_MODELS.keys()), help='Which CLS config to use')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--flush', type=int, default=100)
    args = parser.parse_args()

    ds = TranscriptDataset(args.jsonl, args.meta)
    ext = ClsExtractor(args.cls_id, args.device)
    exp = ClsExperiment(ext, ds, args.out, flush_every=args.flush)
    exp.run()