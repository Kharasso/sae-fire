import argparse
from data_io import TranscriptDataset
from models import SAEExtractor
from experiments import SaeExperiment
from configs import SAE_MODELS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--meta',  required=True)
    parser.add_argument('--out',   required=True)
    parser.add_argument('--sae-id', required=True,
                        choices=list(SAE_MODELS.keys()), help='Which SAE config to use')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--flush', type=int, default=100)
    args = parser.parse_args()

    ds = TranscriptDataset(args.jsonl, args.meta)
    ext = SAEExtractor(args.sae_id, args.device)
    exp = SaeExperiment(ext, ds, args.out, flush_every=args.flush)
    exp.run()