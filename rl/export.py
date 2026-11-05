"""Package a PPO policy using the existing neural bot and game contract."""
import argparse
from pathlib import Path
import sys
import tempfile
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from starters.shared.bundle import bundle_bot


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--latest',action='store_true',help='Export latest weights instead of the stored best policy')
    args=parser.parse_args(argv)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if not args.latest and saved.get('best_policy') is not None:
        weights=saved['best_policy']
    else:
        weights={k:saved[k] for k in ['kind','activation','negative_slope','model_config','state_dict',
                                     'inference_temperature','algorithm','update'] if k in saved}
    with tempfile.TemporaryDirectory() as scratch:
        path=Path(scratch)/'policy.pt'
        torch.save(weights,path)
        return bundle_bot(ROOT/'starters/neural',args.output,path)


if __name__=='__main__':
    print(main())
