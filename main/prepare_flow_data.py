"""Cache native text/reference conditioning before training."""

import argparse

from main.flow.adapters import FlowAdapter, MODEL_IDS
from main.flow.data import prepare_cache, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_family', choices=list(MODEL_IDS), required=True)
    parser.add_argument('--model_id')
    parser.add_argument('--revision')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--cache_dir', required=True)
    parser.add_argument('--task', choices=['t2i', 'i2i', 'mixed'], default='mixed')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--teacher_guidance', type=float)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    records = read_manifest(args.manifest, args.task, args.height, args.width)
    adapter = FlowAdapter(args.model_family, args.model_id, args.revision, args.device, teacher_guidance=args.teacher_guidance)
    prepare_cache(adapter, records, args.cache_dir)
    adapter.offload_encoders()


if __name__ == '__main__':
    main()
