"""Stream a bounded HQ-Edit source/instruction subset without target images."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

from PIL import Image, ImageOps


def image_hash(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def materialize_subset(rows, output, train_size, validation_size, reference_max_side=512):
    output = Path(output)
    for split in ('train', 'validation'):
        (output / split / 'images').mkdir(parents=True, exist_ok=True)
    records = dict(train=[], validation=[])
    selection = dict(train=[], validation=[])
    hashes = set()
    for index, row in enumerate(rows):
        if not isinstance(row['edit'], str) or not row['edit'].strip():
            continue
        source = row['input_image']
        source = source if isinstance(source, Image.Image) else Image.open(BytesIO(source['bytes']))
        image = ImageOps.exif_transpose(source).convert('RGB')
        digest = image_hash(image)
        if digest in hashes or min(image.size) < 64 or max(image.size) / min(image.size) > 8:
            continue
        hashes.add(digest)
        original_size = image.size
        image.thumbnail((reference_max_side, reference_max_side), Image.Resampling.LANCZOS)
        if min(image.size) < 64:
            continue
        split = 'train' if len(records['train']) < train_size else 'validation'
        image.save(output / split / 'images' / f'{digest}.png')
        records[split].append(dict(task='i2i', prompt=row['edit'].strip(), reference_images=[f'images/{digest}.png']))
        selection[split].append(dict(stream_index=index, source_sha256=digest, original_size=list(original_size),
                                     reference_size=list(image.size)))
        count = len(records[split])
        target = train_size if split == 'train' else validation_size
        if count % 64 == 0 or count == target:
            print(f'{split}: {count}/{target} HQ-Edit examples', flush=True)
        if len(records['validation']) == validation_size:
            break
    if len(records['train']) != train_size or len(records['validation']) != validation_size:
        raise ValueError('The stream did not contain enough unique usable HQ-Edit source images')
    for split in records:
        (output / split / 'manifest.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in records[split]))
        (output / split / 'selection.json').write_text(json.dumps(selection[split], indent=2) + '\n')
    return selection


def main(argv=None):
    from datasets import Image as DatasetImage, load_dataset
    from huggingface_hub import HfApi
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--train_size', type=int, default=1024)
    parser.add_argument('--validation_size', type=int, default=64)
    parser.add_argument('--reference_max_side', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--revision')
    args = parser.parse_args(argv)
    if min(args.train_size, args.validation_size) < 1 or args.reference_max_side < 64:
        parser.error('Subset sizes must be positive and reference_max_side >= 64')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'provenance.json').exists():
        raise ValueError('Dataset already materialized; use a new output directory')
    repo = 'UCSC-VLAA/HQ-Edit'
    info = HfApi().dataset_info(repo, revision=args.revision)
    files = [file.rfilename for file in info.siblings if file.rfilename.endswith('.parquet')]
    files.sort(key=lambda name: [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', name)])
    if not files:
        raise ValueError('HQ-Edit has no parquet shards')
    columns = ['input_image', 'edit']
    stream = load_dataset('parquet', data_files=[f'hf://datasets/{repo}@{info.sha}/{file}' for file in files],
                          split='train', streaming=True, columns=columns)
    stream = stream.cast_column('input_image', DatasetImage(decode=False))
    stream = stream.shuffle(seed=args.seed, buffer_size=128)
    materialize_subset(stream, output, args.train_size, args.validation_size, args.reference_max_side)
    provenance = dict(dataset=repo, revision=info.sha, seed=args.seed, train_size=args.train_size,
        validation_size=args.validation_size, reference_max_side=args.reference_max_side,
        streaming=True, shuffle_buffer=128, projected_columns=columns, target_images_downloaded=False,
        split_policy='Unique original source-image pixel hashes; disjoint deterministic subsets of train stream')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('DATASET_READY', output, flush=True)


if __name__ == '__main__':
    main()
