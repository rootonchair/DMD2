"""Prepare the original DMD2 LAION caption corpus and repo COCO evaluation prompts."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle


CAPTION_SHA256 = '5051a4e8745d0c8b0321a36cd22c71823c2054ea34c3a304a3a26c11764c143b'
CAPTION_URL = 'https://huggingface.co/tianweiy/DMD2/resolve/main/data/laion/captions_laion_score6.25.pkl'


class CaptionUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('Caption data must contain plain strings, without Python objects')


def main(argv=None):
    import requests
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', default='data/repo-t2i')
    parser.add_argument('--validation_size', type=int, default=64)
    args = parser.parse_args(argv)
    if args.validation_size < 8:
        parser.error('Use at least eight validation prompts')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'provenance.json').exists():
        raise ValueError('Dataset already prepared; use its existing manifest')
    source = output / 'captions_laion_score6.25.pkl'
    if not source.exists():
        with requests.get(CAPTION_URL, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            with source.with_suffix('.part').open('wb') as stream:
                for chunk in response.iter_content(4 * 1024**2):
                    stream.write(chunk)
        source.with_suffix('.part').replace(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != CAPTION_SHA256:
        raise ValueError('Original repo caption file checksum differs')
    with source.open('rb') as stream:
        captions = CaptionUnpickler(stream).load()
    if not isinstance(captions, list):
        raise ValueError('Expected the original caption list')
    coco = Path(__file__).resolve().parents[1] / 'main/coco_eval/captions_coco14_test.txt'
    def identity(prompt):
        return ' '.join(prompt.casefold().split())
    validation, identities = [], set()
    for prompt in coco.read_text().splitlines():
        key = identity(prompt)
        if key and key not in identities:
            validation.append(prompt.strip())
            identities.add(key)
    if len(validation) < args.validation_size:
        raise ValueError('Insufficient COCO validation captions')
    # Exclude all repo COCO captions from training, not only the 64 selected cases.
    excluded = len(identities)
    count = 0
    manifest = output / 'train.jsonl'
    with manifest.open('w') as stream:
        for prompt in captions:
            if not isinstance(prompt, str):
                continue
            key = identity(prompt)
            if not key or key in identities:
                continue
            identities.add(key)
            stream.write(json.dumps(dict(task='t2i', prompt=prompt.strip(), reference_images=[])) + '\n')
            count += 1
    selected = validation[:args.validation_size]
    (output / 'validation.jsonl').write_text(''.join(json.dumps(dict(task='t2i', prompt=p, reference_images=[])) + '\n' for p in selected))
    provenance = dict(dataset='DMD2 original LAION aesthetic score >6.25 captions',
        source_url=CAPTION_URL, source_sha256=digest, original_captions=len(captions),
        train_size=count, validation_size=len(selected), validation_source=str(coco),
        validation_source_sha256=hashlib.sha256(coco.read_bytes()).hexdigest(),
        excluded_coco_captions=excluded, split_policy='Full unique nonempty LAION corpus, excluding all normalized repo COCO evaluation captions',
        images_downloaded=False, hqedit_used=False)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == '__main__':
    main()
