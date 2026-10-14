"""JSONL inputs and content-addressed frozen conditioning caches."""

import hashlib
from collections import OrderedDict
import json
import os
from pathlib import Path
import tempfile

import torch
from torch.utils.data import Dataset


def read_manifest(path, task='mixed', height=512, width=512):
    path = Path(path).resolve()
    records = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                mode = record['task']
                if mode not in ('t2i', 'i2i'):
                    raise ValueError('task must be t2i or i2i')
                if not isinstance(record['prompt'], str) or not record['prompt'].strip():
                    raise ValueError('prompt must be a nonempty string')
                refs = record.get('reference_images', [])
                if not isinstance(refs, list) or any(not isinstance(x, str) for x in refs):
                    raise ValueError('reference_images must be a list of image paths')
                if (mode == 't2i' and refs) or (mode == 'i2i' and not refs):
                    raise ValueError('t2i requires no references; i2i requires at least one')
                if task != 'mixed' and task != mode:
                    continue
                refs = [str((path.parent / ref).resolve()) for ref in refs]
                for ref in refs:
                    if not Path(ref).is_file():
                        raise ValueError(f'reference image does not exist: {ref}')
                h, w = record.get('height', height), record.get('width', width)
                if type(h) is not int or type(w) is not int or min(h, w) < 32:
                    raise ValueError('height and width must be integers >= 32')
                records.append(dict(task=mode, prompt=record['prompt'], reference_images=refs, height=h, width=w))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'{path}:{line_number}: {error}') from error
    if not records:
        raise ValueError(f'No records match task={task!r} in {path}')
    return records


def cache_key(record, identity):
    digest = hashlib.sha256(json.dumps({'record': record, 'identity': identity}, sort_keys=True).encode())
    for ref in record['reference_images']:
        with open(ref, 'rb') as image:
            for chunk in iter(lambda: image.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def map_tensors(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {key: map_tensors(item, fn) for key, item in value.items()}
    if isinstance(value, list):
        return [map_tensors(item, fn) for item in value]
    if isinstance(value, tuple):
        return tuple(map_tensors(item, fn) for item in value)
    return value


def prepare_cache(adapter, records, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    previous = getattr(adapter, 'keep_encoders_on_device', False)
    adapter.keep_encoders_on_device = True
    try:
        for index, record in enumerate(records):
            path = cache_dir / (cache_key(record, adapter.identity) + '.pt')
            if not path.exists():
                sample = adapter.prepare(record)
                sample = map_tensors(sample, lambda x: x.detach().cpu())
                fd, temporary = tempfile.mkstemp(dir=cache_dir, suffix='.tmp')
                os.close(fd)
                try:
                    torch.save(sample, temporary)
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            paths.append(path)
            if (index + 1) % 16 == 0 or index + 1 == len(records):
                print(f'Conditioning cache: {index + 1}/{len(records)}', flush=True)
    finally:
        adapter.keep_encoders_on_device = previous
        if not previous and hasattr(adapter, 'offload_encoders'):
            adapter.offload_encoders()
    return paths


class CachedConditioning(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return torch.load(self.paths[index], map_location='cpu', weights_only=True)


class OnDemandConditioning(Dataset):
    """Encode full prompt corpora lazily, retaining a bounded disk cache.

    Used directly in the training process, never in DataLoader workers: the
    frozen native encoder is resident on the training device between misses.
    Validation entries and unrelated cache files are never evicted.
    """
    def __init__(self, adapter, records, paths, limit):
        if limit < 1 or len(records) != len(paths):
            raise ValueError('A positive cache limit and one path per record are required')
        self.adapter, self.records, self.paths, self.limit = adapter, records, paths, limit
        existing = {p for p in Path(paths[0]).parent.glob('*.pt')}
        self.recent = OrderedDict((p, None) for p in paths if p in existing)
        self._prune()

    def _prune(self):
        while len(self.recent) > self.limit:
            path, _ = self.recent.popitem(last=False)
            path.unlink(missing_ok=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        if not path.exists():
            previous = self.adapter.keep_encoders_on_device
            self.adapter.keep_encoders_on_device = True
            try:
                prepare_cache(self.adapter, [self.records[index]], path.parent)
            finally:
                self.adapter.keep_encoders_on_device = previous
        self.recent[path] = None
        self.recent.move_to_end(path)
        self._prune()
        return torch.load(path, map_location='cpu', weights_only=True)


def list_collate(samples):
    # Native pipelines share references across a batch. Process independent
    # records separately so variable prompts/references cannot be mixed up.
    return samples
