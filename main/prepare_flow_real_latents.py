"""Convert the DMD2 authors' real LAION SDXL-VAE latents into flow-model latents for the GAN loss.

The authors' SDXL GAN recipe trains its discriminator on real LAION-Aesthetics photos, published as
SDXL VAE latents (data/laion_vae_latents/sdxl_vae_latents_laion_500k, see scripts/download_sdxl.sh and
main/sdxl/generate_vae_latents.py). Each shard is decoded with the same SDXL VAE and re-encoded with the
flow model's native VAE, keeping the caption that came with each photo.
"""

import argparse
import json
import math
from pathlib import Path
import pickle
import zipfile

import numpy as np
from PIL import Image
import torch

from main.flow.adapters import FlowAdapter

SDXL_MODEL_ID = 'stabilityai/stable-diffusion-xl-base-1.0'


class NumpyOnlyUnpickler(pickle.Unpickler):
    """The shards hold only numpy arrays; refuse every other global (no arbitrary code)."""
    ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
               ('numpy.core.numeric', '_frombuffer'), ('numpy._core.numeric', '_frombuffer'),
               ('numpy', 'ndarray'), ('numpy', 'dtype')}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f'Refusing to load {module}.{name} from a real-latent shard')
        return super().find_class(module, name)

    def persistent_load(self, pid):
        raise pickle.UnpicklingError('Real-latent shards must not contain torch storages')


def load_shard(path):
    """Authors' shards are torch zip archives of a pickled numpy dict (protocol 5)."""
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith('/data.pkl')]
        if len(names) != 1:
            raise ValueError(f'Expected one data.pkl in {path}')
        with archive.open(names[0]) as stream:
            data = NumpyOnlyUnpickler(stream).load()
    latents, prompts = data['latents'], data['prompts']
    if latents.ndim != 4 or latents.shape[1] != 4 or len(prompts) != len(latents):
        raise ValueError(f'Unexpected SDXL latent shard layout in {path}')
    return latents, [str(prompt).strip() for prompt in prompts]


def psnr(a, b):
    mse = float(((a.float() - b.float()) ** 2).mean())
    return float('inf') if mse == 0 else 10 * math.log10(4 / mse)


def to_pil(pixels):
    array = ((pixels.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def main(argv=None):
    from diffusers import AutoencoderKL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shards', nargs='+', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_family', default='qwen_image21', choices=['qwen_image21'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='Images per shard; 0 keeps every image')
    parser.add_argument('--previews', type=int, default=4)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adapter = FlowAdapter(args.model_family, device='cuda')
    adapter.pipe.text_encoder.cpu()
    sdxl_vae = AutoencoderKL.from_pretrained(SDXL_MODEL_ID, subfolder='vae').to('cuda').float().eval()
    provenance = dict(source='DMD2 authors real LAION SDXL VAE latents (tianweiy/DMD2, data/laion_vae_latents)',
                      sdxl_vae=SDXL_MODEL_ID, model_id=adapter.model_id, revision=adapter.revision,
                      encoding='SDXL VAE decode (fp32), native flow VAE encode (argmax), normalized and packed',
                      shards=[], roundtrip_psnr=[])
    for shard in args.shards:
        target = output / (Path(shard).stem.replace('vae_latents', 'flow_latents') + '.pt')
        if target.exists():
            provenance['shards'].append(dict(source=str(shard), output=str(target), skipped='exists'))
            continue
        latents, prompts = load_shard(shard)
        count = len(latents) if not args.limit else min(args.limit, len(latents))
        converted, kept = [], []
        with torch.inference_mode():
            for start in range(0, count, args.batch_size):
                batch = torch.from_numpy(latents[start:start + args.batch_size].astype(np.float32)).to('cuda')
                pixels = sdxl_vae.decode(batch / sdxl_vae.config.scaling_factor).sample.clamp(-1, 1)
                packed = adapter.encode(pixels)
                if not torch.isfinite(packed).all():
                    raise FloatingPointError(f'Nonfinite flow latents in {shard} at {start}')
                converted.append(packed.half().cpu())
                kept.extend(prompts[start:start + args.batch_size])
                if start < args.previews:
                    height, width = pixels.shape[-2:]
                    sample = dict(height=height, width=width)
                    for offset in range(min(len(batch), args.previews - start)):
                        roundtrip = adapter.decode(packed[offset:offset + 1].to(adapter.dtype), sample)
                        source = to_pil(pixels[offset])
                        index = f'{Path(shard).stem}-{start + offset:04d}'
                        source.save(output / f'{index}-sdxl.png')
                        roundtrip.save(output / f'{index}-flow-roundtrip.png')
                        reference = torch.from_numpy(np.asarray(source)).float() / 127.5 - 1
                        again = torch.from_numpy(np.asarray(roundtrip.convert('RGB'))).float() / 127.5 - 1
                        provenance['roundtrip_psnr'].append(dict(image=index, psnr=psnr(reference, again)))
                if (start // args.batch_size) % 100 == 0:
                    print(f'{Path(shard).name}: {start + len(batch)}/{count}', flush=True)
        temporary = target.with_suffix('.tmp')
        torch.save(dict(latents=torch.cat(converted), prompts=kept, height=pixels.shape[-2], width=pixels.shape[-1]),
                   temporary)
        temporary.replace(target)
        provenance['shards'].append(dict(source=str(shard), output=str(target), images=len(kept)))
        (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        print(f'wrote {target} ({len(kept)} images)', flush=True)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
