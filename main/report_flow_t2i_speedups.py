"""Score a main.evaluate_flow_t2i_speedups run and render a page that hides model names until revealed.

Metrics are support only: colour numbers have misled this project before, so the page leads with the images.
"""

import argparse
import html
import json
from pathlib import Path
import random

import lpips
import numpy as np
from PIL import Image
import torch

LABELS = dict(e_stochastic='E 16k EMA · stochastic 4', e_euler='E 16k EMA · Euler 4', lightning='LightX2V Lightning · 4',
              dmd2_2step='DMD2 2-step v1.5 · 2', turbo='Official Turbo · 8', fun_acc='Fun-Acc · 4')
ORDER = ('e_stochastic', 'e_euler', 'lightning', 'dmd2_2step', 'turbo', 'fun_acc')


def tensor(path, device):
    array = np.asarray(Image.open(path).convert('RGB').resize((512, 512), Image.LANCZOS)).astype(np.float32)
    return torch.from_numpy(array / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).to(device)


def score(root, variants, device='cuda'):
    metric = lpips.LPIPS(net='vgg', verbose=False).to(device).eval()
    teacher = {(item['index'], item['seed']): item for item in json.loads((root / 'teacher/results.json').read_text())}
    table = {}
    with torch.inference_mode():
        for variant in variants:
            results = json.loads((root / variant / 'results.json').read_text())
            drift, distance, by_key = {k: [] for k in ('saturation_p95', 'saturation_mean', 'clipped_fraction', 'luminance_std')}, [], {}
            for item in results:
                key = (item['index'], item['seed'])
                by_key[key] = item
                for name in drift:
                    drift[name].append(item['statistics'][name] - teacher[key]['statistics'][name])
                distance.append(float(metric(tensor(root / variant / f'{key[0]:03d}-seed{key[1]}.png', device),
                                             tensor(root / 'teacher' / f'{key[0]:03d}-seed{key[1]}.png', device))))
            seeds = sorted({key[1] for key in by_key})
            variety = []
            for index, seed in by_key:
                if seed == seeds[-1] and len(seeds) > 1:
                    variety.append(float(metric(tensor(root / variant / f'{index:03d}-seed{seeds[0]}.png', device),
                                                tensor(root / variant / f'{index:03d}-seed{seed}.png', device))))
            summary = json.loads((root / variant / 'summary.json').read_text())
            table[variant] = dict(images=len(results), steps=summary['recipe'].get('steps'),
                seconds=summary['median_seconds'], lpips_to_teacher=float(np.mean(distance)),
                seed_variety_lpips=float(np.mean(variety)) if variety else None,
                **{f'{name}_drift': float(np.mean(values)) for name, values in drift.items()})
    teacher_variety = []
    with torch.inference_mode():
        for index, seed in teacher:
            if seed == 43:
                teacher_variety.append(float(metric(tensor(root / f'teacher/{index:03d}-seed42.png', device),
                                                    tensor(root / f'teacher/{index:03d}-seed43.png', device))))
    table['teacher'] = dict(images=len(teacher), seed_variety_lpips=float(np.mean(teacher_variety)) if teacher_variety else None,
        seconds=json.loads((root / 'teacher/summary.json').read_text())['median_seconds'], steps=40)
    return table


def blind_letters(variants, seed=20261102):
    shuffled = list(variants)
    random.Random(seed).shuffle(shuffled)
    return shuffled, {variant: chr(65 + i) for i, variant in enumerate(shuffled)}


def thumbnails(root, variants, size=640, quality=82):
    # Directories use the blind letters so image URLs do not reveal the model.
    letters = {**blind_letters(variants)[1], 'teacher': 'teacher'}
    for variant in (*variants, 'teacher'):
        target = root / 'page' / letters[variant]
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted((root / variant).glob('*.png')):
            destination = target / (path.stem + '.jpg')
            if not destination.exists():
                Image.open(path).convert('RGB').resize((size, size), Image.LANCZOS).save(destination, quality=quality)


def render(root, variants, table):
    shuffled, letters = blind_letters(variants)
    labels = {letters[v]: LABELS[v] for v in variants}
    results = json.loads((root / 'teacher/results.json').read_text())
    sections = []
    for item in sorted(results, key=lambda x: (x['seed'], x['index'])):
        key = f"{item['index']:03d}-seed{item['seed']}"
        cells = [f'<figure><img loading="lazy" src="page/teacher/{key}.jpg" alt="Teacher"><figcaption>Teacher · 40 steps</figcaption></figure>']
        for variant in shuffled:
            letter = letters[variant]
            cells.append(f'<figure><img loading="lazy" src="page/{letter}/{key}.jpg" alt="Model {letter}"><figcaption data-letter="{letter}">Model {letter}</figcaption></figure>')
        sections.append(f'<section><h2>{html.escape(item["prompt"])}</h2><p class="seed">Prompt {item["index"] + 1} · seed {item["seed"]}</p><div class="grid">{"".join(cells)}</div></section>')
    head = ('<tr><th>Model</th><th>Steps</th><th>Median s/image</th><th>LPIPS to teacher</th><th>Seed variety</th><th>Saturation p95 drift</th><th>Clipped drift</th></tr>')
    rows = []
    for variant in (*[v for v in ORDER if v in variants],):
        row = table[variant]
        rows.append(f'<tr><td>{html.escape(LABELS[variant])}</td><td>{row["steps"]}</td><td>{row["seconds"]:.1f}</td><td>{row["lpips_to_teacher"]:.3f}</td>'
                    f'<td>{row["seed_variety_lpips"]:.3f}</td><td>{row["saturation_p95_drift"]:+.3f}</td><td>{row["clipped_fraction_drift"]:+.3f}</td></tr>')
    teacher = table['teacher']
    rows.append(f'<tr><td>Teacher · CFG 2.5</td><td>40</td><td>{teacher["seconds"]:.1f}</td><td>–</td><td>{teacher["seed_variety_lpips"]:.3f}</td><td>–</td><td>–</td></tr>')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen Speedup Comparison</title><style>
:root{{--bg:#fbfaf7;--fg:#1f2125;--muted:#62666e;--line:#dcdad3;--panel:#fff;--accent:#0b63ce}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#14161a;--fg:#e8e9ec;--muted:#a0a4ad;--line:#30343b;--panel:#1c1f24;--accent:#7db4ff}}}}
:root[data-theme="dark"]{{--bg:#14161a;--fg:#e8e9ec;--muted:#a0a4ad;--line:#30343b;--panel:#1c1f24;--accent:#7db4ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:24px 16px 64px}}h1{{font-size:26px;margin:0 0 6px}}h2{{font-size:16px;font-weight:600;margin:0}}
.lead,.seed{{color:var(--muted);margin:2px 0 12px}}button{{font:inherit;padding:8px 14px;border:1px solid var(--line);border-radius:7px;background:var(--panel);color:var(--fg);cursor:pointer}}
#metrics{{display:none;overflow-x:auto;margin:16px 0}}.revealed #metrics{{display:block}}table{{border-collapse:collapse;background:var(--panel);min-width:640px}}td,th{{border:1px solid var(--line);padding:6px 10px;text-align:left;font-variant-numeric:tabular-nums}}
section{{margin-top:28px;border-top:1px solid var(--line);padding-top:14px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}}
figure{{margin:0}}img{{width:100%;aspect-ratio:1;display:block;border-radius:5px;background:var(--line)}}figcaption{{font-size:13px;margin-top:4px;color:var(--muted)}}
</style><main id="root"><h1>Qwen Image 2.1 speedup comparison</h1>
<p class="lead">Text-to-image at 1024×1024, BF16, same prompts and seeds. Each model uses its published recipe. Rate plastic look and prompt fidelity before you reveal the names.</p>
<button id="reveal">Reveal names and metrics</button>
<div id="metrics"><table>{head}{"".join(rows)}</table><p class="lead">E's timing includes moving its text encoder on and off the GPU, so it is not a fair speed figure. Seed variety is the LPIPS between seeds 42 and 43 of one prompt; a low value suggests repeated templates.</p></div>
{"".join(sections)}</main><script>
const names={json.dumps(labels)};
document.getElementById('reveal').addEventListener('click',()=>{{document.getElementById('root').classList.add('revealed');
document.querySelectorAll('figcaption[data-letter]').forEach(n=>{{n.textContent=names[n.dataset.letter]}});}});
</script></html>'''
    (root / 'index.html').write_text(page)
    (root / 'blind_key.json').write_text(json.dumps(letters, indent=2) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--variants', nargs='+', required=True, choices=ORDER)
    args = parser.parse_args(argv)
    root = Path(args.root)
    table = score(root, args.variants)
    (root / 'metrics.json').write_text(json.dumps(table, indent=2) + '\n')
    thumbnails(root, args.variants)
    render(root, args.variants, table)
    print(json.dumps(table, indent=2))


if __name__ == '__main__':
    main()
