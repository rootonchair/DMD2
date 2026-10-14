from pathlib import Path

import torch

from main.flow.data import OnDemandConditioning, cache_key


def test_lazy_conditioning_reuses_and_evicts_only_training_files(tmp_path):
    class Adapter:
        identity = {'model': 'test'}
        keep_encoders_on_device = False
        calls = []
        def prepare(self, record):
            self.calls.append(record['prompt'])
            return {'value': torch.tensor(len(self.calls))}
    adapter = Adapter()
    records = [dict(task='t2i', prompt=p, reference_images=[]) for p in ['a', 'b', 'c']]
    paths = [tmp_path / (cache_key(r, adapter.identity) + '.pt') for r in records]
    unrelated = tmp_path / 'validation.pt'
    torch.save({'value': torch.tensor(9)}, unrelated)
    dataset = OnDemandConditioning(adapter, records, paths, 2)
    assert not adapter.calls
    assert dataset[0]['value'] == 1
    assert dataset[1]['value'] == 2
    assert dataset[0]['value'] == 1  # hit makes a most recent
    assert dataset[2]['value'] == 3
    assert paths[0].exists() and not paths[1].exists() and paths[2].exists()
    assert unrelated.exists() and not adapter.keep_encoders_on_device
    assert dataset[1]['value'] == 4
    assert not paths[0].exists()


def test_t2i_validation_comparison_contains_teacher_current_and_full_prompt(tmp_path, monkeypatch):
    from PIL import Image
    from main.flow.validation import evaluate_student
    class Adapter:
        device = torch.device('cpu')
        def decode(self, *args):
            return Image.new('RGB', (512, 512), (100, 120, 140))
        def offload_encoders(self):
            pass
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr('main.flow.losses.sample_student', lambda *a: torch.zeros(1, 4, 8))
    prompt = 'A ceramic teapot with blue flowers on a table beside a window. ' * 8
    record = dict(prompt=prompt, reference_images=[])
    teacher_path = tmp_path / 'teacher.png'
    Image.new('RGB', (512,512), (220,100,80)).save(teacher_path)
    teachers = [dict(index=0,seed=42,prompt=prompt,teacher_steps=50,image=str(teacher_path))]
    base = evaluate_student(Adapter(), model, [{}], [record], tmp_path / 'base', [42])
    result = evaluate_student(Adapter(), model, [{}], [record], tmp_path / 'current', [42], base['results'], teachers)
    assert result['artifact_gate_passed']
    with Image.open(tmp_path / 'current/000-seed42-comparison.png') as image:
        assert image.width == 1024 and image.height > 600
        assert image.info['Prompt'] == prompt
        layout = result['results'][0]['comparison']
        assert ' '.join(layout['prompt_lines']) == 'Prompt: ' + ' '.join(prompt.split())
        assert layout['panels'][0]['label'] == 'Teacher, 50 steps'
        assert image.getpixel((10,layout['image_y']+10)) == (220,100,80)
        assert image.getpixel((522,layout['image_y']+10)) == (100,120,140)
