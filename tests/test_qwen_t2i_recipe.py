import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from main.flow.adapters import QWEN_DISTILL_SCHEDULE_POLICY
from main.flow.comparisons import prepare_teacher_references
from main.flow.data import cache_key, read_manifest
from main.flow.validation import evaluate_student
from scripts.run_qwen_image21_t2i import training_command


def test_offline_sharded_transformer_loading_is_explicit():
    from main.flow.adapters import FlowAdapter
    calls = []
    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(kwargs)
            return cls()
        def to(self, device): return self
    adapter = FlowAdapter.__new__(FlowAdapter)
    adapter.transformer_class = Model
    adapter.model_id, adapter.revision = 'base', 'pinned'
    adapter.dtype, adapter.device, adapter.local_files_only = torch.bfloat16, 'cpu', True
    adapter.load_transformer()
    assert calls[0]['local_files_only'] is True


@pytest.mark.parametrize('family,steps,scale', [('flux_klein', 50, 4), ('qwen_image21', 40, 2.5)])
def test_native_references_use_family_cfg_and_preserve_rng(tmp_path, family, steps, scale):
    calls = []
    original_scheduler = FlowMatchEulerDiscreteScheduler(shift_terminal=.02 if family == 'qwen_image21' else None)
    class Pipe:
        transformer = None
        scheduler = original_scheduler
        def to(self, device): return self
        def set_progress_bar_config(self, **kwargs): pass
        def __call__(self, **kwargs):
            assert self.scheduler.config.shift_terminal == original_scheduler.config.shift_terminal
            calls.append(kwargs)
            torch.randn(5)
            return SimpleNamespace(images=[Image.new('RGB', (32, 32), 'red')])
    pipe = Pipe()
    adapter = SimpleNamespace(family=family, model_id='base', revision='pinned', teacher_guidance=scale,
        device=torch.device('cpu'), dtype=torch.bfloat16, pipe=pipe, offload_encoders=lambda: None)
    records = [dict(prompt='A photo', reference_images=[], height=32, width=32)]
    rng = torch.get_rng_state().clone()
    result = prepare_teacher_references(adapter, object(), records, tmp_path, [42, 43])
    assert torch.equal(torch.get_rng_state(), rng)
    assert pipe.transformer is None and pipe.scheduler is original_scheduler
    assert len(result) == 2 and all(row['teacher_steps'] == steps for row in result)
    for kwargs in calls:
        assert kwargs['num_inference_steps'] == steps
        if family == 'qwen_image21':
            assert kwargs['true_cfg_scale'] == scale and kwargs['negative_prompt'] == ''
            assert 'guidance_scale' not in kwargs
        else:
            assert kwargs['guidance_scale'] == scale and 'true_cfg_scale' not in kwargs
    assert prepare_teacher_references(adapter, object(), records, tmp_path, [42, 43]) == result
    assert len(calls) == 2
    adapter.teacher_guidance += .5
    with pytest.raises(ValueError, match='differ'):
        prepare_teacher_references(adapter, object(), records, tmp_path, [42, 43])


def test_qwen_recipe_keeps_authorized_optimizer_and_sampling_settings(tmp_path):
    command = training_command(Path('/repo'), tmp_path, Path('/repo/data/repo-t2i'))
    def value(flag): return command[command.index(flag)+1]
    expected = {'--model_family':'qwen_image21', '--model_id':'Qwen/Qwen-Image-2.1',
        '--revision':'d26bb61231c349cf6b7896fa83353113880e1ba3', '--task':'t2i', '--num_steps':'4',
        '--teacher_guidance':'2.5', '--height':'1024', '--width':'1024', '--lora_rank':'64',
        '--lora_alpha':'8', '--fake_lora_rank':'64', '--fake_lora_alpha':'8',
        '--student_lr':'5e-5', '--fake_lr':'5e-5', '--max_grad_norm':'10', '--warmup_step':'500',
        '--batch_size':'1', '--gradient_accumulation_steps':'8', '--fake_gen_ratio':'5',
        '--train_iters':'10000', '--score_noise_sampling':'shifted', '--score_noise_shift':'8',
        '--rollout_mode':'stochastic', '--student_ema_decay':'.98', '--save_every':'500',
        '--keep_checkpoints':'2', '--validation_every':'100',
        '--student_schedule_policy':QWEN_DISTILL_SCHEDULE_POLICY}
    assert all(value(flag)==wanted for flag,wanted in expected.items())
    assert '--resume' not in command
    assert value('--cache_dir') == '/repo/.cache/qwen-image21-t2i-conditioning'


def test_qwen_supervisor_records_child_progress_and_errors(tmp_path):
    import subprocess
    import sys
    from scripts.run_qwen_image21_t2i import supervise_child
    updates=[]
    metric=dict(iteration=11,completed_student_updates=3,seconds_per_iteration=2)
    program="print('TEACHER_REFERENCE 0 42');print('PILOT_BASELINE_READY 16');print("+repr(json.dumps(metric))+")"
    supervise_child([sys.executable,'-c',program],tmp_path/'log',lambda s,**fields:updates.append((s,fields)),'training')
    assert all(stage=='training' for stage,_ in updates)
    assert any(row.get('completed_student_updates')==3 and row['estimated_training_seconds_remaining']==19978 for _,row in updates)
    assert any(row.get('training_phase')=='teacher_references' for _,row in updates)
    with pytest.raises(subprocess.CalledProcessError):
        supervise_child([sys.executable,'-c','raise SystemExit(3)'],tmp_path/'error',lambda *a,**kw:None,'training')


def test_final_qwen_evaluator_uses_cfg_and_restores_pipeline(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from main import evaluate_flow_t2i as evaluator
    config=dict(family='qwen_image21',task='t2i',model_id='base',revision='pinned',
        teacher_guidance=2.5,scheduler_config={},height=1024,width=1024,num_steps=4)
    class Pipe:
        transformer=None
        scheduler=FlowMatchEulerDiscreteScheduler()
        def to(self,device):return self
        def set_progress_bar_config(self,**kwargs):pass
        def __call__(self,*,true_cfg_scale,negative_prompt,num_inference_steps,**kwargs):
            assert true_cfg_scale==2.5 and negative_prompt=='' and num_inference_steps==40
            assert 'guidance_scale' not in kwargs
            return SimpleNamespace(images=[Image.new('RGBA',(1024,1024),(0,0,255,255))])
    pipe=Pipe();original_scheduler=pipe.scheduler
    class Adapter:
        def __init__(self,*a,**kw):self.pipe=pipe
        def prepare(self,record):return {}
        def load_transformer(self):return torch.nn.Linear(1,1)
        def decode(self,*a):return Image.new('RGBA',(1024,1024),(255,0,0,255))
        def offload_encoders(self):pass
    monkeypatch.setattr(evaluator,'FlowAdapter',Adapter)
    monkeypatch.setattr(evaluator,'read_export_config',lambda *a:config)
    monkeypatch.setattr(evaluator,'load_export',lambda *a:torch.nn.Linear(1,1))
    monkeypatch.setattr(evaluator,'sample_student',lambda *a:torch.zeros(1,4,8))
    generator=torch.Generator
    monkeypatch.setattr(evaluator.torch,'Generator',lambda **kw:generator())
    monkeypatch.setattr(evaluator.torch,'autocast',lambda *a,**kw:nullcontext())
    manifest=tmp_path/'validation.jsonl'
    manifest.write_text(json.dumps(dict(task='t2i',prompt='A blue teapot',reference_images=[]))+'\n')
    output=tmp_path/'evaluation'
    evaluator.main(['--checkpoint','unused','--manifest',str(manifest),'--output_dir',str(output)])
    result=json.loads((output/'results.json').read_text())[0]
    assert result['teacher_steps']==40 and result['student_steps']==4
    assert pipe.transformer is None and pipe.scheduler is original_scheduler
    with Image.open(output/'000-student.png') as image:
        assert image.size==(1024,1024) and image.mode=='RGBA'


@pytest.mark.parametrize('family,size,steps', [('flux_klein', 512, 50), ('qwen_image21', 1024, 40)])
def test_audit_checks_full_resolution_and_resized_panels(tmp_path, monkeypatch, family, size, steps):
    from main.audit_flow_validation import audit_pilot
    run = tmp_path/'run'; run.mkdir()
    train = tmp_path/'train.jsonl'; validation = tmp_path/'validation.jsonl'
    train.write_text(json.dumps(dict(task='t2i', prompt='training', reference_images=[]))+'\n')
    validation.write_text(json.dumps(dict(task='t2i', prompt='held out', reference_images=[]))+'\n')
    records = read_manifest(validation, 't2i', size, size)
    identity = dict(format_version=1, family=family, model_id='base', revision='pinned',
        diffusers_version='test', teacher_guidance=2.5, dtype='torch.bfloat16')
    config = dict(manifest=str(train), validation_manifest=str(validation), task='t2i', height=size, width=size,
        validation_limit=1, validation_every=100, validation_seeds=[42], num_steps=4, train_iters=1,
        fake_gen_ratio=5, lora_rank=64, fake_lora_rank=64, student_lr=5e-5, fake_lr=5e-5,
        model_id='base', revision='pinned', teacher_guidance=2.5, seed=42,
        data_keys=[cache_key(r,identity) for r in read_manifest(train,'t2i',size,size)])
    checkpoint = run/'checkpoint-000001'
    for name in ['student','fake']:
        (checkpoint/name).mkdir(parents=True)
        (checkpoint/name/'flow_config.json').write_text(json.dumps(dict(identity,lora_config={'r':64})))
    (checkpoint/'training_config.json').write_text(json.dumps(config))
    torch.save({},checkpoint/'random_states_0.pkl')
    monkeypatch.setattr('main.audit_flow_validation.audit_checkpoint',lambda *a:dict(completed_student_updates=1))
    pixels=np.random.default_rng(42).integers(40,230,(size,size,3),dtype=np.uint8)
    image=Image.fromarray(pixels)
    teacher_dir=run/'teacher-references';teacher_dir.mkdir()
    teacher=teacher_dir/'000-seed42.png';image.save(teacher)
    teacher_rows=[dict(index=0,seed=42,prompt='held out',teacher_steps=steps,image=str(teacher.resolve()))]
    (teacher_dir/'summary.json').write_text(json.dumps(dict(completed=True,results=teacher_rows,
        identity=dict(model_id='base',revision='pinned',teacher_guidance=2.5,teacher_steps=steps))))
    class Adapter:
        device=torch.device('cpu')
        def decode(self,*args):return image
        def offload_encoders(self):pass
    adapter=Adapter();adapter.family=family
    monkeypatch.setattr('main.flow.losses.sample_student',lambda *a:torch.zeros(1,4,8))
    model=torch.nn.Linear(1,1)
    for step in [0,1]:
        evaluate_student(adapter,model,[{}],records,run/f'validation-{step:06d}',[42],teacher_references=teacher_rows)
    (run/'pilot_summary.json').write_text(json.dumps(dict(completed_iterations=1,completed_student_updates=1,
        reached_requested_updates=True,stopped_for_artifacts=False,validation_policy='report_only')))
    assert audit_pilot(run)['artifacts_verified']
    bad=run/'validation-000001/000-seed42.png';Image.new('RGB',(size//2,size//2)).save(bad)
    with pytest.raises(ValueError,match='dimensions'):audit_pilot(run)
