"""Queue ordering and one-step supervisor configuration."""

import json
from pathlib import Path

import pytest

from scripts import run_flux_klein_t2i as supervisor


def test_queue_waits_for_final_evaluation_and_predecessor_exit(tmp_path, monkeypatch):
    phases = [('training', True), ('auditing', True), ('final_evaluation', True),
              ('complete', True), ('complete', False)]
    position = 0
    status_path = tmp_path / 'run_status.json'
    def write():
        status_path.write_text(json.dumps(dict(pid=123, started_at=456,
            stage=phases[position][0], completed_iterations=10000)))
    def advance(seconds):
        nonlocal position
        assert seconds == 30
        position += 1
        write()
    write()
    monkeypatch.setattr(supervisor, 'process_start', lambda pid: 'start' if phases[position][1] else None)
    monkeypatch.setattr(supervisor.time, 'sleep', advance)
    updates = []
    supervisor.wait_for_run(tmp_path, lambda stage, **fields: updates.append((stage, fields)))
    assert position == 4
    assert [fields['predecessor_stage'] for _, fields in updates] == [stage for stage, _ in phases]
    assert all(stage == 'queued' for stage, _ in updates)


@pytest.mark.parametrize('stage,alive', [('failed', True), ('superseded', False), ('training', False)])
def test_queue_does_not_launch_after_failed_or_unfinished_predecessor(stage, alive, tmp_path, monkeypatch):
    (tmp_path / 'run_status.json').write_text(json.dumps(dict(pid=123, started_at=456, stage=stage)))
    monkeypatch.setattr(supervisor, 'process_start', lambda pid: 'start' if alive else None)
    with pytest.raises(RuntimeError, match='Predecessor'):
        supervisor.wait_for_run(tmp_path, lambda *args, **kwargs: None)


def test_one_step_queue_changes_only_step_count_and_output(tmp_path):
    root = Path('/example/repository')
    data = root / 'data/repo-t2i'
    one = supervisor.training_command(root, tmp_path / 'one', data, 1)
    four = supervisor.training_command(root, tmp_path / 'four', data, 4)
    def options(command):
        result, flag = {}, None
        for word in command:
            if word.startswith('--'):
                flag = word
                result[flag] = []
            elif flag:
                result[flag].append(word)
        return result
    one, four = options(one), options(four)
    differences = {key for key in one if one[key] != four[key]}
    assert differences == {'--num_steps', '--output_dir'}
    assert one['--num_steps'] == ['1'] and '--resume' not in one
    assert one['--train_iters'] == ['10000'] and one['--fake_gen_ratio'] == ['5']
    assert one['--student_ema_decay'] == ['.98'] and one['--score_noise_shift'] == ['8']
