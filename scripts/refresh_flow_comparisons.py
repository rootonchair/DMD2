"""Refresh a live run's saved comparison panels without changing training."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from main.flow.comparisons import save_comparison


def refresh(run):
    reference_file = run / 'teacher-references/summary.json'
    if not reference_file.exists():
        return 0
    references = json.loads(reference_file.read_text())
    if not references['completed']:
        return 0
    teachers = {(row['index'], row['seed']): row for row in references['results']}
    count = 0
    for path in sorted(run.glob('validation-*/summary.json')):
        destination = path.parent / 'comparison_metadata.json'
        if destination.exists():
            continue
        validation = json.loads(path.read_text())
        if not validation['completed']:
            continue
        rows = []
        for row in validation['results']:
            teacher = teachers[row['index'], row['seed']]
            if teacher['prompt'] != row['prompt']:
                raise ValueError('Teacher/student prompt mismatch')
            with Image.open(teacher['image']) as reference, Image.open(row['image']) as student:
                layout = save_comparison(path.parent / f'{row["index"]:03d}-seed{row["seed"]}-comparison.png',
                    [('Teacher, 50 steps', reference), ('Student, 4 steps', student)],
                    row['prompt'], row['seed'], int(path.parent.name.split('-')[-1]))
            layout.update(index=row['index'], teacher_image=teacher['image'], teacher_steps=50)
            rows.append(layout)
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(completed=True, results=rows), indent=2) + '\n')
        temporary.replace(destination)
        count += len(rows)
        print('REFRESHED', path.parent.name, len(rows), flush=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    while True:
        refresh(run)
        if not args.watch:
            return
        state = json.loads((run / 'run_status.json').read_text())
        if state['stage'] in ['complete', 'failed', 'superseded', 'stopped_for_artifacts']:
            refresh(run)
            return
        try:
            os.kill(state['pid'], 0)
        except ProcessLookupError:
            refresh(run)
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
