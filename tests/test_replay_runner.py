import json
from pathlib import Path


def test_failed_first_job_stops_pair_and_records_failure(tmp_path, monkeypatch):
    import replay_runner
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        return False
    monkeypatch.setattr(replay_runner, 'run_experiment', fail)
    result = replay_runner.run_pair(tmp_path, tmp_path / 'artifacts', gpu='0', wandb_mode='disabled')
    assert not result['matched']
    assert len(calls) == 1
    assert calls[0]['epochs'] == 35
    assert calls[0]['strict_reproducibility'] and calls[0]['replay_diagnostics']
    assert json.loads((tmp_path / 'artifacts' / 'replay-comparison.json').read_text()) == result


def test_completed_pair_requires_matching_diagnostics(tmp_path, monkeypatch):
    import replay_runner
    calls = []
    def complete(**kwargs):
        from test_reproducibility import replay_fixture
        name = kwargs['run_id']
        folder = tmp_path / name
        folder.mkdir()
        rows = replay_fixture(folder / 'replay', epochs=35)
        rows[24]['model_sha256'] = ('e' if len(calls) else 'f') * 64
        (folder / 'replay' / 'replay.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        kwargs['manifest']['experiments'][name] = {'tracking_dir': str(folder), 'wandb_url': 'example'}
        calls.append(name)
        return True
    monkeypatch.setattr(replay_runner, 'run_experiment', complete)
    result = replay_runner.run_pair(tmp_path, tmp_path / 'artifacts', gpu='0', wandb_mode='disabled')
    assert len(calls) == 2
    assert not result['matched']
    assert result['failures'][0]['epoch'] == 23
