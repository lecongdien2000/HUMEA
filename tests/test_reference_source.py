import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_restored_reference_matches_original_tracking_fingerprint(tmp_path):
    from reference_source import prepare_reference, SOURCE_SHA256
    from experiment_tracking import source_fingerprint
    destination = tmp_path / 'source'
    record = prepare_reference(destination)
    assert record['source_sha256'] == SOURCE_SHA256
    assert source_fingerprint(destination) == SOURCE_SHA256
    assert not (destination / 'reproducibility.py').exists()
    assert not (destination / 'replay_runner.py').exists()
    assert record['reference_run_id'] == 'zifzbhun'
    with pytest.raises(FileExistsError):
        prepare_reference(destination)


def test_corrupt_reference_rejected_before_writing_destination(tmp_path):
    from reference_source import prepare_reference
    bundle = tmp_path / 'corrupt.zip'
    bundle.write_bytes(b'invalid archive')
    with pytest.raises(ValueError, match='checksum'):
        prepare_reference(tmp_path / 'source', bundle=bundle)
    assert not (tmp_path / 'source').exists()


def test_reference_manifest_has_portable_provenance_and_file_checksums():
    from reference_source import SOURCE_SHA256
    from zipfile import ZipFile
    manifest = json.loads((ROOT / 'reproduction' / 'reference-source.json').read_text())
    assert manifest['source_sha256'] == SOURCE_SHA256
    assert manifest['reference_run_id'] == 'zifzbhun'
    with ZipFile(ROOT / 'reproduction' / 'reference-source.zip') as archive:
        assert set(manifest['files']) == set(archive.namelist())
        for name, digest in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest


def test_reference_notebook_uses_external_runner_and_baseline_plus_loss_removals():
    notebook = json.loads((ROOT / 'kaggle' / 'humea_reference_ablations.ipynb').read_text())
    sources = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
    for source in sources:
        compile(source, '<reference-ablation-notebook>', 'exec')
    combined = '\n'.join(sources)
    assert 'reference_source.py' in combined
    assert 'str(TOOL_ROOT / "kaggle_runner.py")' in combined
    assert '"--repo-root", str(WORK_ROOT)' in combined
    assert 'for without in (0, 7, 8)' in combined
    assert '--strict_reproducibility' not in combined
    assert '--replay_diagnostics' not in combined
    assert 'source_sha256' in combined
    assert 'os.killpg' in combined


def test_archived_commands_match_recorded_config_and_remove_only_selected_loss(tmp_path):
    from reference_source import prepare_reference
    from kaggle_runner import build_train_command, EXPERIMENTS
    destination = tmp_path / 'source'
    prepare_reference(destination)
    manifest = json.loads((ROOT / 'reproduction' / 'reference-source.json').read_text())
    commands = [build_train_command(destination, EXPERIMENTS['db15k-20'], without=n)[2:]
                for n in (0, 7, 8)]
    script = '''
import sys, json, argparse, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path('tests').resolve()))
from train import HUMEA
from test_training_tracking import make_training, RecordingTracker
payload=json.loads(sys.stdin.read())
for n, argv in zip((0,7,8), payload['commands']):
    sys.argv=['train.py'] + argv
    options=vars(HUMEA.parse_options(argparse.ArgumentParser()))
    expected={**payload['config'], 'without': n}
    assert options==expected, (options,expected)
    tracker=RecordingTracker()
    model=make_training(tracker)
    model.args.without=n
    before=model.multimodal_encoder.weight.detach().clone()
    def forbidden(*args):
        raise AssertionError('Removed objective executed')
    if n==7: model.inner_view_loss=forbidden
    if n==8: model.kl_alignment_loss=forbidden
    np.random.seed(42)
    model.train()
    assert len(tracker.events)==2
    assert not __import__('torch').equal(before,model.multimodal_encoder.weight.detach())
    if n==7: assert all(metrics['inner']==0 and metrics['alignment']>0 for _,metrics in tracker.events)
    if n==8: assert all(metrics['alignment']==0 and metrics['inner']>0 for _,metrics in tracker.events)
print('Archived baseline/7/8 configurations and objective execution verified')
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=destination,
                            input=json.dumps({'commands': commands, 'config': manifest['training_config']}),
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
