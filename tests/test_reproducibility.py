import copy
import json
import os
import random

import numpy as np
import pytest
import torch

from test_training_tracking import make_training


def api():
    import reproducibility
    return reproducibility


def replay_fixture(folder, epochs=1):
    folder.mkdir()
    (folder / 'inputs.json').write_text(json.dumps({
        'files': {'x': {'sha256': 'a' * 64, 'bytes': 1}},
        'train_split_sha256': 'b' * 64, 'test_split_sha256': 'c' * 64,
        'source_sha256': 'd' * 64, 'config': {'strict_reproducibility': True, 'check_point': 10}}))
    (folder / 'settings.json').write_text(json.dumps({
        'deterministic_algorithms': True, 'warn_only': False,
        'cublas_workspace_config': ':4096:8'}))
    rows = []
    for e in range(-1, epochs):
        rows.append({'epoch': e, 'model_sha256': 'a' * 64, 'rng_sha256': 'b' * 64,
                     'train_pairs_sha256': 'c' * 64, 'train_sequence_sha256': 'd' * 64,
                     'routing': {'encoder': {'joint': {'logits': [0] * 7,
                         'probabilities': [1 / 7] * 7, 'selected': [0, 1, 2, 3, 4],
                         'cutoff_margin': 0}}, 'estimator': {'joint': {}}} if e >= 0 else {},
                     'losses': dict.fromkeys(['total', 'mi', 'inner', 'alignment', 'joint', 'estimator'], 0) if e >= 0 else {},
                     'evaluation': {'best_mrr': 0, 'best_result': None, 'current': None}})
    for row in rows[1:]:
        row['routing']['estimator'] = copy.deepcopy(row['routing']['encoder'])
        if row['epoch'] > 0 and row['epoch'] % 10 == 0:
            row['evaluation']['current'] = {'epoch': row['epoch'], **dict.fromkeys(
                ['hits1', 'hits5', 'hits10', 'mr', 'mrr'], 0)}
            row['routing']['evaluation'] = copy.deepcopy(row['routing']['encoder'])
    return rows


def test_strict_controls_are_explicit_and_do_not_reseed(monkeypatch):
    r = api()
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    before = r.state_hash(r.capture_rng())
    try:
        settings = r.configure_determinism(True)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
        assert settings['deterministic_algorithms'] is True
        assert before == r.state_hash(r.capture_rng())
    finally:
        torch.use_deterministic_algorithms(False)


def test_hashes_include_content_shape_and_dtype_and_preserve_rng():
    r = api()
    before = r.state_hash(r.capture_rng())
    x = torch.arange(4, dtype=torch.float32)
    assert r.state_hash(x) == r.state_hash(x.clone())
    assert r.state_hash(x) != r.state_hash(x.reshape(2, 2))
    assert r.state_hash(x) != r.state_hash(x.double())
    assert r.state_hash({'b': x, 'a': 1}) == r.state_hash({'a': 1, 'b': x})
    assert before == r.state_hash(r.capture_rng())


def test_input_manifest_changes_when_file_changes(tmp_path):
    r = api()
    folder = tmp_path / 'dataset'
    folder.mkdir()
    (folder / 'x').write_bytes(b'abc')
    a = r.input_manifest(folder)
    (folder / 'x').write_bytes(b'abd')
    assert a != r.input_manifest(folder)


def test_diagnostics_preserve_training_and_rng_and_restore_checkpoint(tmp_path):
    r = api()
    reference = make_training(None)
    observed = make_training(None)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    initial_rng = r.capture_rng()
    reference.train()
    final_rng = r.state_hash(r.capture_rng())
    r.restore_rng(initial_rng)
    observed.diagnostics = r.ReplayDiagnostics(tmp_path, observed, checkpoint_epochs={0, 1})
    observed.train()
    assert r.state_hash(r.capture_rng()) == final_rng
    assert r.state_hash(observed.optimizer.state_dict()) == r.state_hash(reference.optimizer.state_dict())
    assert r.state_hash(observed.multimodal_encoder.state_dict()) == r.state_hash(reference.multimodal_encoder.state_dict())
    expected = copy.deepcopy(observed.multimodal_encoder.state_dict())
    expected_rng = r.state_hash(r.capture_rng())
    observed.multimodal_encoder.weight.data.fill_(100)
    torch.rand(3); np.random.rand(2); random.random()
    next_epoch, evaluation = r.restore_checkpoint(tmp_path / 'checkpoint-0001.pt', observed)
    assert next_epoch == 2
    assert r.state_hash(observed.multimodal_encoder.state_dict()) == r.state_hash(expected)
    assert r.state_hash(r.capture_rng()) == expected_rng
    records = [json.loads(line) for line in (tmp_path / 'replay.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in records] == [-1, 0, 1]
    assert 'train_sequence_sha256' in records[1]


def test_replay_gate_rejects_missing_epochs_and_changed_state(tmp_path):
    r = api()
    first, second = tmp_path / 'a', tmp_path / 'b'
    rows = replay_fixture(first)
    replay_fixture(second)
    def write(p, rows):
        (p / 'replay.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    write(first, rows); write(second, rows)
    assert r.compare_replays(first, second, epochs=1)['matched']
    write(second, rows[:1])
    assert not r.compare_replays(first, second, epochs=1)['matched']
    write(second, [rows[0], {**rows[1], 'model_sha256': 'e' * 64}])
    assert not r.compare_replays(first, second, epochs=1)['matched']
    incomplete = [{k: v for k, v in row.items() if k != 'rng_sha256'} for row in rows]
    write(first, incomplete); write(second, incomplete)
    assert not r.compare_replays(first, second, epochs=1)['matched']


def test_resume_continues_with_identical_updates_and_rng(tmp_path):
    r = api()
    def make_adam():
        model = make_training(None)
        model.optimizer = torch.optim.AdamW(model.multimodal_encoder.parameters(), lr=0.001)
        model.mi_optimizer = torch.optim.AdamW(model.mi_estimator.parameters(), lr=0.001)
        return model
    uninterrupted = make_adam()
    uninterrupted.args.epochs = 3
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    initial = r.capture_rng()
    uninterrupted.train()
    expected = r.state_hash(r.training_state(uninterrupted))
    expected_rng = r.state_hash(r.capture_rng())
    interrupted = make_adam()
    interrupted.args.epochs = 1
    r.restore_rng(initial)
    interrupted.diagnostics = r.ReplayDiagnostics(tmp_path / 'before', interrupted, checkpoint_epochs={0})
    interrupted.train()
    resumed = make_adam()
    resumed.args.epochs = 3
    resumed.args.resume_diagnostic = tmp_path / 'before' / 'checkpoint-0000.pt'
    resumed.diagnostics = r.ReplayDiagnostics(tmp_path / 'after', resumed)
    resumed.train()
    assert r.state_hash(r.training_state(resumed)) == expected
    assert r.state_hash(r.capture_rng()) == expected_rng


def test_diagnostics_record_each_evaluation_not_only_best(tmp_path):
    r = api()
    model = make_training(None)
    model.args.epochs = 3
    model.args.check_point = 1
    model.test = lambda epoch: setattr(model, 'latest_evaluation', {'mrr': 1.0 / epoch})
    model.diagnostics = r.ReplayDiagnostics(tmp_path, model)
    model.train()
    rows = [json.loads(x) for x in (tmp_path / 'replay.jsonl').read_text().splitlines()]
    assert rows[2]['evaluation']['current']['mrr'] == 1.0
    assert rows[3]['evaluation']['current']['mrr'] == 0.5
    assert 'estimator' in rows[1]['routing']


def test_actual_fusion_trace_does_not_change_output_or_gradients():
    from model import PowerSetFusionWithMoEGating
    from types import SimpleNamespace
    r = api()
    a = PowerSetFusionWithMoEGating(SimpleNamespace(fusion_weight_dim=512, topk=5))
    b = copy.deepcopy(a)
    b.record_routing = True
    inputs = [torch.randn(4, 8) for _ in range(6)]
    before = r.state_hash(r.capture_rng())
    out_a, out_b = a(inputs), b(inputs)
    assert r.state_hash(r.capture_rng()) == before
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
    out_a.square().sum().backward(); out_b.square().sum().backward()
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    assert len(b.routing_snapshot['logits']) == 7
    assert len(b.routing_snapshot['selected']) == 5
    assert b.routing_snapshot['cutoff_margin'] >= 0
