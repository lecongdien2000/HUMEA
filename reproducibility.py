"""Opt-in strict replay controls. Diagnostics never sample random numbers."""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def deterministic_settings():
    return {
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
        'torch': str(torch.__version__),
        'cuda': torch.version.cuda,
    }


def configure_determinism(enabled):
    if enabled:
        workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        if workspace not in (None, ':4096:8', ':16:8'):
            raise RuntimeError('Strict replay requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8')
        if workspace is None and torch.cuda.is_initialized():
            raise RuntimeError('Configure strict replay before CUDA initialization')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
    return deterministic_settings()


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def state_hash(value):
    """Content hash independent of tensor storage identities and dictionary order."""
    digest = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu()
            if tensor.layout == torch.sparse_coo:
                tensor = tensor.coalesce()
                add(('sparse', tuple(tensor.shape), tensor.indices(), tensor.values()))
            else:
                add(('tensor', str(tensor.dtype), tuple(tensor.shape)))
                digest.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            add(('numpy', str(item.dtype), item.shape))
            digest.update(item.tobytes(order='C'))
        elif isinstance(item, dict):
            digest.update(b'dict{')
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                add(key); add(item[key])
            digest.update(b'}')
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b'[')
            for child in item:
                add(child)
            digest.update(b']')
        elif item is None or isinstance(item, (str, bytes, int, float, bool, np.generic)):
            data = (type(item).__name__ + ':' + repr(item)).encode()
            digest.update(str(len(data)).encode() + b':' + data)
        else:
            raise TypeError(f'Unsupported replay hash type: {type(item)}')
    add(value)
    return digest.hexdigest()


def input_manifest(directory, extra_file=None):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    paths = {p.relative_to(directory).as_posix(): p for p in directory.rglob('*') if p.is_file()}
    if extra_file:
        paths['external_train_pairs'] = Path(extra_file)
    result = {}
    for name, path in sorted(paths.items()):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        result[name] = {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
    return result


_MODULES = ('multimodal_encoder', 'mi_estimator', 'multi_loss_layer', 'align_multi_loss_layer')
_OPTIMIZERS = ('optimizer', 'mi_optimizer')


def training_state(model):
    return {name: getattr(model, name).state_dict() for name in (*_MODULES, *_OPTIMIZERS)}


def experiment_config(model):
    return {k: v for k, v in vars(model.args).items()
            if k not in ('epochs', 'resume_diagnostic', 'replay_diagnostics')}


class ReplayDiagnostics:
    def __init__(self, output_dir, model, checkpoint_epochs=(20, 22, 23, 24)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_epochs = set(checkpoint_epochs)
        folder = getattr(model.args, 'file_dir', None)
        self.inputs = {
            'files': input_manifest(folder, getattr(model.args, 'train_ill_path', None)) if folder else {},
            'train_split_sha256': state_hash(model.train_ill),
            'test_split_sha256': state_hash(getattr(model, 'test_ill', None)),
            'config': experiment_config(model),
            'source_sha256': getattr(getattr(model, 'tracker', None), 'source_sha256', None),
        }
        self._json('inputs.json', self.inputs)
        self._json('settings.json', deterministic_settings())
        (self.output_dir / 'replay.jsonl').write_text('', encoding='utf-8')
        for module in model.multimodal_encoder.modules():
            if hasattr(module, 'combo_masks'):
                module.record_routing = True

    def _json(self, name, value):
        (self.output_dir / name).write_text(json.dumps(value, indent=2), encoding='utf-8')

    def routing(self, model):
        return {name: module.routing_snapshot
                for name, module in model.multimodal_encoder.named_modules()
                if hasattr(module, 'routing_snapshot')}

    def record(self, epoch, model, *, routing=None, losses=None, evaluation=None):
        state = training_state(model)
        rng = capture_rng()
        record = {'epoch': epoch, 'model_sha256': state_hash(state),
                  'rng_sha256': state_hash(rng),
                  'train_pairs_sha256': state_hash(model.train_ill),
                  'train_sequence_sha256': state_hash(getattr(model, 'train_list', None)),
                  'routing': routing or {},
                  'losses': {k: v for k, v in (losses or {}).items()
                             if k not in ('seconds', 'gpu_peak_allocated_gib')},
                  'evaluation': evaluation}
        with (self.output_dir / 'replay.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, default=lambda x: x.tolist()) + '\n')
        if epoch == -1 or epoch in self.checkpoint_epochs:
            payload = {'format_version': 1, 'epoch': epoch, 'state': state, 'rng': rng,
                       'inputs': self.inputs, 'train_ill': model.train_ill,
                       'train_list': getattr(model, 'train_list', None),
                       'train_list_alias': getattr(model, 'train_list', None) is model.train_ill,
                       'evaluation': evaluation}
            path = self.output_dir / f'checkpoint-{epoch:04d}.pt'
            temporary = path.with_suffix('.tmp')
            torch.save(payload, temporary)
            temporary.replace(path)


def restore_checkpoint(path, model):
    """Load only trusted local diagnostic checkpoints (they contain Python RNG data)."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['format_version'] != 1:
        raise ValueError('Unknown diagnostic checkpoint format')
    if payload['inputs']['config'] != experiment_config(model):
        raise ValueError('Checkpoint training configuration does not match')
    diagnostics = getattr(model, 'diagnostics', None)
    if diagnostics and diagnostics.inputs != payload['inputs']:
        raise ValueError('Checkpoint inputs or initial split do not match')
    for name, state in payload['state'].items():
        getattr(model, name).load_state_dict(state)
    model.train_ill = payload['train_ill']
    model.train_list = (model.train_ill if payload['train_list_alias'] else payload['train_list'])
    restore_rng(payload['rng'])
    return payload['epoch'] + 1, payload['evaluation']


def compare_replays(first, second, *, epochs):
    """Exact gate: inputs/settings and all initial/per-epoch records must agree."""
    failures = []
    first, second = Path(first), Path(second)
    inputs = json.loads((first / 'inputs.json').read_text())
    for folder in (first, second):
        data = json.loads((folder / 'inputs.json').read_text())
        required = {'files', 'train_split_sha256', 'test_split_sha256', 'source_sha256', 'config'}
        if (not required <= data.keys() or not data.get('files') or not data.get('source_sha256')
                or data.get('config', {}).get('strict_reproducibility') is not True
                or not isinstance(data.get('config', {}).get('check_point'), int)
                or data['config']['check_point'] <= 0):
            failures.append({'file': 'inputs.json', 'reason': 'missing provenance fields'})
    for filename in ('inputs.json', 'settings.json'):
        if json.loads((first / filename).read_text()) != json.loads((second / filename).read_text()):
            failures.append({'file': filename, 'reason': 'different'})
    for folder in (first, second):
        settings = json.loads((folder / 'settings.json').read_text())
        if (not settings.get('deterministic_algorithms') or settings.get('warn_only') is not False
                or settings.get('cublas_workspace_config') not in (':4096:8', ':16:8')):
            failures.append({'file': 'settings.json', 'reason': 'strict determinism not enabled'})
    def rows(folder):
        return [json.loads(line) for line in (folder / 'replay.jsonl').read_text().splitlines()]
    a, b = rows(first), rows(second)
    expected = list(range(-1, epochs))
    for name, records in (('first', a), ('second', b)):
        if [row['epoch'] for row in records] != expected:
            failures.append({'run': name, 'reason': 'missing, duplicate or unexpected epochs'})
        required = {'epoch', 'model_sha256', 'rng_sha256', 'train_pairs_sha256',
                    'train_sequence_sha256', 'routing', 'losses', 'evaluation'}
        for row in records:
            if not required <= row.keys():
                failures.append({'run': name, 'reason': 'incomplete epoch evidence'})
                break
            hashes = ('model_sha256', 'rng_sha256', 'train_pairs_sha256', 'train_sequence_sha256')
            if any(not isinstance(row[k], str) or len(row[k]) != 64 for k in hashes):
                failures.append({'run': name, 'epoch': row['epoch'], 'reason': 'invalid state hash'})
                break
            if row['epoch'] >= 0:
                loss_keys = {'total', 'mi', 'inner', 'alignment', 'joint', 'estimator'}
                if (not loss_keys <= row['losses'].keys()
                        or not {'encoder', 'estimator'} <= row['routing'].keys()
                        or not all(row['routing'][phase] for phase in ('encoder', 'estimator'))
                        or not isinstance(row['evaluation'], dict)
                        or not {'best_mrr', 'best_result', 'current'} <= row['evaluation'].keys()):
                    failures.append({'run': name, 'epoch': row['epoch'], 'reason': 'incomplete losses, routing or evaluation'})
                    break
                route_keys = {'logits', 'probabilities', 'selected', 'cutoff_margin'}
                if any(not route_keys <= snapshot.keys()
                       for phase in row['routing'].values() for snapshot in phase.values()):
                    failures.append({'run': name, 'epoch': row['epoch'], 'reason': 'incomplete routing snapshot'})
                    break
                interval = inputs.get('config', {}).get('check_point', 0)
                if interval and row['epoch'] > 0 and row['epoch'] % interval == 0:
                    current = row['evaluation']['current']
                    if (not isinstance(current, dict)
                            or not {'epoch', 'hits1', 'hits5', 'hits10', 'mr', 'mrr'} <= current.keys()
                            or not row['routing'].get('evaluation')):
                        failures.append({'run': name, 'epoch': row['epoch'], 'reason': 'missing checkpoint evaluation'})
                        break
    for x, y in zip(a, b):
        if x != y:
            failures.append({'epoch': x['epoch'], 'different_fields': sorted(
                key for key in x.keys() | y.keys() if x.get(key) != y.get(key))})
            break
    return {'matched': not failures, 'epochs': epochs, 'failures': failures}
