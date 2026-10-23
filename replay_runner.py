"""Two short, independent DB15K-20 strict replays; never starts a full campaign."""

import argparse
import json
from pathlib import Path
from threading import Lock

from kaggle_runner import EXPERIMENTS, run_experiment, find_missing_files, discover_gpu_ids
from reproducibility import compare_replays


def run_pair(repo_root, artifacts_dir, *, gpu, epochs=35, wandb_mode='online',
             wandb_entity='diencongle', wandb_project='humea-reproduction'):
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if (artifacts_dir / 'manifest.json').exists():
        raise FileExistsError('Use a fresh artifacts directory for each replay pair')
    manifest = {'experiments': {}}
    result = {'matched': False, 'epochs': epochs, 'failures': [], 'runs': []}
    try:
        for index in (1, 2):
            name = f'diagnostic-db15k20-repeat{index}'
            success = run_experiment(
                repo_root=Path(repo_root), artifacts_dir=artifacts_dir,
                experiment=EXPERIMENTS['db15k-20'], gpu_id=gpu,
                manifest=manifest, manifest_lock=Lock(), run_id=name, force=True,
                epochs=epochs, strict_reproducibility=True, replay_diagnostics=True,
                wandb_mode=wandb_mode, wandb_entity=wandb_entity, wandb_project=wandb_project,
            )
            record = manifest['experiments'].get(name, {})
            result['runs'].append({'id': name, 'wandb_url': record.get('wandb_url')})
            if not success:
                result['failures'].append({'run': name, 'reason': 'training or tracking failed; inspect raw log'})
                break
        else:
            folders = [Path(manifest['experiments'][run['id']]['tracking_dir']) / 'replay'
                       for run in result['runs']]
            result.update(compare_replays(*folders, epochs=epochs))
    except Exception as error:
        result['failures'].append({'reason': f'{type(error).__name__}: {error}'})
    finally:
        (artifacts_dir / 'replay-comparison.json').write_text(
            json.dumps(result, indent=2), encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--artifacts-dir', type=Path, default=Path('artifacts/strict-replay'))
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    parser.add_argument('--wandb-entity', default='diencongle')
    parser.add_argument('--wandb-project', default='humea-reproduction')
    args = parser.parse_args(argv)
    if args.epochs < 26:
        parser.error('Use at least 26 epochs to cover the observed epoch-23 divergence')
    root = args.repo_root.resolve()
    missing = find_missing_files(root, [EXPERIMENTS['db15k-20']])
    if missing:
        parser.error('Missing required data files: ' + ', '.join(map(str, missing)))
    if args.gpu not in discover_gpu_ids():
        parser.error('Requested CUDA GPU is not visible')
    artifacts = args.artifacts_dir if args.artifacts_dir.is_absolute() else root / args.artifacts_dir
    result = run_pair(root, artifacts, gpu=args.gpu, epochs=args.epochs,
                      wandb_mode=args.wandb_mode, wandb_entity=args.wandb_entity,
                      wandb_project=args.wandb_project)
    print(json.dumps(result, indent=2))
    return 0 if result['matched'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
