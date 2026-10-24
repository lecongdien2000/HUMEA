"""Restore the immutable source snapshot used by the recorded reproduction."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

SOURCE_SHA256 = '545b95884f15dfc94ceae0d985cbe0968da5cc733f718ab611be9688e2e51b34'
REFERENCE_RUN_ID = 'zifzbhun'


def prepare_reference(destination, *, bundle=None):
    """Verify source bytes, then extract into a new directory without changing files."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Use a new source directory; refusing to overwrite {destination}')
    bundle = Path(bundle) if bundle else Path(__file__).resolve().parent / 'reproduction' / 'reference-source.zip'
    content = bundle.read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError('Reference source checksum mismatch; refusing extraction')
    with ZipFile(io.BytesIO(content)) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in member.filename or ':' in member.filename:
                raise ValueError('Unsafe reference archive path')
        destination.mkdir(parents=True, exist_ok=False)
        archive.extractall(destination)
    return {'reference_run_id': REFERENCE_RUN_ID, 'source_sha256': SOURCE_SHA256,
            'training_source': str(destination.resolve())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare_reference(args.destination), indent=2))


if __name__ == '__main__':
    main()
