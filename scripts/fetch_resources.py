#!/usr/bin/env python3
"""Download all manifest-pinned files into the configured workspace and verify SHA256.

No extraction, model derivation, package installation, Git clone, or device I/O.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'src/dexstation/examples/ar5_g20/data/sources/external_resources.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(item, target):
    """Keep existing verified files; publish new bytes only after verification."""
    target = Path(target)
    if target.is_file() and sha256(target) == item['sha256']:
        return 'verified'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
        url = item['url']
        for attempt in range(2):
            request = urllib.request.Request(url, headers={'User-Agent': 'dexstation-resource-fetcher'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output)
            with temporary.open('rb') as stream:
                prefix = stream.read(512)
            # raw GitHub may serve an LFS pointer. Fetch the payload, never save the pointer as a mesh.
            if attempt == 0 and prefix.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
                source = urllib.parse.urlsplit(url)
                if source.netloc == 'raw.githubusercontent.com':
                    url = 'https://media.githubusercontent.com/media' + source.path
                    continue
            break
        actual = sha256(temporary)
        if actual != item['sha256']:
            raise ValueError(f"SHA256 mismatch for {target}: expected {item['sha256']}, got {actual}")
        os.replace(temporary, target)
        return 'downloaded'
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch(manifest, workspace):
    total = 0
    for resource in json.loads(Path(manifest).read_text())['resources']:
        for item in resource.get('files', []):
            target = Path(workspace) / resource['destination'] / item['path']
            status = download(item, target)
            print(f'{status}: {target}')
            total += 1
    print(f'Verified {total} pinned files under {workspace}')
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--workspace', type=Path, help='workspace:// root; default: configured workspace_root()')
    args = parser.parse_args()
    if args.workspace is None:
        from dexstation.config.site import workspace_root
        args.workspace = workspace_root()
    fetch(args.manifest, args.workspace.expanduser())


if __name__ == '__main__':
    main()
