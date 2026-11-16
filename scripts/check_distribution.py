#!/usr/bin/env python3
"""Check that a built wheel carries the operator UI and recovery entry point."""
import argparse
from pathlib import Path
import zipfile


def check(path):
    with zipfile.ZipFile(path) as archive:
        if any("site.local.json" in name or name.startswith(".local/") for name in archive.namelist()):
            raise ValueError("private configuration or archive is present")
        for filename in ("index.html", "app.js"):
            archive.getinfo(f"dexstation/render/web/{filename}")
        for filename in ('server/web/index.html','server/web/overlay/collection.css',
                         'server/web/overlay/fonts/IBMPlexSansCondensed-Regular.woff2',
                         'server/web/overlay/fonts/JetBrainsMono-Regular.woff2'):
            archive.getinfo('dexstation/'+filename)
        entry = next(name for name in archive.namelist() if name.endswith("/entry_points.txt"))
        for filename in ('station.json','provenance.json','source_ledger.json','vendor/provenance.json'):
            archive.getinfo('dexstation/examples/ar5_g20/motion_assets/'+filename)
        obsolete = ('station_math.hpp', 'station_math.cpp', 'STATION_SOURCE_LICENSE',
                    'sdk_filter_reference.json', 'vendor/ar5_meta.yaml', 'vendor/ar5_right.mjcf',
                    'vendor/l25_meta.yaml', 'vendor/l25_right.mjcf', 'vendor/protocol_excerpt.json')
        names = set(archive.namelist())
        for filename in obsolete:
            if 'dexstation/examples/ar5_g20/motion_assets/' + filename in names:
                raise ValueError('obsolete motion asset is present: ' + filename)
        for filename in ('g20_left.urdf', 'g20_right.urdf', 'motor_mapping.json'):
            archive.getinfo('dexstation/inputs/retarget_assets/' + filename)
        entries = archive.read(entry)
        for command in (b"dexstation-server", b"dexstation-recover"):
            if command not in entries:
                raise ValueError(f"{command.decode()} CLI is missing")
    print(f"distribution OK: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    check(parser.parse_args().wheel)
