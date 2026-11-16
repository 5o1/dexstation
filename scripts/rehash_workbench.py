#!/usr/bin/env python3
"""Refresh only workbench manifest hash values, preserving JSON formatting."""
import argparse
import hashlib
import json
from pathlib import Path

from dexstation.profiles.compose import profile_digest, _reference


def value_spans(text):
    """Locate dictionary values by JSON path without serializing the document."""
    decoder = json.JSONDecoder()
    spans = {}

    def whitespace(index):
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    def walk(index, path):
        index = whitespace(index)
        start = index
        if text[index] == '{':
            index = whitespace(index + 1)
            while text[index] != '}':
                key, index = decoder.raw_decode(text, index)
                index = whitespace(index)
                index = walk(index + 1, (*path, key))
                index = whitespace(index)
                if text[index] == ',':
                    index = whitespace(index + 1)
            index += 1
        else:
            _, index = decoder.raw_decode(text, index)
        spans[path] = (start, index)
        return index

    walk(0, ())
    return spans


def rehash(directory, *, check=False):
    directory = Path(directory).resolve()
    path = directory / 'workbench.json'
    with path.open(newline='') as stream:
        text = stream.read()
    profile = json.loads(text)
    updates = {}
    for name, component in profile['manifest']['components'].items():
        source = _reference(component['name'], directory)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if component['sha256'] != digest:
            updates[('manifest', 'components', name, 'sha256')] = digest
            component['sha256'] = digest
    digest = profile_digest(profile, directory)
    if profile['manifest']['sha256'] != digest:
        updates[('manifest', 'sha256')] = digest
    if updates:
        print('Stale fields: ' + ', '.join('.'.join(key) for key in updates))
        if not check:
            spans = value_spans(text)
            for key in sorted(updates, key=lambda key: spans[key][0], reverse=True):
                start, end = spans[key]
                text = text[:start] + json.dumps(updates[key]) + text[end:]
            with path.open('w', newline='') as stream:
                stream.write(text)
    return 1 if check and updates else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workbench', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    return rehash(args.workbench, check=args.check)


if __name__ == '__main__':
    raise SystemExit(main())
