"""Dependency-free validator for the JSON Schema subset used by this project."""

import json
import math
import re


def _json_text(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _validate(value, schema, root=None, label='profile'):
    """Validate the subset used by the project schemas, without extra dependencies."""
    if isinstance(schema, bool):
        if not schema:
            raise ValueError(f'{label}: forbidden field')
        return
    root = schema if root is None else root
    if '$ref' in schema:
        target = root
        for key in schema['$ref'].removeprefix('#/').split('/'):
            target = target[key.replace('~1', '/').replace('~0', '~')]
        _validate(value, target, root, label)
    for union in ('anyOf', 'oneOf'):
        if union in schema:
            matches = 0
            for option in schema[union]:
                try:
                    _validate(value, option, root, label)
                except ValueError:
                    continue
                matches += 1
            if not matches or (union == 'oneOf' and matches != 1):
                raise ValueError(f'{label}: invalid {union}')
    types = {'object': dict, 'array': list, 'string': str, 'boolean': bool,
             'number': (int, float), 'integer': int, 'null': type(None)}
    if 'type' in schema:
        names = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
        if not any(isinstance(value, types[t]) and not (
                t in ('number', 'integer') and isinstance(value, bool)) for t in names):
            raise ValueError(f'{label}: expected {schema["type"]}')
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError(f'{label}: number must be finite')
        if value < schema.get('minimum', -math.inf) or value > schema.get('maximum', math.inf):
            raise ValueError(f'{label}: number outside range')
        if value <= schema.get('exclusiveMinimum', -math.inf):
            raise ValueError(f'{label}: number outside range')
    if 'const' in schema and (value != schema['const'] or type(value) is not type(schema['const'])):
        raise ValueError(f'{label}: invalid constant')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f'{label}: invalid choice')
    if isinstance(value, dict):
        missing = set(schema.get('required', [])) - value.keys()
        if missing:
            raise ValueError(f'{label}: missing {sorted(missing)}')
        for name, child in value.items():
            child_schema = schema.get('properties', {}).get(name, schema.get('additionalProperties', True))
            _validate(child, child_schema, root, f'{label}.{name}')
    if isinstance(value, list):
        if not schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', math.inf):
            raise ValueError(f'{label}: invalid array length')
        if schema.get('uniqueItems') and len({_json_text(v) for v in value}) != len(value):
            raise ValueError(f'{label}: duplicate items')
        for index, child in enumerate(value):
            _validate(child, schema.get('items', True), root, f'{label}[{index}]')
    if isinstance(value, str):
        if len(value) < schema.get('minLength', 0) or not re.search(schema.get('pattern', ''), value):
            raise ValueError(f'{label}: invalid string')


