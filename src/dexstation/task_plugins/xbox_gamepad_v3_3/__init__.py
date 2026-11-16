"""Xbox gamepad task hooks; importing this package does not load physics assets."""
API_VERSION = 1
ACTIONS = {}


def cell_state(task):
    return {'signals': task.get('channels', {})}


def task_sets(data_directory, source=None):
    import json
    from pathlib import Path
    root = Path(data_directory)
    profile = json.loads((root / 'task.json').read_text())
    return sorted(str(path) for declaration in profile['sources']
        if source is None or declaration['id'] == source
        for path in root.glob(declaration.get('task_sets', '')) if path.is_file())
