"""Strict saved-episode validation and non-destructive discard sidecars."""
import json
from pathlib import Path
import time

from dexstation.recording.episode import read_episode, _read_layers, _read_jsonl, _check_rows, _store_json, _json


def check_integrity(path):
    path = Path(path)
    result = dict(episode_id=episode_id(path), state='running', passed=False, checked_wall_time_ns=time.time_ns(), checks={}, errors=[])
    try:
        episode = read_episode(path)
        meta = dict(episode.workbench.meta, **episode.meta)
        rows, events = episode.workbench.rows, episode.inputs
        result['checks']['strict_readback'] = True
        result['checks']['layer_alignment'] = True
        result['rows'] = len(rows)
        if meta['status'] != 'complete' or not rows:
            raise ValueError('episode is incomplete or empty')
        seq = [e['sequence'] for e in events]
        if any(b != a+1 for a, b in zip(seq, seq[1:])):
            raise ValueError('input sequence has a gap')
        received = meta.get('received_input_messages')
        if received is None:
            raise ValueError('no independent receiver boundary count')
        if received != len(events):
            raise ValueError('command journal count differs from receiver count')
        if events and (seq[0] != meta['receive_start_sequence'] or seq[-1]+1 != meta['receive_end_sequence']):
            raise ValueError('input sequence differs from receiver boundaries')
        result['checks']['receiver_count_and_sequences'] = True
        fps = float(meta['fps'])
        start = float(meta['recording_start_sim_time_s'])
        end = float(meta['recording_end_sim_time_s'])
        dt = max(float(meta.get('physics_timestep_s', 0)), 1e-6)
        expected = len(rows)/fps
        if abs(end-start-expected) > dt+1e-6:
            raise ValueError('row count and recorded duration differ')
        if abs(rows[-1]['sim_time_s']-end) > 1e-6 or abs(rows[0]['sim_time_s']-start-1/fps) > dt+1e-6:
            raise ValueError('row time boundaries differ')
        if any(abs(b['sim_time_s']-a['sim_time_s']-1/fps)>dt+1e-6 for a,b in zip(rows,rows[1:])):
            raise ValueError('row clock has a gap')
        result['duration_s'] = end-start
        result['input_messages'] = len(events)
        result['checks']['row_duration'] = True
        result['passed'] = True
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        result['errors'].append(str(exc))
    result['state'] = 'passed' if result['passed'] else 'failed'
    return result


def discard_episode(path, note=''):
    path = Path(path)
    if not (path/'episode.json').is_file():
        raise ValueError('episode not found')
    marker = dict(discarded=True, wall_time_ns=time.time_ns(), note=str(note))
    (path/'discarded.json').write_text(json.dumps(marker, ensure_ascii=False, indent=2)+'\n')
    return marker



def episode_id(path):
    path=Path(path)
    return path.parent.name+'~'+path.name


def pending_integrity(path, phase='saving'):
    return dict(episode_id=episode_id(path),state='pending',phase=phase,passed=None,checks={},errors=[])


def store_integrity(path, result):
    path=Path(path)
    previous = path/'integrity.json'
    if previous.exists():
        recovery = json.loads(previous.read_text()).get('recovery')
        if recovery is not None:
            result = dict(result, recovery=recovery)
    temporary=path/'integrity.json.tmp'
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path/'integrity.json')


def read_integrity(path):
    path=Path(path)
    file=path/'integrity.json'
    if not file.exists():return pending_integrity(path,'awaiting_check')
    result=json.loads(file.read_text())
    # Legacy sidecars remain readable; bind them to their actual directory.
    result.setdefault('episode_id',episode_id(path))
    result.setdefault('state','passed' if result.get('passed') is True else 'failed' if result.get('passed') is False else 'pending')
    return result


def check_and_store(path):
    """CPU worker entrypoint: never imports or constructs a simulator."""
    path=Path(path)
    running=dict(pending_integrity(path,'checking'),state='running')
    store_integrity(path,running)
    result=check_integrity(path)
    store_integrity(path,result)
    return result


def recover_episode(path):
    """Explicitly trim an interrupted recording to its valid common prefix.

    Strict reads never mutate disk. The saved status does not decide which rows
    survive; recovery does not certify an episode as complete.
    """
    path = Path(path)
    manifest, layers = _read_layers(path, prefix=True)
    count = 0
    last = -1.
    for rows in zip(*(layer.rows for layer in layers)):
        candidates = [type(layer)(layer.format, [row]) for layer, row in zip(layers, rows)]
        try:
            _check_rows(candidates, start_step=count, last=last)
        except (ValueError, KeyError, TypeError):
            break
        count += 1
        last = rows[0]['sim_time_s']
    recovery = dict(original_status=manifest['status'], original_rows=manifest['rows'],
                    recovered_rows=count, checked_wall_time_ns=time.time_ns(), journals={})
    for descriptor, layer in zip(manifest['layers'], layers):
        directory = path/descriptor['path']
        for filename, rows in [('steps.jsonl', layer.rows[:count]), ('events.jsonl', layer.events)]:
            file = directory/filename
            original = file.read_bytes()
            content = ''.join(_json(row)+'\n' for row in rows).encode('utf-8')
            recovery['journals'][str(file.relative_to(path))] = dict(original_bytes=len(original),
                recovered_bytes=len(content), retained_rows=len(rows))
            file.write_bytes(content)
    commands = path/'workbench/commands.jsonl'
    inputs = _read_jsonl(commands, prefix=True)
    recovery['journals']['workbench/commands.jsonl'] = dict(original_bytes=commands.stat().st_size,
        recovered_bytes=len(''.join(_json(row)+'\n' for row in inputs).encode('utf-8')), retained_rows=len(inputs))
    commands.write_text(''.join(_json(row)+'\n' for row in inputs), encoding='utf-8')
    workbench = layers[0].meta
    workbench['input_messages'] = len(inputs)
    if count:
        workbench['recording_end_sim_time_s'] = layers[0].rows[count-1]['sim_time_s']
    _store_json(path/'workbench/meta.json', workbench)
    manifest.update(status='interrupted', rows=count)
    _store_json(path/'episode.json', manifest)
    result = dict(episode_id=episode_id(path), state='failed', passed=False,
                  checks={}, errors=['interrupted recording recovered to common prefix'], recovery=recovery)
    store_integrity(path, result)
    return read_episode(path)


def main(argv=None):
    """Recover an interrupted layered episode; the untouched original is kept beside it."""
    import argparse
    import shutil
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('episode', type=Path)
    args = parser.parse_args(argv)
    path = args.episode.resolve()
    if not (path/'episode.json').is_file():
        parser.error(f'not a layered episode: {path}')
    backup = path.with_name(path.name+'.before-recover')
    if backup.exists():
        parser.error(f'backup already exists: {backup}')
    shutil.copytree(path, backup, symlinks=True)
    episode = recover_episode(path)
    print(json.dumps(dict(episode=str(path), original=str(backup), status=episode.meta['status'],
                          rows=episode.meta['rows'])))
    return 0
