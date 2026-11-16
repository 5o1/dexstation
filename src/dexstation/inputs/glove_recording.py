"""Lossless glove recording helpers, separate from collection episode schemas."""
from dexstation.contracts.versions import GLOVE_RECORDING_FORMAT
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from uuid import uuid4

PROTOCOL = [('open', 8), ('fist', 8)]+[(f'curl_{f}', 8) for f in ('thumb','index','middle','ring','pinky')]+[('spread', 8)]+[(f'pinch_{f}', 8) for f in ('index','middle','ring','pinky')]+[('free', 12)]
FINGER_NAMES = dict(thumb='thumb', index='index finger', middle='middle finger', ring='ring finger', pinky='little finger')
PROMPTS = dict(open='Open: straighten and spread all five fingers, then hold',
               fist='Fist: slowly close the hand, then open it; repeat',
               spread='Spread: keep the four fingers straight, slowly spread and close them; repeat',
               free='Free: move your fingers freely')
PROMPTS.update({f'curl_{f}': f'{f.capitalize()} curl: bend only the {n} fully, then straighten it; repeat'
                for f, n in FINGER_NAMES.items()})
PROMPTS.update({f'pinch_{f}': f'Pinch {f}: touch the thumb tip to the {n} tip, hold 1 s, release; repeat'
                for f, n in FINGER_NAMES.items() if f != 'thumb'})


def accepts_side(side, hand):
    """Keep only the guided hand's glove when recording one hand at a time."""
    return hand is None or str(side).strip().lower() == hand


def protocol_segment(elapsed, protocol=PROTOCOL):
    start = 0
    for label, duration in protocol:
        if elapsed < start+duration:
            return label
        start += duration
    return None


def json_value(value):
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)) or hasattr(value, 'tolist'):
        return [json_value(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return list(value)
    return value


class GloveRecording:
    def __init__(self, root, label, *, metadata=None):
        if not re.fullmatch(r'[\w-]{1,100}', label):
            raise ValueError('label must contain 1-100 letters, digits, underscore or hyphen')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        self.path = Path(root).expanduser()/f'{stamp}_{label}'
        self.path.mkdir(parents=True)
        self.stream = (self.path/'messages.jsonl').open('x')
        self.meta = dict(format=GLOVE_RECORDING_FORMAT, label=label, domain=88,
            topics=['/manus_glove_0','/manus_glove_1'], message_type='manus_ros2_msgs/msg/ManusGlove',
            started_wall_ns=time.time_ns(), started_monotonic_ns=time.monotonic_ns(),
            source_timestamp_note='Current ManusGlove has no header. DDS source time is transport time, not glove capture time.',
            complete=False, count=0, **(metadata or {}))
        self._save_meta()

    def _save_meta(self):
        (self.path/'metadata.json').write_text(json.dumps(self.meta, ensure_ascii=False, indent=2)+'\n')

    def append(self, topic, cdr, message, segment, *, wall_ns=None, monotonic_ns=None, source_ns=None, received_ns=None):
        if topic not in self.meta['topics']:
            raise ValueError('unexpected glove topic')
        row = dict(sequence=self.meta['count'], topic=topic, segment=segment,
            received_wall_ns=wall_ns if wall_ns is not None else time.time_ns(),
            received_monotonic_ns=monotonic_ns if monotonic_ns is not None else time.monotonic_ns(),
            dds_source_timestamp_ns=source_ns, dds_received_timestamp_ns=received_ns,
            cdr_base64=base64.b64encode(cdr).decode('ascii'), message=json_value(message))
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        self.meta['count'] += 1

    def close(self, *, complete=True):
        self.stream.flush()
        self.stream.close()
        self.meta.update(complete=bool(complete), stopped_wall_ns=time.time_ns())
        self._save_meta()


def read_glove_records(path):
    path = Path(path)
    with (path/'messages.jsonl' if path.is_dir() else path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['topic'] not in ('/manus_glove_0','/manus_glove_1'):
                raise ValueError('unexpected recorded glove topic')
            yield row


def nodes_from_message(message):
    from types import SimpleNamespace as N
    return [N(node_id=n['node_id'], parent_node_id=n['parent_node_id'],
              chain_type=n['chain_type'], joint_type=n['joint_type'],
              pose=N(position=N(**n['pose']['position']), orientation=N(**n['pose']['orientation'])))
            for n in message['raw_nodes']]


def guide_status(phase='idle', elapsed=0, *, hand=None, remaining=None):
    """One protocol clock for terminal and web, including all action text."""
    step, prompt, following, seconds = 0, '', PROMPTS[PROTOCOL[0][0]], 0
    if phase == 'recording':
        start = 0
        for i, (name, duration) in enumerate(PROTOCOL):
            if elapsed < start+duration:
                step, prompt, seconds = i+1, PROMPTS[name], max(0, start+duration-elapsed)
                following = PROMPTS[PROTOCOL[i+1][0]] if i+1 < len(PROTOCOL) else None
                break
            start += duration
    elif phase == 'completed':
        step, following = len(PROTOCOL), None
    return dict(phase=phase, hand=hand, step=step, total_steps=len(PROTOCOL), prompt=prompt,
                remaining_s=seconds if remaining is None else remaining, next_prompt=following,
                count=0, output_dir=None, error=None)


class GloveRecordingConflict(RuntimeError):
    pass


class WebGloveRecording:
    """Own one CLI child. Never launch, stop or publish to an input chain."""
    def __init__(self, output_root, *, command_factory=None):
        self.output_root = Path(output_root).expanduser()
        self.command_factory = command_factory or self.command
        self.task = self.process = None
        self.value = guide_status()
        self.cancel_requested = False

    @property
    def active(self):
        return self.task is not None and not self.task.done()

    def status(self):
        return dict(self.value)

    def command(self, hand, label):
        raise GloveRecordingConflict('Glove recording is unavailable; configure a recording command service')

    def start(self, hand, label, session_dir, session_alive):
        import asyncio
        if self.active:
            raise GloveRecordingConflict('A glove recording is already running')
        if hand not in ('left', 'right') or not isinstance(label, str) or not re.fullmatch(r'[\w-]{1,94}', label):
            raise ValueError('hand must be left/right; label must contain 1-94 letters, digits, underscore or hyphen')
        self.cancel_requested = False
        self.value = dict(guide_status('countdown', hand=hand, remaining=5), id=uuid4().hex, label=label)
        self.task = asyncio.create_task(self._run(hand, label, Path(session_dir), session_alive))
        return self.status()

    async def cancel(self):
        self.cancel_requested = True
        if self.active:
            await self.task
        return self.status()

    async def _run(self, hand, label, session_dir, session_alive):
        import asyncio
        import signal
        from dexstation.runtime.logs import CappedProcessLog
        error = None
        terminal = None
        self.process = None
        log = CappedProcessLog(session_dir/f'glove-recording-{self.value["id"]}.log')
        try:
            argv, env = self.command_factory(hand, label)
            self.process = await asyncio.create_subprocess_exec(*argv, env=env,
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=log,
                start_new_session=True, limit=1024*1024)
            self.value['pid'] = self.process.pid
            line_task = asyncio.create_task(self.process.stdout.readline())
            signalled_at = None
            while True:
                if not session_alive():
                    self.cancel_requested = True
                if self.cancel_requested and self.process.returncode is None:
                    if signalled_at is None:
                        self.process.send_signal(signal.SIGINT)
                        signalled_at = time.monotonic()
                    elif time.monotonic()-signalled_at > 5:
                        self.process.kill()
                done, _ = await asyncio.wait([line_task], timeout=.1)
                if not done:
                    continue
                line = line_task.result()
                if not line:
                    break
                if line.startswith(b'GLOVE_PROGRESS '):
                    row = json.loads(line[len(b'GLOVE_PROGRESS '):])
                    self.value.update(row)
                    terminal = row['phase'] if row['phase'] in ('completed', 'failed', 'cancelled') else terminal
                line_task = asyncio.create_task(self.process.stdout.readline())
            code = await self.process.wait()
            if code and terminal != 'failed' and not self.cancel_requested:
                error = f'Recording process exited ({code}); see the recording log'
            elif terminal is None and not self.cancel_requested:
                error = 'Recording process ended without reporting completion'
        except Exception as exc:
            error = str(exc)
        finally:
            if self.process and self.process.returncode is None:
                self.process.kill()
                await self.process.wait()
            log.close()
            phase = 'cancelled' if self.cancel_requested else 'failed' if error else terminal or 'failed'
            self.value.update(phase=phase, remaining_s=0, error=error or self.value.get('error'), finished_wall_ns=time.time_ns())
            # A forced child exit still preserves its raw file and marks metadata
            # incomplete; the normal SIGINT path finalizes count in the CLI.
            if self.value.get('output_dir') and phase in ('cancelled', 'failed'):
                path = Path(self.value['output_dir'])/'metadata.json'
                meta = json.loads(path.read_text())
                meta['complete'] = False
                path.write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
            with (session_dir/'events.jsonl').open('a') as journal:
                journal.write(json.dumps(dict(action='glove-recording/finished', wall_time_ns=time.time_ns(),
                    recording=self.status()), ensure_ascii=False)+'\n')
