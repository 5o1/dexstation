"""Recorder lifecycle builders with inert simulation metadata."""
from pathlib import Path

from dexstation.recording.session import Session


def session_for(output, *, operator_notes=()):
    """Exercise the production recorder without creating a simulator or devices."""
    session = Session.__new__(Session)
    session.output = Path(output)
    session.output.mkdir(parents=True, exist_ok=True)
    session.meta = dict(scene_spec_text='{}', fps=50, operator_notes=list(operator_notes))
    session.writer = None
    session.last_sim_time_s = 0.
    session.episode_index = 0
    session.pending_notes = []
    session.control_log = (session.output/'events.jsonl').open('a')
    session.recording_state = 'idle'
    session.last_saved = None
    session.on_status = lambda: None
    session.commands = dict.fromkeys(('left_arm', 'right_arm', 'left_hand', 'right_hand'))
    session.input_sequences = {}
    session.save_pool = session.save_future = None
    return session
