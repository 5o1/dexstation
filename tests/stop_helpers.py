"""Inert HTTP service and minimal checkout for the standard-library stop client."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import threading
import pytest
from helpers import ROOT


@pytest.fixture
def stop_checkout(tmp_path):
    checkout = tmp_path/'code'
    for name in ('src/dexstation/__init__.py', 'src/dexstation/config/defaults.py', 'src/dexstation/config/__init__.py',
                 'scripts/collection_ctl.py'):
        target = checkout/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    return checkout


@contextmanager
def api_stub(mode):
    requests=[];release=threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            requests.append(self.path)
            self.rfile.read(int(self.headers.get('Content-Length',0)))
            if mode=='hang':release.wait(10)
            body=json.dumps(dict(state=dict(session_state='stopped'),message='remote control disabled' if mode=='denied' else 'done')).encode()
            try:
                self.send_response(403 if mode=='denied' else 200)
                self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,kwargs=dict(poll_interval=.05),daemon=True)
    thread.start()
    try:yield f'http://127.0.0.1:{server.server_port}',requests
    finally:
        release.set();server.shutdown();server.server_close();thread.join(timeout=2)

