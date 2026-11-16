#!/usr/bin/env python3
"""Exercise real web/physics/render/record startup with simulated input readiness.

Run in the station's web/SAPIEN environment. Uses UDP with no sender, never an
input-device chain. The injected early timing recreates the events.jsonl race.
Output directories are retained; existing output is never overwritten.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time

from aiohttp import ClientSession, CookieJar
from aiohttp.test_utils import TestServer

from dexstation.config.defaults import DEFAULT_WORKBENCH
from dexstation.server.app import SessionManager, make_app
from dexstation.config.site import configure_runtime


class StartupProbe(SessionManager):
    async def start(self, values):
        result=await super().start(values)
        self.timing('first_linkerta_data_s',time.monotonic())
        return result


async def validate(output,workbench,task,cycles):
    def no_input_devices():
        raise AssertionError('startup probe must not construct an input-device chain')
    manager=StartupProbe(output/'sessions',default_values=dict(workbench=workbench,task=task),input_chain_factory=no_input_devices)
    server=TestServer(make_app(manager));await server.start_server()
    results=[]
    try:
        async with ClientSession(cookie_jar=CookieJar(unsafe=True)) as browser:
            async with browser.get(server.make_url('/')) as response:
                assert response.status==200
            async def call(path,body=None):
                async with browser.request('GET' if body is None else 'POST',server.make_url('/api/v1/'+path),
                        json=body,headers={'X-Dexstation-Client':'web'}) as response:
                    data=await response.json()
                    if response.status!=200:raise RuntimeError(f'{path}: {response.status}: {data}')
                    return data
            async def until(predicate,timeout=150):
                deadline=time.monotonic()+timeout
                while time.monotonic()<deadline:
                    state=await call('status')
                    if state['session_state']=='failed':raise RuntimeError(state.get('error'))
                    if predicate(state):return state
                    await asyncio.sleep(.02)
                raise TimeoutError(f'Waiting for startup/recording: {state}')
            for cycle in range(cycles):
                await call('session/start',dict(backend='sapien',source='udp',replay=None,
                    workbench=workbench,task=task,duration_s=None))
                state=await until(lambda s:s['session_state']=='running' and
                    'first_frame_s' in s['input_chain']['timings'])
                assert state['motion_profile_resolved']=='station'
                assert state['remote_control_enabled'] is False
                folder=manager.output
                await call('recording/start',{'note':f'startup probe {cycle}'})
                await asyncio.sleep(.2)
                await call('recording/stop',{})
                saved=await until(lambda s:s['recording_status']['state'] in ('saved','failed'),timeout=30)
                assert saved['recording_status']['state']=='saved',saved['recording_status']
                await call('session/stop',{})
                rows=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
                actions=[r['action'] for r in rows]
                names=[r['name'] for r in rows if r['action']=='session/timing']
                assert names.count('first_linkerta_data_s')==1 and names.count('first_frame_s')==1
                assert actions.count('start')==1 and actions.count('stop')==1 and actions.count('note')==1
                exits=json.loads((folder/'view/worker_exit.json').read_text());assert all(v==0 for v in exits.values()),exits
                results.append(dict(cycle=cycle,session=str(folder),journal_actions=actions,
                    timings=state['input_chain']['timings'],recorded_rows=saved['recording_status']['rows'],
                    recording_integrity=saved['recording_status']['integrity'],renderer_exits=exits,passed=True))
                print(json.dumps(results[-1]),flush=True)
    finally:
        await server.close()
    return dict(passed=True,hardware_inputs_started=False,readiness_event='simulated first_linkerta_data_s',cycles=results)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workbench',default=DEFAULT_WORKBENCH)
    p.add_argument('--task',default='xbox_gamepad_v3_3')
    p.add_argument('--cycles',type=int,default=2)
    args=p.parse_args()
    if args.cycles<1:p.error('cycles must be positive')
    configure_runtime();args.output.mkdir(parents=True,exist_ok=False)
    result=asyncio.run(validate(args.output,args.workbench,args.task,args.cycles))
    (args.output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cycles=len(result['cycles']))))


if __name__=='__main__':main()
