"""Encoder shutdown must drain stdout after the normal stream reader stops."""
import asyncio
import sys
from types import SimpleNamespace

from dexstation.render.operator_stream import OperatorStream


def test_close_drains_backpressured_encoder_stdout_without_hanging():
    async def scenario():
        encoder=await asyncio.create_subprocess_exec(sys.executable,'-u','-c',
            'import sys;sys.stdout.buffer.write(b"x"*1048576);sys.stdout.flush();sys.stdin.buffer.read()',
            stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE)
        stream=OperatorStream.__new__(OperatorStream)
        stream.tasks=[];stream.peers={};stream.encoder=encoder;stream.demand=None
        closed=[]
        for key in ('frames','encoder_log','timing','recording'):
            setattr(stream,key,SimpleNamespace(close=lambda k=key:closed.append(k)))
        try:
            # Reach an actual full StreamReader buffer, not just a fake process.
            async def full_pipe():
                while len(encoder.stdout._buffer)<=131072:
                    await asyncio.sleep(.01)
            await asyncio.wait_for(full_pipe(),3)
            await asyncio.wait_for(stream.close(),2)
            assert encoder.returncode==0
            assert set(closed)=={'frames','encoder_log','timing','recording'}
        finally:
            if encoder.returncode is None:encoder.kill()
            await asyncio.wait_for(encoder.communicate(),3)
    asyncio.run(scenario())


def test_closing_stream_rejects_browser_reconnects():
    import pytest
    pytest.importorskip('aiohttp')
    from aiohttp import ClientSession, WSServerHandshakeError, web
    from aiohttp.test_utils import TestServer
    async def scenario():
        stream=OperatorStream.__new__(OperatorStream)
        stream.tasks=[];stream.peers={};stream.encoder=None;stream.demand=None
        stream.closing=False;stream.auxiliary=False
        for key in ('frames','encoder_log','timing','recording'):
            setattr(stream,key,SimpleNamespace(close=lambda:None))
        app=web.Application();app.router.add_get('/video',stream.video)
        server=TestServer(app);await server.start_server()
        try:
            async with ClientSession() as client:
                ws=await client.ws_connect(server.make_url('/video'))
                close=asyncio.create_task(stream.close())
                await asyncio.sleep(0)
                with pytest.raises(WSServerHandshakeError) as rejected:
                    await client.ws_connect(server.make_url('/video'))
                assert rejected.value.status==503
                await ws.receive()
                await asyncio.wait_for(close,1)
                assert not stream.peers
        finally:await server.close()
    asyncio.run(scenario())
