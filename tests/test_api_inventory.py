"""Inventory every registered route, including HEAD and static resources."""
import asyncio

from dexstation.server.app import SessionManager, make_app
from helpers import ROOT, assert_golden, golden_scene


def test_http_route_inventory(tmp_path, monkeypatch):
    # Also isolate the site/workspace paths consumed by make_app's composition.
    golden_scene(ROOT / 'src/dexstation/task_plugins/empty/data/task.json', tmp_path, monkeypatch)

    async def inventory():
        manager = SessionManager(tmp_path / 'sessions', default_values={'task': 'empty'})
        app = make_app(manager, managed=False)
        try:
            routes = sorted([route.method, route.resource.canonical] for route in app.router.routes())
            assert_golden('routes.json', routes)
        finally:
            # make_app opens the audit journal before its cleanup context starts.
            # Run that context so its finally block closes the journal/client.
            app.freeze()
            await app.startup()
            await app.cleanup()

    asyncio.run(inventory())
