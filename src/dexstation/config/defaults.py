"""Code defaults for checkout-based collection and private site configuration."""

DEFAULT_WORKBENCH = 'ar5_08_g20_bench_bimanual'
DEFAULT_TASK = 'empty'
def __getattr__(name):
    # Resolve after discovery so long-lived clients see external registrations.
    if name in ('COLLECTION_SOURCES', 'API_COLLECTION_SOURCES'):
        from dexstation.inputs.registry import source_names
        return source_names(api=name == 'API_COLLECTION_SOURCES')
    raise AttributeError(name)


# Generic defaults do not select physical inputs or station services.
SITE_DEFAULTS = {'web': {'pane_weights': {}},
 'workspace': '',
 'conda_executable': 'conda',
 'simulation': {'python': '', 'resources': ''},
 'compute': {'ssh_profile': '',
             'remote_code': '',
             'python': '',
             'cuda_visible_devices': '',
             'render_device': ''},
 'udp': {'bind_host': '127.0.0.1',
         'peer_host': '127.0.0.1',
         'source_host': '',
         'port': 50200,
         'allow_sources': []},
 'operator': {'bind_host': '127.0.0.1',
              'port': 8780,
              'browser_url': 'http://127.0.0.1:8780/',
              'debug_url': 'http://127.0.0.1:19222'},
 'network': {'proxy_url': ''},
 'snapshots': {'delay_s': 30,
               'width': 1280,
               'height': 720,
               'max_bytes': 67108864},
 'collection': {'motion_profile': 'auto',
                'backend': 'sapien',
                'source': 'udp',
                'scene_variant': 'full',
                'duration': None,
                'rate_hz': 60,
                'loop_replay': False,
                'replay': '',
                'replay_files': [],
                'ros_topic_prefix': '',
                'output_root': '',
                'export_python': ''}}
