"""The public input example selects owned anatomy and injected preflight only."""
import pytest

from dexstation.inputs.supervisor import InputChain, RETARGET_METHODS, retarget_command
from dexstation.inputs.glove_recording import WebGloveRecording, GloveRecordingConflict


def test_public_input_catalog_and_anatomical_command():
    assert RETARGET_METHODS == ('manus_anatomical',)
    command = retarget_command({'python': '/configured/python'})
    assert command[:5] == ['/configured/python', '-m', 'dexstation.inputs.retarget_node',
                           '--method', 'manus_anatomical']
    with pytest.raises(ValueError, match='invalid retarget'):
        retarget_command({'python': '/configured/python', 'retarget': 'manus_dex'})


def test_public_input_uses_injected_non_mutating_can_service(tmp_path):
    class Can:
        def require(self, role):
            assert role == 'linkerta'
            return {'ready': True, 'interface': 'synthetic_can'}
    chain = InputChain(log_root=tmp_path, can_system=Can())
    assert chain.check_can() == {'ready': True, 'interface': 'synthetic_can'}


def test_public_glove_job_requires_an_explicit_command_service(tmp_path):
    recorder = WebGloveRecording(tmp_path)
    with pytest.raises(GloveRecordingConflict, match='unavailable'):
        recorder.command('left', 'synthetic')


def test_public_composition_shares_its_injected_preflight_service(tmp_path):
    from dexstation.config.service_defaults import public_services
    class Can:
        def require(self, role):
            return {'role': role, 'ready': True}
    can = Can()
    services = public_services(can_system=can)
    chain = services.input_chain_factory(log_root=tmp_path)
    assert chain.can_system is services.can_system is can
    assert chain.check_can() == {'role': 'linkerta', 'ready': True}
