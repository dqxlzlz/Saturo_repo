import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from saturo.runtime import worker as module
from saturo.runtime.workload import DECODE, PREFILL, TPUT


@pytest.fixture
def observed_gc(monkeypatch):
    calls = []
    monkeypatch.delenv('SATURO_STATIC_DECODE_GC_FREEZE', raising=False)
    monkeypatch.setattr(module.gc, 'isenabled', lambda: True)
    monkeypatch.setattr(module.gc, 'collect', lambda gen=2: calls.append(('collect', gen)))
    monkeypatch.setattr(module.gc, 'freeze', lambda: calls.append(('freeze',)))
    monkeypatch.setattr(module.gc, 'get_freeze_count', lambda: 1234)
    return calls


def worker(role=DECODE, adaptive=False, saturation=True):
    # Existing Saturo tests likewise bypass the constructor; no test runner
    # should perform a real process-wide freeze through a fake engine.
    obj = module.RayEngineWorker.__new__(module.RayEngineWorker)
    obj.worker_role = role
    obj._role_switching_enabled = adaptive
    obj._decode_role_initialized = role == DECODE and saturation
    return obj


def test_static_decode_collects_then_freezes_once(observed_gc):
    obj = worker()
    thresholds = gc.get_threshold()
    obj._initialize_static_decode_gc()
    obj._initialize_static_decode_gc()
    assert observed_gc == [('collect', 2), ('freeze',)]
    assert obj._static_decode_gc_frozen
    assert obj._static_decode_gc_frozen_count == 1234
    assert gc.isenabled() and gc.get_threshold() == thresholds


@pytest.mark.parametrize('role,adaptive,saturation', [
    (PREFILL, False, True), (TPUT, False, True),
    (DECODE, True, True), (DECODE, False, False),
])
def test_other_roles_and_saturation_ablation_do_not_freeze(
        observed_gc, role, adaptive, saturation):
    obj = worker(role, adaptive, saturation)
    obj._initialize_static_decode_gc()
    assert not obj._static_decode_gc_frozen and not observed_gc


@pytest.mark.parametrize('value', ['0', 'false', 'no', 'off', ' FALSE '])
def test_operator_optout(observed_gc, monkeypatch, value):
    monkeypatch.setenv('SATURO_STATIC_DECODE_GC_FREEZE', value)
    obj = worker()
    obj._initialize_static_decode_gc()
    assert not obj._static_decode_gc_frozen and not observed_gc


def test_disabled_gc_is_respected(observed_gc, monkeypatch):
    monkeypatch.setattr(module.gc, 'isenabled', lambda: False)
    obj = worker()
    obj._initialize_static_decode_gc()
    assert not obj._static_decode_gc_frozen and not observed_gc


def test_constructor_runs_gc_after_engine_and_request_state_setup(monkeypatch):
    calls = []
    engine = SimpleNamespace(engine=SimpleNamespace())
    monkeypatch.setattr(module, '_configure_worker_environment', lambda _: None)

    def build(args):
        calls.append('engine_and_predictor_ready')
        return engine, object, False

    def initialize(obj):
        assert obj.engine is engine and not obj.active and not obj._request_tasks
        assert obj._decode_role_initialized and not obj._role_switching_enabled
        calls.append('gc_before_actor_ready')

    monkeypatch.setattr(module, '_build_async_engine', build)
    monkeypatch.setattr(module, '_validate_p_pool_budget', lambda *_: None)
    monkeypatch.setattr(module.RayEngineWorker, '_initialize_static_decode_gc', initialize)
    module.RayEngineWorker(dict(worker_id=1, worker_role=DECODE,
                                model='fake', rt_prefill_deadline_s=2,
                                enable_decode_saturation=True))
    assert calls == ['engine_and_predictor_ready', 'gc_before_actor_ready']


def test_new_request_cycles_still_collect_in_isolated_subprocess():
    # Load this exact module path so staged validation also exercises the
    # staged helper; only the child interpreter may really freeze objects.
    source = str(Path(module.__file__).resolve())
    code = '''
import gc, importlib.util, os, sys, weakref
spec = importlib.util.spec_from_file_location("saturo.runtime.worker", SOURCE)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
os.environ.pop("SATURO_STATIC_DECODE_GC_FREEZE", None)
before = (gc.isenabled(), gc.get_threshold())
obj = m.RayEngineWorker.__new__(m.RayEngineWorker)
obj.worker_role = m.DECODE
obj._role_switching_enabled = False
obj._decode_role_initialized = True
obj._initialize_static_decode_gc()
assert obj._static_decode_gc_frozen and gc.get_freeze_count() > 0
class RequestCycle:
    pass
request = RequestCycle()
request.self = request
reference = weakref.ref(request)
del request
assert reference() is not None
assert gc.collect(2) > 0 and reference() is None
assert before == (gc.isenabled(), gc.get_threshold())
print("new request cycles collected; GC enabled and thresholds preserved")
'''.replace('SOURCE', json.dumps(source))
    env = dict(os.environ)
    env.pop('SATURO_GC_TIMING_DIR', None)
    result = subprocess.run([sys.executable, '-c', code], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'new request cycles collected' in result.stdout
