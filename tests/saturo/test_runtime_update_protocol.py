"""Publication and ingress contracts independent of GPU timing/model accuracy."""
import asyncio
from collections import deque
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from saturo.cli.global_scheduler import parse_args
from saturo.runtime.updates import RuntimeUpdatesMixin
from saturo.runtime.worker import RayEngineWorker
from saturo.runtime.workload import SaturoRequest, RT


def make_worker(role='decode', *, p_enabled=True, d_enabled=True):
    worker = RayEngineWorker.__new__(RayEngineWorker)
    worker._shutting_down = False
    worker.worker_id = 1
    worker.worker_role = role
    worker._role_args = {}
    worker._events = []
    worker._pool_layout_generation = 0
    worker._history_version = 0
    worker._prepared_history = None
    worker.applied = []
    provider = SimpleNamespace(set_history_statistics=lambda v: worker.applied.append(copy.deepcopy(v)),
                               reset_history_observations=lambda: worker.applied.append('reset'))
    worker.native = SimpleNamespace(scheduler_config=SimpleNamespace(
        p_pool_be_chunk_ms=100, enable_p_pool_saturation=p_enabled),
        decode_saturation=SimpleNamespace(provider=provider, enabled=d_enabled))
    worker.engine = SimpleNamespace(engine=SimpleNamespace(scheduler=[worker.native]))
    return worker


def command(kind='history_prepare', version=1, epoch=0):
    return dict(cmd=kind, version=version, pool_layout_generation=epoch,
                p_chunk_ms=75, statistics=dict(arrival_rate=2, mean_prompt_len=16,
                mean_output_len=12, completed_requests=8, valid_until_s=time.time()+100))


class WorkerHistoryProtocolTests(unittest.TestCase):
    def test_prepare_does_not_apply_and_commit_does(self):
        w=make_worker(); c=command()
        self.assertTrue(w._history_control(c)['accepted']); self.assertEqual(w.applied, [])
        self.assertTrue(w._history_control(command('history_commit'))['accepted'])
        self.assertEqual(w.applied,[c['statistics']]); self.assertEqual(w._history_version,1)

    def test_duplicate_commit_is_idempotent(self):
        w=make_worker();w._history_control(command());w._history_control(command('history_commit'))
        self.assertTrue(w._history_control(command('history_commit'))['accepted'])
        self.assertEqual(len(w.applied),1)

    def test_old_or_unprepared_commit_rejected(self):
        w=make_worker(); self.assertFalse(w._history_control(command('history_commit'))['accepted'])
        w._history_control(command(version=2))
        self.assertFalse(w._history_control(command(version=1))['accepted'])
        self.assertFalse(w._history_control(command('history_commit',version=1))['accepted'])

    def test_layout_invalidates_prepared_bundle(self):
        w=make_worker();w._history_control(command())
        w._history_control(dict(cmd='update_epoch',pool_layout_generation=1))
        self.assertIsNone(w._prepared_history)
        self.assertFalse(w._history_control(command('history_commit'))['accepted'])
        self.assertEqual(w.applied,['reset'])

    def test_stale_layout_cannot_reset_current_worker(self):
        w=make_worker();w._history_control(dict(cmd='update_epoch',pool_layout_generation=2))
        self.assertFalse(w._history_control(dict(cmd='update_epoch',pool_layout_generation=1))['accepted'])
        self.assertEqual(w._pool_layout_generation,2)

    def test_p_budget_applies_only_at_commit(self):
        w=make_worker('prefill');w._history_control(command())
        self.assertEqual(w.native.scheduler_config.p_pool_be_chunk_ms,100)
        w._history_control(command('history_commit'))
        self.assertEqual(w.native.scheduler_config.p_pool_be_chunk_ms,75)

    def test_invalid_p_budget_is_not_staged(self):
        for value in (-1,0,float('nan'),float('inf'),None,'invalid'):
            w=make_worker('prefill');c=command();c['p_chunk_ms']=value
            self.assertFalse(w._history_control(c)['accepted']);self.assertIsNone(w._prepared_history)

    def test_enabled_P_missing_budget_is_rejected(self):
        w = make_worker('prefill')
        c = command()
        c.pop('p_chunk_ms')
        self.assertFalse(w._history_control(c)['accepted'])
        self.assertIsNone(w._prepared_history)

    def test_disabled_P_commits_only_version_without_touching_budget(self):
        w = make_worker('prefill', p_enabled=False)
        w._role_args['p_pool_be_chunk_ms'] = 100
        c = command()
        c.pop('p_chunk_ms')
        self.assertTrue(w._history_control(c)['accepted'])
        self.assertEqual(w.native.scheduler_config.p_pool_be_chunk_ms, 100)
        self.assertTrue(w._history_control(command('history_commit'))['accepted'])
        self.assertEqual(w._history_version, 1)
        self.assertEqual(w.native.scheduler_config.p_pool_be_chunk_ms, 100)
        self.assertEqual(w._role_args['p_pool_be_chunk_ms'], 100)

    def test_disabled_P_never_validates_or_applies_unused_budget(self):
        w = make_worker('prefill', p_enabled=False)
        c = command()
        c['p_chunk_ms'] = float('nan')
        self.assertTrue(w._history_control(c)['accepted'])
        self.assertTrue(w._history_control(command('history_commit'))['accepted'])
        self.assertEqual(w.native.scheduler_config.p_pool_be_chunk_ms, 100)

    def test_P_budget_only_mutates_enabled_native_PP_stages(self):
        w = make_worker('prefill', p_enabled=False)
        enabled = SimpleNamespace(scheduler_config=SimpleNamespace(
            enable_p_pool_saturation=True, p_pool_be_chunk_ms=120))
        w.engine.engine.scheduler.append(enabled)
        self.assertTrue(w._history_control(command())['accepted'])
        self.assertTrue(w._history_control(command('history_commit'))['accepted'])
        self.assertEqual(w.native.scheduler_config.p_pool_be_chunk_ms, 100)
        self.assertEqual(enabled.scheduler_config.p_pool_be_chunk_ms, 75)

    def test_P_reenabled_after_version_only_prepare_requires_budget(self):
        w = make_worker('prefill', p_enabled=False)
        c = command()
        c.pop('p_chunk_ms')
        self.assertTrue(w._history_control(c)['accepted'])
        w.native.scheduler_config.enable_p_pool_saturation = True
        self.assertFalse(w._history_control(command('history_commit'))['accepted'])
        self.assertEqual(w.native.scheduler_config.p_pool_be_chunk_ms, 100)
        self.assertEqual(w._history_version, 0)

    def test_expired_statistics_rejected(self):
        w=make_worker();c=command();c['statistics']['valid_until_s']=time.time()-1
        self.assertFalse(w._history_control(c)['accepted'])

    def test_incomplete_or_missing_completion_statistics_rejected(self):
        for value in (None, 0):
            w=make_worker();c=command()
            if value is None:c['statistics'].pop('completed_requests')
            else:c['statistics']['completed_requests']=value
            self.assertFalse(w._history_control(c)['accepted'])

    def test_history_expiring_after_prepare_is_not_committed(self):
        w=make_worker();c=command();c['statistics']['valid_until_s']=110
        with patch('saturo.runtime.worker.time.time',return_value=100):
            self.assertTrue(w._history_control(c)['accepted'])
        with patch('saturo.runtime.worker.time.time',return_value=111):
            self.assertFalse(w._history_control(command('history_commit'))['accepted'])
        self.assertEqual(w.applied,[])

    def test_ingress_deadline_caps_relative_deadline(self):
        w=make_worker('prefill');w.rt_prefill_deadline_s=3600
        w._use_dummy_kv=False;w.SamplingParams=lambda **kw:SimpleNamespace(**kw)
        with patch('saturo.runtime.worker.time.time',return_value=100):
            _,p,_=w._request_inputs(dict(request_id='r',kind='rt',phase='prefill',
                max_tokens=1,prompt_len=2,absolute_prefill_deadline_s=101))
        self.assertEqual(p.prefill_ddl,101)


class Controller(RuntimeUpdatesMixin):
    def __init__(self, path):
        self.output_dir=Path(path);self.args=SimpleNamespace(saturation_update_timeout_s=2)
        self._pool_layout_generation=0;self._history_rollout=None;self._epoch_pending={}
        self.history_updates=0;self._p_pool_be_chunk_ms_map={'tp1_pp1':100}
        self.workers=[SimpleNamespace(worker_id=i,ready=True,role=role,
             config_name='tp1_pp1',pending_commands=[]) for i,role in enumerate(('prefill','decode'))]
        self.bundle=dict(version=1,pool_layout_generation=0,history_end_s=100,
            p_budgets={'tp1_pp1':{'chunk_ms':75}},decode_statistics=command()['statistics'])
        self.history_controller=SimpleNamespace(poll=lambda *a,**kw:copy.deepcopy(self.bundle),
                                               request_refresh=lambda:None)
    def _live_workers(self):return self.workers
    def _history_worker_specs(self):return []
    def _elapsed(self):return 1
    def _deployment_pauses_new_work(self):return False
    def _deployment_needs_workers(self):return False
    def ack(self,index,operation,accepted=True):
        c=dict(cmd=operation,version=1,pool_layout_generation=0)
        r=dict(accepted=accepted,version=1,pool_layout_generation=0)
        self._acknowledge_runtime_controls(self.workers[index],[c],[r])


class PublicationTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.c=Controller(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_all_prepare_and_commit_acks_required(self):
        c=self.c;c._poll_history_update_protocol();c.ack(0,'history_prepare')
        self.assertEqual(c._history_rollout['phase'],'preparing')
        c.ack(1,'history_prepare');self.assertEqual(c._history_rollout['phase'],'committing')
        c.ack(0,'history_commit');self.assertEqual(c.history_updates,0)
        c.ack(1,'history_commit');self.assertEqual(c.history_updates,1)
        self.assertEqual(c._p_pool_be_chunk_ms_map['tp1_pp1'],75)

    def test_rejected_prepare_retains_previous_controls(self):
        c=self.c;c._poll_history_update_protocol();c.ack(0,'history_prepare',False)
        self.assertIsNone(c._history_rollout);self.assertEqual(c.history_updates,0)
        self.assertEqual(c._p_pool_be_chunk_ms_map['tp1_pp1'],100)

    def test_failed_commit_invalidates_layout(self):
        c=self.c;c._poll_history_update_protocol();c.ack(0,'history_prepare');c.ack(1,'history_prepare')
        c.ack(1,'history_commit',False)
        self.assertEqual(c._pool_layout_generation,1);self.assertEqual(len(c._epoch_pending),2)

    def test_old_bundle_and_ack_ignored_after_layout_change(self):
        c=self.c;c._poll_history_update_protocol();c._advance_layout_generation('test')
        c.ack(0,'history_prepare');self.assertIsNone(c._history_rollout)
        self.assertEqual(c.history_updates,0)

    def test_disabled_P_has_version_only_participation_with_D_on_or_off(self):
        for d_enabled in (False, True):
            with self.subTest(d_enabled=d_enabled):
                c = Controller(self.tmp.name)
                c.args.enable_prefill_saturation = False
                c.args.enable_decode_saturation = d_enabled
                c.bundle['p_budgets'] = {}
                actors = [make_worker('prefill', p_enabled=False),
                          make_worker('decode', d_enabled=d_enabled)]
                c._poll_history_update_protocol()
                self.assertEqual(c._history_rollout['phase'], 'preparing')
                self.assertNotIn('p_chunk_ms', c.workers[0].pending_commands[0])
                for phase in ('history_prepare', 'history_commit'):
                    for worker, actor in zip(c.workers, actors):
                        commands = worker.pending_commands
                        worker.pending_commands = []
                        self.assertEqual([cmd['cmd'] for cmd in commands], [phase])
                        replies = [actor._history_control(cmd) for cmd in commands]
                        self.assertTrue(all(reply['accepted'] for reply in replies))
                        c._acknowledge_runtime_controls(worker, commands, replies)
                self.assertIsNone(c._history_rollout)
                self.assertEqual(c.history_updates, 1)
                self.assertEqual([actor._history_version for actor in actors], [1, 1])
                self.assertEqual(actors[0].native.scheduler_config.p_pool_be_chunk_ms, 100)
                self.assertEqual(actors[1].applied, [c.bundle['decode_statistics']])
                self.assertEqual(c._p_pool_be_chunk_ms_map, {'tp1_pp1': 100})
                self.assertEqual(actors[1].native.decode_saturation.enabled, d_enabled)

    def test_enabled_P_missing_budget_rejects_entire_bundle(self):
        c = self.c
        c.args.enable_prefill_saturation = True
        c.bundle['p_budgets'] = {}
        c._poll_history_update_protocol()
        self.assertIsNone(c._history_rollout)
        self.assertEqual(c.history_updates, 0)
        self.assertTrue(all(not worker.pending_commands for worker in c.workers))


class CommonSettingsTests(unittest.TestCase):
    def test_periodic_allocation_preserves_explicit_native_role_mode(self):
        from saturo.runtime.launcher import WorkerLauncherMixin
        from saturo.runtime.worker import WorkerPlacement
        for adaptive, periodic in ((False, True), (True, True), (False, False)):
            with patch.dict('os.environ', {}, clear=True):
                args = parse_args(['--num-req', '2', '--target-rps', '1',
                    '--adaptive-role-switching' if adaptive else '--no-adaptive-role-switching',
                    '--periodic-reallocation' if periodic else '--no-periodic-reallocation'])
            launcher = WorkerLauncherMixin()
            launcher.args = args
            launcher._resolve_p_pool_be_chunk_ms = lambda config: 100
            result = launcher._worker_args_dict(
                WorkerPlacement('decode', 'local', '', config_name='tp1_pp1'), 1)
            self.assertEqual(result['adaptive_role_switching'], adaptive)

    def test_shared_defaults_and_horizons(self):
        with patch.dict('os.environ',{},clear=True):
            a=parse_args(['--num-req','2','--target-rps','1'])
        self.assertEqual(a.saturation_update_interval_s,40)
        self.assertEqual(a.decode_saturation_lookahead_s,40)
        self.assertEqual(a.resource_decode_lookahead_window_s,40)
        self.assertEqual(a.rt_prefill_deadline_s,2)

    def test_delta_is_an_actual_interval_and_explicit_cli_wins_env(self):
        with patch.dict('os.environ',{'SATURO_HISTORY_REFRESH_S':'50'}):
            a=parse_args(['--num-req','2','--target-rps','1','--saturation-update-interval-s','3'])
        self.assertEqual(a.saturation_update_interval_s,3)
        self.assertEqual(a.decode_saturation_lookahead_s,3)

    def test_ingress_buffered_during_deployment_without_timestamp_shift(self):
        from saturo.runtime.scheduler import SaturoGlobalScheduler
        s=SaturoGlobalScheduler.__new__(SaturoGlobalScheduler)
        s.all_rt=[SaturoRequest('r1',RT,4,2,.5),SaturoRequest('r2',RT,4,2,1)]
        s.next_rt_idx=0;s.pending_prefill=deque();s.trace_recorder=None;s.role_switcher=None
        s._observed_runtime_arrivals=set();s._source_arrival_epoch_s=100;s._elapsed=lambda:1.2
        s._deployment_pauses_new_work=lambda:True
        seen=[];s.history_controller=SimpleNamespace(record_arrival=lambda *x:seen.append(x))
        s._inject_due_requests()
        self.assertEqual([r.arrival_offset_s for r in s.pending_prefill],[.5,1])
        self.assertEqual([row[2] for row in seen],[100.5,101])


if __name__ == '__main__':unittest.main()
