"""CPU-only contract tests for atomic, observation-only P/D updates."""

import copy
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from saturo.history import HistorySnapshot
from saturo.runtime.history import RuntimeHistoryController


class Clock:
    def __init__(self, wall=1000.0, mono=0.0):
        self.wall = wall
        self.mono = mono

    def advance(self, seconds):
        self.wall += seconds
        self.mono += seconds


SPECS = [
    {"worker_id": "p0", "role": "prefill", "config_name": "tp1_pp1"},
    {"worker_id": "d0", "role": "decode", "config_name": "tp1_pp1"},
    {"worker_id": "d1", "role": "decode", "config_name": "tp1_pp1"},
]


class RuntimeHistoryUpdateTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {"SATURO_HISTORY_SOURCE": "csv"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.clock = Clock()
        self.controllers = []
        self.addCleanup(self.close_all)

    def close_all(self):
        for controller in self.controllers:
            controller.close(timeout=0.1)

    def controller(self, **overrides):
        args = SimpleNamespace(model="m", enable_prefill_saturation=True,
                               saturation_history_window_s=600,
                               saturation_update_interval_s=40,
                               saturation_update_timeout_s=30)
        for key, value in overrides.items():
            setattr(args, key, value)
        controller = RuntimeHistoryController(
            args, SPECS, autostart=False, clock=lambda: self.clock.wall,
            monotonic=lambda: self.clock.mono)
        controller._optimize_p_budget = lambda *a: {"chunk_ms": 80.0}
        self.controllers.append(controller)
        return controller

    def observe(self, controller):
        self.clock.advance(5)
        controller.record_arrival("done", 16, self.clock.wall)
        self.clock.advance(1)
        controller.record_completion("done", 12, self.clock.wall)
        self.clock.advance(4)
        controller.record_arrival("pending", 48, self.clock.wall)
        self.clock.advance(10)

    def test_common_args_override_legacy_env_and_invalid_values_reject(self):
        with patch.dict("os.environ", {"SATURO_HISTORY_REFRESH_S": "99",
                                      "SATURO_RUNTIME_HISTORY_WINDOW_S": "999"}):
            controller = self.controller()
            self.assertEqual((controller.refresh_s, controller.window_s), (40, 600))
            legacy = self.controller(saturation_update_interval_s=None,
                                     saturation_history_window_s=None)
            self.assertEqual((legacy.refresh_s, legacy.window_s), (99, 999))
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.controller(saturation_update_interval_s=value)

    def test_observed_live_window_and_pending_outputs_not_oracle(self):
        controller = self.controller()
        self.observe(controller)
        snapshot = controller.snapshot(window_s=1800)
        self.assertEqual((snapshot.start_s, snapshot.end_s), (1000, 1020))
        self.assertEqual(snapshot.statistics()["arrival_rate"], 2 / 20)
        self.assertEqual(snapshot.statistics()["mean_output_len"], 12)
        self.assertEqual(snapshot.records[1]["output_len"], None)
        self.assertEqual(len(snapshot.workload_rows()), 1)
        self.assertTrue(snapshot.metadata()["provenance"]["cold_start"])
        self.assertEqual(snapshot.metadata()["source"], "live_ingress")
        snapshot.records[0]["output_len"] = 99999
        self.assertEqual(controller.snapshot().records[0]["output_len"], 12)

    def test_snapshot_masks_future_completion_and_future_arrival(self):
        controller = self.controller()
        self.observe(controller)
        controller.record_completion("pending", 99, 1030)
        controller.record_arrival("future", 10, 1031)
        snapshot = controller.snapshot(end_s=1015)
        self.assertEqual(len(snapshot.records), 2)
        self.assertIsNone(snapshot.records[1]["output_len"])
        with self.assertRaises(ValueError):
            controller.snapshot(end_s=1035)

    def test_ingress_idempotence_and_conflicts(self):
        controller = self.controller()
        self.assertTrue(controller.record_arrival("a", 16, 1000))
        self.assertFalse(controller.record_arrival("a", 16, 1000))
        with self.assertRaises(ValueError):
            controller.record_arrival("a", 17, 1000)
        self.assertFalse(controller.record_completion("unknown", 8, 1001))
        self.assertTrue(controller.record_completion("a", 8, 1001))
        self.assertFalse(controller.record_completion("a", 8, 1001))
        with self.assertRaises(ValueError):
            controller.record_completion("a", 9, 1001)
        with self.assertRaises(ValueError):
            controller.record_completion("a", 9, 999)

    def test_completion_replaces_record_without_mutating_captured_snapshot(self):
        controller = self.controller()
        controller.record_arrival("a", 16, 1000)
        captured = controller._records["a"]
        controller.record_completion("a", 8, 1001)
        self.assertIsNone(captured["output_len"])
        self.assertIsNone(captured["finished_time_s"])
        self.assertEqual(controller._records["a"]["output_len"], 8)

    def test_snapshot_copy_does_not_hold_ingress_or_poll_lock(self):
        controller = self.controller()
        self.observe(controller)
        entered, release = threading.Event(), threading.Event()
        original = copy.deepcopy
        snapshots = []

        def delayed(value, *args, **kwargs):
            if isinstance(value, dict) and value.get("request_id") == "done":
                entered.set()
                release.wait(2)
            return original(value, *args, **kwargs)

        with patch("saturo.runtime.history.copy.deepcopy", side_effect=delayed):
            worker = threading.Thread(target=lambda: snapshots.append(
                controller.snapshot()))
            worker.start()
            self.assertTrue(entered.wait(1))
            started = time.monotonic()
            controller.record_completion("pending", 7, 1019)
            self.assertIsNone(controller.poll())
            self.assertLess(time.monotonic() - started, 0.5)
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(snapshots[0].records[1]["output_len"])

    def test_live_retention_and_cutoff_boundaries(self):
        controller = self.controller()
        self.observe(controller)
        self.clock.advance(1800)
        snapshot = controller.snapshot(window_s=1800)
        self.assertEqual(snapshot.records, [])
        self.assertFalse(snapshot.metadata()["provenance"]["cold_start"])
        with self.assertRaises(ValueError):
            controller.snapshot(window_s=1801)
        with self.assertRaises(ValueError):
            controller.snapshot(end_s=999)

    def test_whole_bundle_publication_and_defensive_poll_copy(self):
        controller = self.controller()
        self.observe(controller)
        calls = []
        controller._optimize_p_budget = lambda snap, spec, count: (
            calls.append((snap.end_s, count)) or {"chunk_ms": 80})
        controller._run_refresh()
        bundle = controller.poll()
        self.assertEqual(bundle["version"], 1)
        self.assertEqual(bundle["pool_layout_generation"], 0)
        self.assertEqual(bundle["history_end_s"], 1020)
        self.assertEqual(bundle["p_budgets"], {"tp1_pp1": {"chunk_ms": 80}})
        self.assertEqual(bundle["decode_statistics"]["arrival_rate"], 2 / 20 / 2)
        self.assertEqual(calls, [(1020, 1)])
        bundle["p_budgets"]["tp1_pp1"]["chunk_ms"] = -1
        self.assertIn(b'80.0', controller._latest)
        self.assertIsNone(controller.poll())

    def test_no_partial_D_publication_during_P_optimization(self):
        controller = self.controller()
        self.observe(controller)
        entered, release = threading.Event(), threading.Event()

        def optimize(*args):
            entered.set()
            self.assertTrue(release.wait(2))
            return {"chunk_ms": 80}

        controller._optimize_p_budget = optimize
        worker = threading.Thread(target=controller._run_refresh)
        worker.start()
        self.assertTrue(entered.wait(1))
        self.assertIsNone(controller.poll())
        self.assertIsNone(controller._latest)
        release.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertIsNotNone(controller.poll())

    def test_any_P_failure_retains_entire_previous_bundle(self):
        controller = self.controller()
        self.observe(controller)
        controller._run_refresh()
        previous = controller._latest
        controller._optimize_p_budget = lambda *a: (_ for _ in ()).throw(RuntimeError())
        controller._run_refresh()
        self.assertEqual(controller._latest, previous)
        self.assertEqual(controller.status()["state"], "failed_retaining_previous")
        self.assertEqual(controller.status()["last_error_type"], "RuntimeError")

    def test_missing_or_invalid_P_budget_retains_bundle(self):
        for value in (None, {"chunk_ms": 0}, {"chunk_ms": float("nan")}):
            controller = self.controller()
            self.observe(controller)
            controller._optimize_p_budget = lambda *a, value=value: value
            controller._run_refresh()
            self.assertIsNone(controller.poll())
            self.assertEqual(controller.status()["state"], "failed_retaining_previous")

    def test_disabled_P_saturation_allows_empty_P_controls(self):
        controller = self.controller(enable_prefill_saturation=False)
        self.observe(controller)
        controller._optimize_p_budget = lambda *a: None
        controller._run_refresh()
        self.assertEqual(controller.poll()["p_budgets"], {})

    def test_no_completed_outputs_is_structured_cold_start_skip(self):
        controller = self.controller()
        controller.record_arrival("pending", 100, 1000)
        self.clock.advance(10)
        controller._run_refresh()
        self.assertIsNone(controller.poll())
        self.assertEqual(controller.status()["state"],
                         "cold_start_no_completed_history")
        self.assertEqual(controller.status()["counters"]["cold_start_skips"], 1)

    def test_timeout_result_never_publishes_or_overwrites_previous(self):
        controller = self.controller()
        self.observe(controller)
        controller._run_refresh()
        previous = controller._latest

        def slow(*args):
            self.clock.advance(31)
            return {"chunk_ms": 99}

        controller._optimize_p_budget = slow
        controller._run_refresh()
        self.assertEqual(controller._latest, previous)
        self.assertEqual(controller.status()["counters"]["timeouts"], 1)
        self.assertEqual(controller.status()["counters"]["discarded_late_or_layout"], 1)

    def test_layout_switch_discards_old_inflight_result_then_recomputes(self):
        controller = self.controller()
        self.observe(controller)
        entered, release = threading.Event(), threading.Event()

        def optimize(*args):
            entered.set()
            self.assertTrue(release.wait(2))
            return {"chunk_ms": 80}

        controller._optimize_p_budget = optimize
        worker = threading.Thread(target=controller._run_refresh)
        worker.start()
        self.assertTrue(entered.wait(1))
        specs = copy.deepcopy(SPECS)
        specs[1]["worker_id"] = "new-d0"
        self.assertIsNone(controller.poll(specs, pool_layout_generation=1))
        release.set()
        worker.join(2)
        self.assertIsNone(controller.poll())
        controller._optimize_p_budget = lambda *a: {"chunk_ms": 90}
        controller._run_refresh()
        bundle = controller.poll(specs, pool_layout_generation=1)
        self.assertEqual(bundle["pool_layout_generation"], 1)
        self.assertEqual(bundle["version"], 2)
        with self.assertRaises(ValueError):
            controller.poll(specs, pool_layout_generation=0)

    def test_role_change_and_reordered_specs_guard(self):
        controller = self.controller()
        self.observe(controller)
        controller._run_refresh()
        self.assertIsNotNone(controller.poll(list(reversed(SPECS))))
        specs = copy.deepcopy(SPECS)
        specs[2]["role"] = "prefill"
        self.assertIsNone(controller.poll(specs))
        self.assertEqual(controller.status()["pool_layout_generation"], 1)
        controller._run_refresh()
        self.assertEqual(controller.poll()["decode_statistics"]["arrival_rate"], 2 / 20)

    def test_expired_before_delivery_is_not_applied(self):
        controller = self.controller()
        self.observe(controller)
        controller._run_refresh()
        self.clock.advance(81)
        self.assertIsNone(controller.poll())
        self.assertEqual(controller.status()["state"], "expired_retaining_previous")
        self.assertEqual(controller.status()["counters"]["expired_before_delivery"], 1)

    def test_serialization_failure_is_retained_not_thread_crash(self):
        controller = self.controller()
        self.observe(controller)
        controller._run_refresh()
        previous = controller._latest
        original = controller.refresh_once

        def bad(specs):
            update = original(specs)
            update["decode_statistics"]["mean_output_len"] = float("nan")
            return update

        controller.refresh_once = bad
        controller._run_refresh()
        self.assertEqual(controller._latest, previous)
        self.assertEqual(controller.status()["state"], "failed_retaining_previous")

    def test_close_is_bounded_and_suppresses_late_publication(self):
        controller = self.controller()
        self.observe(controller)
        entered, release = threading.Event(), threading.Event()

        def optimize(*args):
            entered.set()
            release.wait(2)
            return {"chunk_ms": 80}

        controller._optimize_p_budget = optimize
        controller._thread.start()
        self.assertTrue(entered.wait(1))
        started = time.monotonic()
        status = controller.close(timeout=0.01)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(status["thread_alive"])
        self.assertTrue(status["closed"])
        release.set()
        controller._thread.join(2)
        self.assertFalse(controller._thread.is_alive())
        self.assertIsNone(controller.poll())
        self.assertIsNone(controller._latest)
        self.assertFalse(controller.record_arrival("late", 1))
        self.assertFalse(controller.request_refresh())

    def test_background_requests_coalesce_in_single_thread(self):
        controller = self.controller()
        self.observe(controller)
        entered, release, second = (threading.Event(), threading.Event(),
                                    threading.Event())
        calls = []

        def optimize(*args):
            calls.append(threading.get_ident())
            if len(calls) == 1:
                entered.set()
                release.wait(2)
            else:
                second.set()
            return {"chunk_ms": 80}

        controller._optimize_p_budget = optimize
        controller._thread.start()
        self.assertTrue(entered.wait(1))
        for _ in range(100):
            controller.request_refresh()
        release.set()
        self.assertTrue(second.wait(1))
        controller.close()
        self.assertEqual(len(set(calls)), 1)
        self.assertEqual(len(calls), 2)
        self.assertGreater(controller.status()["counters"]["coalesced_requests"], 0)

    def test_redis_shared_snapshot_keeps_explicit_cutoff_and_source(self):
        controller = self.controller()
        controller.source = "redis-history:k"
        snapshot = HistorySnapshot("k", "m", 400, 1000, [])
        with patch("saturo.runtime.history.read_redis_history",
                   return_value=snapshot) as read:
            merged = controller.snapshot(window_s=600, end_s=1000)
            self.assertEqual(merged.source, "redis+live_ingress")
            self.assertEqual(merged.records, [])
            self.assertEqual(merged.start_s, 1000)
            self.assertTrue(merged.metadata()["provenance"]["cold_start"])
            read.assert_called_once_with("redis-history:k", window_s=600,
                                         model="m", end_s=1000)

    def test_independent_redis_history_merges_current_live_observations(self):
        controller = self.controller()
        self.observe(controller)
        controller.source = "redis-history:independent"
        durable = HistorySnapshot("independent", "m", 420, 1020, [{
            "run_id": "prior-run", "request_id": "old", "model": "m",
            "arrival_time_s": 900, "prompt_len": 20,
            "output_len": 10, "finished_time_s": 901,
        }])
        with patch("saturo.runtime.history.read_redis_history",
                   return_value=durable):
            merged = controller.snapshot(window_s=600)
        self.assertEqual(len(merged.records), 3)
        self.assertEqual(merged.start_s, 900)
        self.assertEqual(merged.statistics()["arrival_rate"], 3 / 120)
        self.assertEqual(merged.statistics()["mean_output_len"], 11)
        self.assertEqual(merged.metadata()["source"], "redis+live_ingress")
        merged.records[0]["output_len"] = 888
        self.assertEqual(durable.records[0]["output_len"], 10)

    def test_redis_live_same_run_dedup_keeps_observed_completion(self):
        controller = self.controller()
        controller._run_id = "shared-trace-run"
        self.observe(controller)
        controller.source = "redis-history:shared"
        duplicate = copy.deepcopy(controller._records["done"])
        duplicate["output_len"] = duplicate["finished_time_s"] = None
        remote_completed = copy.deepcopy(controller._records["pending"])
        remote_completed.update(output_len=9, finished_time_s=1011)
        durable = HistorySnapshot("shared", "m", 420, 1020,
                                  [duplicate, remote_completed])
        with patch("saturo.runtime.history.read_redis_history",
                   return_value=durable):
            merged = controller.snapshot(window_s=600)
        self.assertEqual(len(merged.records), 2)
        self.assertEqual([row["output_len"] for row in merged.records], [12, 9])

    def test_redis_same_request_different_run_does_not_dedup(self):
        controller = self.controller()
        self.observe(controller)
        controller.source = "redis-history:independent"
        old = copy.deepcopy(controller._records["done"])
        old["run_id"] = "different-run"
        durable = HistorySnapshot("k", "m", 420, 1020, [old])
        with patch("saturo.runtime.history.read_redis_history",
                   return_value=durable):
            self.assertEqual(len(controller.snapshot().records), 3)

    def test_redis_failure_or_conflict_retains_previous_bundle(self):
        controller = self.controller()
        self.observe(controller)
        controller._run_refresh()
        previous = controller._latest
        controller.source = "redis-history:k"
        with patch("saturo.runtime.history.read_redis_history",
                   side_effect=ConnectionError):
            controller._run_refresh()
        self.assertEqual(controller._latest, previous)
        duplicate = copy.deepcopy(controller._records["done"])
        duplicate["output_len"] = 999
        durable = HistorySnapshot("k", "m", 420, 1020, [duplicate])
        with patch("saturo.runtime.history.read_redis_history",
                   return_value=durable):
            controller._run_refresh()
        self.assertEqual(controller._latest, previous)
        self.assertEqual(controller.status()["last_error_type"], "ValueError")

    def test_redis_unfinished_length_label_cannot_become_observed_output(self):
        controller = self.controller()
        self.observe(controller)
        controller.source = "redis-history:k"
        unknown = {"run_id": "old", "request_id": "unverified", "model": "m",
                   "arrival_time_s": 900, "prompt_len": 64,
                   "output_len": 99999, "requested_output_len": 99999,
                   "finished_time_s": None}
        durable = HistorySnapshot("k", "m", 420, 1020, [unknown])
        with patch("saturo.runtime.history.read_redis_history",
                   return_value=durable):
            snapshot = controller.snapshot()
        self.assertIsNone(snapshot.records[0]["output_len"])
        self.assertEqual(snapshot.statistics()["mean_output_len"], 12)
        self.assertEqual(snapshot.statistics()["completed_requests"], 1)
        unknown.update(run_id=controller._run_id, request_id="done",
                       prompt_len=16, arrival_time_s=1005)
        with patch("saturo.runtime.history.read_redis_history",
                   return_value=durable):
            deduplicated = controller.snapshot()
        self.assertEqual(len(deduplicated.records), 2)
        self.assertEqual(deduplicated.statistics()["mean_output_len"], 12)

    def test_larger_reallocation_window_is_retained_without_fake_idle(self):
        controller = self.controller(resource_reallocation_history_window_s=3600)
        self.observe(controller)
        self.clock.advance(2000)
        snapshot = controller.snapshot(window_s=3600)
        self.assertEqual(len(snapshot.records), 2)
        self.assertEqual(snapshot.start_s, 1000)
        self.assertEqual(snapshot.statistics()["arrival_rate"], 2 / 2020)
        self.assertTrue(snapshot.metadata()["provenance"]["cold_start"])


if __name__ == "__main__":
    unittest.main()
