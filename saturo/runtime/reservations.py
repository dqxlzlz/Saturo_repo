"""Early D-instance claims and control-before-prefill RPC ordering."""

from saturo.runtime.workload import DECODE, PREFILL, RT

RESERVATION_COMMANDS = ("reserve_decode", "release_decode")
CONTROL_COMMANDS = RESERVATION_COMMANDS + ("update_epoch", "history_prepare", "history_commit")


class DecodeReservationMixin:
    def _decode_rt_slot_occupants(self, worker):
        """Count each live RT once across its P claim and D attempt.

        Native D prioritizes RT over BE, but cannot preempt another running
        RT to create a sequence slot. Reserve that slot before P emits its
        first token, including when adaptive role selection is disabled.
        """
        occupants = set()
        for attempt_id, request_id in worker.active_attempts.items():
            req = self._request_by_id(request_id)
            if (req is not None and req.kind == RT and not req.is_finished
                    and req.active_attempt_id == attempt_id):
                occupants.add(req.request_id)
        for reservation_id in worker.decode_reservations:
            request_id = reservation_id.rsplit(":prefill:", 1)[0]
            req = self._request_by_id(request_id)
            if (req is not None and req.kind == RT and not req.is_finished
                    and req.decode_reservation_id == reservation_id
                    and req.reserved_decode_worker_id == worker.worker_id):
                occupants.add(req.request_id)
        return occupants

    def _reserve_decode_for_prefill(self, req):
        current = self.workers.get(req.reserved_decode_worker_id)
        if (req.decode_reservation_id and current is not None
                and current.role == DECODE and current.ready
                and not current.retired
                and req.decode_reservation_id in current.decode_reservations):
            return True
        self._release_decode_reservation(req)
        candidates = [w for w in self._live_workers()
                      if w.role == DECODE and w.ready]
        switcher = getattr(self, "role_switcher", None)
        if switcher is not None:
            worker = switcher.select(DECODE, req)
            if worker is None:
                return False
        else:
            # BE slots are reclaimable by native D; live RT slots are not.
            # Zero is retained as the legacy embedding/test "unknown" limit.
            candidates = [(w, len(self._decode_rt_slot_occupants(w)))
                          for w in candidates]
            candidates = [(w, occupied) for w, occupied in candidates
                          if not w.max_num_seqs or occupied < w.max_num_seqs]
            if not candidates:
                return False
            worker, _ = min(candidates, key=lambda item: (
                item[1], len(item[0].active_attempts)
                + len(item[0].decode_reservations), item[0].worker_id))
        # A claim can be released and replaced before P starts. Give each
        # incarnation a unique ID so an older in-flight ACK cannot confirm it.
        sequence = getattr(self, "_decode_reservation_sequence", 0) + 1
        self._decode_reservation_sequence = sequence
        reservation_id = (f"{req.request_id}:prefill:{req.prefill_attempts + 1}"
                          f":claim:{sequence}")
        req.reserved_decode_worker_id = worker.worker_id
        req.decode_reservation_id = reservation_id
        req.decode_reservation_confirmed = False
        worker.decode_reservations.add(reservation_id)
        worker.pending_commands.append({
            "cmd": "reserve_decode", "request_id": req.request_id,
            "reservation_id": reservation_id,
            "prompt_len": req.decode_prompt_len + 1,
        })
        return True

    def _release_decode_reservation(self, req, *, notify=True):
        reservation_id = req.decode_reservation_id
        worker = self.workers.get(req.reserved_decode_worker_id)
        if reservation_id and worker is not None:
            worker.decode_reservations.discard(reservation_id)
            if notify and not worker.retired:
                worker.pending_commands = [c for c in worker.pending_commands
                    if not (c.get("cmd") == "reserve_decode"
                            and c.get("reservation_id") == reservation_id)]
                worker.pending_commands.append({
                    "cmd": "release_decode", "request_id": req.request_id,
                    "reservation_id": reservation_id,
                })
        req.reserved_decode_worker_id = None
        req.decode_reservation_id = None
        req.decode_reservation_confirmed = False

    def _invalidate_decode_reservations(self, worker):
        for req in getattr(self, "all_rt", ()):
            if req.reserved_decode_worker_id == worker.worker_id:
                self._release_decode_reservation(req, notify=False)
                if not req.is_finished:
                    self._reserve_decode_for_prefill(req)
        worker.decode_reservations.clear()

    def _flush_decode_reservations(self):
        ray = self._ensure_ray()
        pending = {}
        failed = set()
        for worker in self._live_workers():
            controls = [c for c in worker.pending_commands
                        if c.get("cmd") in CONTROL_COMMANDS]
            if not controls:
                continue
            worker.pending_commands = [c for c in worker.pending_commands
                                       if c.get("cmd") not in CONTROL_COMMANDS]
            ref = worker.actor.submit_batch.remote(controls)
            pending[ref] = (worker, controls)
        for ref, (worker, controls) in pending.items():
            try:
                acknowledgements = ray.get(ref, timeout=5.0)
            except Exception as exc:
                if self._fail_dead_engine_rpc(worker, exc, "decode_reservations"):
                    return failed | {worker.worker_id}
                self._retire_worker(worker, requeue_active=True)
                failed.add(worker.worker_id)
                continue
            self._acknowledge_decode_reservations(
                worker, controls, acknowledgements)
            callback = getattr(self, "_acknowledge_runtime_controls", None)
            if callback is not None:
                callback(worker, controls, acknowledgements)
        return failed

    def _acknowledge_decode_reservations(self, worker, commands, acknowledgements):
        """Confirm only matching early claims after the actor acknowledges them."""
        if len(acknowledgements) != len(commands):
            raise RuntimeError("Incomplete D reservation acknowledgement")
        for command, ack in zip(commands, acknowledgements):
            if command.get("cmd") not in RESERVATION_COMMANDS:
                continue
            if not ack.get("accepted", False):
                raise RuntimeError("D decode reservation rejected: "
                                   + str(ack.get("reason", "unknown")))
            if command["cmd"] != "reserve_decode":
                continue
            req = self._request_by_id(command["request_id"])
            if (req is not None and req.decode_reservation_id
                    == command["reservation_id"]
                    and req.reserved_decode_worker_id == worker.worker_id
                    and command["reservation_id"] in worker.decode_reservations):
                req.decode_reservation_confirmed = True

    def _prefill_reservation_ready(self, command):
        if (command.get("cmd", "add") != "add"
                or command.get("kind") != RT
                or command.get("phase") != PREFILL):
            return True
        req = self._request_by_id(str(command["request_id"]))
        if req is None or req.is_finished:
            return False
        if not req.decode_reservation_id:
            self._reserve_decode_for_prefill(req)
        return req.decode_reservation_confirmed

    def _reserved_decode_worker(self, req):
        worker = self.workers.get(req.reserved_decode_worker_id)
        if (worker is None or worker.retired or not worker.ready):
            self._release_decode_reservation(req, notify=False)
            self._reserve_decode_for_prefill(req)
            return None
        return worker if req.decode_reservation_confirmed else None
