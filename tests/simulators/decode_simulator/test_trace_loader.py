import csv

import pytest

from saturo.simulators.decode_simulator.trace_loader import load_rt_trace


def test_duration_window_preserves_trace_bursts_without_fixed_count(tmp_path):
    trace = tmp_path / "rt.csv"
    with trace.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "request_id",
                "prompt_len",
                "output_len",
                "arrival_offset_s",
            ),
        )
        writer.writeheader()
        for request_id, arrival_s in enumerate((0.0, 1.0, 9.0, 10.0)):
            writer.writerow({
                "request_id": request_id,
                "prompt_len": 8,
                "output_len": 4,
                "arrival_offset_s": arrival_s,
            })

    # Native mean rate is 4/10=0.4 RPS, so targeting 0.8 RPS uniformly
    # halves every gap: [1, 8, 1] -> [0.5, 4, 0.5]. The 4.75-second
    # duration cuts by scaled timestamp and therefore retains three requests;
    # it does not force round(4.75 * 0.8)=4 requests into the window.
    requests = load_rt_trace(
        str(trace),
        duration_s=4.75,
        target_rps=0.8,
    )

    assert [request.request_id for request in requests] == ["0", "1", "2"]
    assert [request.arrival_time_s for request in requests] == pytest.approx(
        [0.0, 0.5, 4.5]
    )
    assert [
        requests[index + 1].arrival_time_s - requests[index].arrival_time_s
        for index in range(len(requests) - 1)
    ] == pytest.approx([0.5, 4.0])
