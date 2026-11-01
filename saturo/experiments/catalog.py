"""Paper experiment catalog. Grid choices are reviewable, not result data."""

STUDIES = {
    "fig1_fragmentation": {
        "section": "3.1.1", "figure": 1, "category": "motivation",
        "description": "RT prefill queue/execution timeline and decode KV fragmentation",
        "backends": ["prefill_timeline", "decode_kv"],
    },
    "fig2_colocation": {
        "section": "3.1.2", "figure": 2, "category": "motivation",
        "description": "RT-decode/BE colocation: arrival-rate and P99 TBT timeline",
        "backends": ["decode_colocation"],
    },
    "fig3_budget_surface": {
        "section": "3.2", "figure": 3, "category": "motivation",
        "description": "BE throughput across RT load and P budget / D target TBT",
        "backends": ["p_budget_surface", "d_target_surface"],
    },
    "fig5_workload_cdf": {
        "section": "8.1", "figure": 5, "category": "workloads",
        "description": "Real input/output length CDFs for Conv, Code, BE-Mix",
        "backends": ["workload_stats"],
    },
    "fig6_e2e": {
        "section": "8.2", "figure": 6, "category": "e2e",
        "description": "Llama/Mistral x Conv/Code x Saturo",
    },
    "fig7_arrivals": {
        "section": "8.2", "figure": 7, "category": "workloads",
        "description": "Conv-ShareGPT arrival-rate timeline",
        "backends": ["workload_stats"],
    },
    "fig8_timeline": {
        "section": "8.2", "figure": 8, "category": "e2e",
        "description": "Mistral-Conv, 20 RT req/s, 400 s; BE throughput/P99 TTFT/TBT",
    },
    "fig9_slo_scale": {
        "section": "8.3", "figure": 9, "category": "sensitivity",
        "description": "Mistral-Conv, RT=16; scale BOTH default SLOs",
        "values": [1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.75],
        "grid_note": "Endpoints 1.3 and 0.75 are explicit; intermediate grid is a review default.",
    },
    "fig10_cluster_size": {
        "section": "8.3", "figure": 10, "category": "sensitivity",
        "description": "Mistral-Conv, RT=12, G=32/24/16",
        "values": [32, 24, 16],
    },
    "fig11_saturation": {
        "section": "8.4", "figure": 11, "category": "ablation",
        "description": "P/D disabled pools serve RT ONLY; BE remains in eligible pools",
        "panel_a_rps": [12, 16, 20, 24, 28],
        "panel_b_rps": [16, 20, 24, 28, 32],
        "variants": {
            "saturo": [True, True],
            "without_p_saturation": [False, True],
            "without_d_saturation": [True, False],
            "without_saturation": [False, False],
        },
        "grid_note": "The two printed panels have different RT-load axes; retain both rather than silently merging their plotting ranges.",
    },
    "fig12_allocation": {
        "section": "8.4", "figure": 12, "category": "ablation",
        "description": "Saturo vs minimum-RT-demand allocation; BOTH retain P/D saturation",
        "values": [4, 8, 12, 16, 20, 24, 28],
    },
    "fig13_p_overhead": {
        "section": "8.5", "figure": "13a", "category": "overhead",
        "description": "P budget optimization CPU time vs history length",
        "values": [60, 120, 300, 600, 1200],
        "grid_note": "60/300/1200 are printed ticks; 600 is discussed; 120 is a review default.",
    },
    "fig13_d_overhead": {
        "section": "8.5", "figure": "13b", "category": "overhead",
        "description": "D target optimization sync/async overhead vs RT batch size",
        "values": [16, 32, 64, 128, 256, 512],
        "grid_note": "16/64/512 are printed ticks; other powers of two are review defaults.",
    },
    "fig13_allocator_overhead": {
        "section": "8.5", "figure": "13c", "category": "overhead",
        "description": "Allocator median/P95 CPU time vs GPU budget",
        "values": [32, 64, 128, 256, 512, 1024],
        "grid_note": "32/128/1024 are printed ticks; intermediate powers of two are review defaults.",
    },
    "fig14_history_window": {
        "section": "8.6", "figure": "14a", "category": "sensitivity",
        "description": "Per-P-instance BE throughput vs history K; default K=600 s",
        "values": [60, 120, 300, 600, 1200],
        "grid_note": "Intermediate points not fully specified in text; review the chosen grid before reproduction.",
    },
    "fig14_quantile": {
        "section": "8.6", "figure": "14b", "category": "sensitivity",
        "description": "Per-D-instance BE throughput and RT attainment vs QRF eta",
        "values": [0.5, 0.75, 0.9, 0.95, 0.99],
    },
    "fig14_update_interval": {
        "section": "8.6", "figure": "14c", "category": "sensitivity",
        "description": "Whole-system P budget AND D target update interval Delta",
        "values": [5, 10, 20, 40, 80, 160],
        "grid_note": "5/20/160 printed and 40 discussed; other points are review defaults. Delta is NOT the prediction lookahead horizon.",
    },
}

METHODS = ["saturo"]

NATIVE_PAPER_GAPS = [
    "content-QRF capacity-profile rebuilding still requires predictor-type alignment",
    "decode saturation memory peak checks still require correction",
    "continuous pull-based BE dispatch is not yet implemented in the e2e runtime",
    "canonical per-request joint TTFT/TBT attainment is not yet persisted (diagnostic token timelines are available)",
]

BACKENDS = {
    "prefill_timeline": {"entry": "saturo/experiments/queue_sim/simulate.py", "kind": "legacy_pipeline",
                         "gap": "requires measured prefill trace followed by queue reconstruction"},
    "decode_kv": {"entry": "saturo/experiments/kv_usage/benchmark.py", "kind": "legacy_gpu",
                   "gap": "requires an explicitly reviewed per-instance GPU configuration"},
    "decode_colocation": {"entry": "saturo/experiments/colocate/benchmark_fixed_budget.py", "kind": "legacy_gpu",
                          "gap": "decode-only RT/BE timeline adapter needs review; do not substitute prefill RT"},
    "p_budget_surface": {"entry": "saturo/experiments/pool_validation/benchmark_p_pool_saturation.py", "kind": "legacy_gpu",
                         "gap": "requires explicit P-budget grid and matching trace inputs"},
    "d_target_surface": {"entry": None, "kind": "missing_production_control",
                         "gap": "no verified fixed-D-target control; changing TBT SLO is not equivalent"},
    "d_async_overhead": {"entry": None, "kind": "missing_measurement_adapter",
                         "gap": "native async targets are implemented; the standalone real-GPU overlap measurement adapter is not yet connected"},
    "p_history": {"entry": "saturo/experiments/pool_validation/run_history_window.py", "kind": "gpu"},
    "d_quantile": {"entry": "saturo/experiments/pool_validation/run_quantile.py", "kind": "gpu"},
    "update_interval": {"entry": "saturo/run_saturo_endtoend_experiment.py", "kind": "gpu"},
}

