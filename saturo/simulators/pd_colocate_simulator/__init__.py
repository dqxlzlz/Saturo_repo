"""PD-aggregated RT-BE colocation throughput simulator (steady-state model).

Unlike the disaggregated (PD-separate) setup, here a single engine runs
chunked-prefill: every iteration processes a mix of decode tokens (RT + BE,
one token per resident sequence) and a prefill chunk, all bounded by a single
per-iteration token budget ``chunk_size`` (vLLM ``max_num_batched_tokens``).

Given an RT arrival rate (RPS) and a chunk_size, the model:

1. Places decode first (RT decode + BE decode) — each resident sequence emits
   one token per iteration.
2. Gives RT-Prefill just enough of the remaining token budget so that RT
   prefill throughput >= RPS (i.e. RT does not accumulate).
3. Fills BE decode concurrency with whatever KV blocks RT decode leaves free.
4. Gives every token left in the chunk budget to BE-Prefill.

BE useful throughput is ``min(BE-Prefill, BE-Decode)`` because every BE request
must be both prefilled and decoded. Sweeping chunk_size trades a larger
per-iteration batch (more BE prefill room, higher iteration latency, and — via
Little's law — more RT decode residency eating the KV budget) against BE decode
concurrency. The optimizer picks the feasible chunk_size that maximizes BE
throughput.
"""
