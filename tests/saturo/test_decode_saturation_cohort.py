"""Exact cohort fast-path parity, boundaries, fallbacks and callback contracts."""
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from saturo.saturation.types import RequestForecast, SaturationParams
from saturo.saturation import decode_saturation_core as core

if os.environ.get("SATURO_TEST_COHORT_SOURCE"):
    spec=importlib.util.spec_from_file_location("saturo.saturation._cohort_stage",
        os.environ["SATURO_TEST_COHORT_SOURCE"])
    core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core
    spec.loader.exec_module(core)


def scalar(**values):
    return float(values["decode_tokens"])/8+float(values["decode_sum_ctx"])/1024


def batch(**values):
    return (np.asarray(values["decode_tokens"],dtype=np.float64)/8
            +np.asarray(values["decode_sum_ctx"],dtype=np.float64)/1024)


def forward(prefill,decode):
    return len(decode)/8+sum(ctx for _,ctx in decode)/1024


class Broadcast(core.DecodeSaturationEngine):
    def _evaluate_tau_from_arrays(self,arrays,tau_ms):
        return self._evaluate_tau_broadcast(arrays,tau_ms)


@pytest.mark.parametrize("seed",range(30))
def test_original_floor_and_nextafter_boundaries(seed):
    rng=np.random.default_rng(seed)
    for _ in range(10):
        count=int(rng.integers(2,200));tau=float(rng.choice([1e-6,.001,.025111826289163787,.1,1.0]))
        remaining=int(rng.integers(0,150));start=int(rng.integers(0,1000));block=int(rng.choice([1,8,16,32]))
        arrivals=np.sort(rng.uniform(0,60,count))
        if seed%3==0:arrivals=np.round(arrivals,2)
        if seed%5==0:arrivals[:count//2]=0
        offsets=np.r_[0.,60.,arrivals[:5],arrivals[:5]+remaining*tau,
                      arrivals[:5]+(remaining+1-1e-9)*tau]
        offsets=np.r_[offsets,np.nextafter(offsets,-np.inf),np.nextafter(offsets,np.inf)]
        actual=core._identical_cohort_statistics(arrivals,start,remaining,offsets,tau,block)
        elapsed=offsets[:,None]-arrivals[None,:]
        steps=np.floor(elapsed/tau+1e-9).astype(np.int64)
        alive=(elapsed>=0)&(steps<=remaining)
        np.maximum(steps,0,out=steps);np.minimum(steps,remaining,out=steps)
        lengths=steps+start
        expected=(alive.sum(axis=1),np.where(alive,lengths,0).sum(axis=1),
                  np.where(alive,(lengths+block-1)//block,0).sum(axis=1))
        for value,want in zip(actual,expected):np.testing.assert_array_equal(value,want)


@pytest.mark.parametrize("seed",range(5))
@pytest.mark.parametrize("use_batch",[False,True])
def test_complete_decision_matches_original_with_residual_rt(seed,use_batch):
    rng=np.random.default_rng(seed)
    future=[RequestForecast(f"f{i}",486,0,563,float(x))
            for i,x in enumerate(np.sort(rng.uniform(0,60,1100)))]
    active=[RequestForecast(str(i),int(rng.integers(1,900)),int(rng.integers(0,100)),
                int(rng.integers(0,200)),0) for i in range(3)]
    params=SaturationParams(tbt_slo_ms=100,lookahead_window_s=60,
                           block_size=16,total_gpu_blocks=19663)
    kwargs=dict(aggregate_batch_time_predictor_fn=batch) if use_batch else {}
    model=core.DecodeSaturationEngine(params,forward,scalar,**kwargs)
    reference=Broadcast(params,forward,scalar)
    assert model.compute_decision(active,future,100,5000,0,30)==reference.compute_decision(active,future,100,5000,0,30)


def test_custom_scalar_callback_order_and_none_predictor():
    arrays=(np.linspace(0,60,1024),np.full(1024,32),np.full(1024,10))
    records=[]
    def callback(**values):records.append(values);return scalar(**values)
    model=core.DecodeSaturationEngine(SaturationParams(lookahead_window_s=60),forward,callback)
    actual=model._evaluate_tau_from_arrays(arrays,25)
    called=list(records);records.clear()
    assert actual==model._evaluate_tau_broadcast(arrays,25)
    assert records==called
    records.clear();model.time_predictor_fn=None
    assert model._evaluate_tau_from_arrays(arrays,25)==model._evaluate_tau_broadcast(arrays,25)
    assert not records


@pytest.mark.parametrize("change",["small","large","block","negative_start","negative_remaining",
    "huge_start","huge_remaining","huge_quotient","no_cohort","no_aggregate","grid"])
def test_unsupported_or_unbounded_domain_keeps_original_path(monkeypatch,change):
    n=1023 if change=="small" else 4097 if change=="large" else 1024
    arr=np.linspace(0,60,n);start=np.full(n,32,dtype=np.int64);remaining=np.full(n,10,dtype=np.int64)
    block=129 if change=="block" else 16
    if change=="negative_start":start[0]=-1
    if change=="negative_remaining":remaining[0]=-1
    if change=="huge_start":start[0]=2**31
    if change=="huge_remaining":remaining[0]=2**31
    if change=="huge_quotient":arr[-1]=2**30
    if change=="no_cohort":start=np.arange(n)
    model=core.DecodeSaturationEngine(SaturationParams(block_size=block,lookahead_window_s=60),
        forward,None if change=="no_aggregate" else scalar)
    if change=="grid":model._MIN_TBT_OFFSETS=5000
    monkeypatch.setattr(model,"_evaluate_tau_broadcast",lambda *args:(123,4.0))
    assert model._evaluate_tau_from_arrays((arr,start,remaining),.001)==(123,4.0)


def test_rounding_plateau_falls_back_before_callbacks(monkeypatch):
    arrays=(np.linspace(0,60,1024),np.full(1024,32),np.full(1024,10))
    def fail(*args):raise ArithmeticError("rounding plateau")
    monkeypatch.setattr(core,"_identical_cohort_statistics",fail)
    model=core.DecodeSaturationEngine(SaturationParams(lookahead_window_s=60),forward,scalar)
    assert model._evaluate_tau_from_arrays(arrays,25)==model._evaluate_tau_broadcast(arrays,25)


def test_real_close_arrival_plateau_preserves_original(monkeypatch):
    arrays=(np.arange(1,1025)*1e-20,np.full(1024,16),np.ones(1024,dtype=np.int64))
    model=core.DecodeSaturationEngine(SaturationParams(lookahead_window_s=(2-1e-9)*.1),forward,scalar)
    assert model._evaluate_tau_from_arrays(arrays,100)==model._evaluate_tau_broadcast(arrays,100)
