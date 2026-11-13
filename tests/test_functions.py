"""Explicit k-assignment functions and dispatcher equivalence."""
import sys
import time

import numpy as np
import pytest

import kap
from conftest import random_instance


@pytest.fixture(params=[True, False])
def costs(request):
    return random_instance(4, equal=request.param, dtype=np.int64, seed=13)


def test_ils_equivalence(costs):
    assert kap.ils(costs, max_iter=3, n_chains=4, neighborhood="sDv", random_state=42,
                   return_free=True) == kap.k_assignment(costs, algo="ils", max_iter=3,
                   n_trials=4, refine="sdv", random_state=42, return_free=True)


def test_memetic_equivalence(costs):
    assert kap.memetic(costs, generations=2, population=6, random_state=42) == \
        kap.k_assignment(costs, algo="memetic", max_iter=2, population=6, random_state=42)


@pytest.mark.parametrize("start", ["Am", "Bm", "Dm", "Shift"])
@pytest.mark.parametrize("neighborhood", ["1dv", "sdv"])
def test_local_search_equivalence(costs, start, neighborhood):
    assert kap.local_search(costs, start=start, neighborhood=neighborhood, max_sweeps=5) == \
        kap.k_assignment(costs, algo=start, refine=neighborhood, max_iter=5, random_state=42)


@pytest.mark.parametrize("method", ["Am", "Bm", "Cm", "Dm", "Em", "Fm", "Shift"])
@pytest.mark.parametrize("backend", ["kap", "scipy"])
def test_multiple_hungarian_equivalence(costs, method, backend):
    if method == "Shift" and backend == "scipy":
        with pytest.raises(ValueError):
            kap.multiple_hungarian(costs, method, backend=backend)
        return
    options = dict(max_iter=3, n_trials=2, random_state=42, backend=backend, return_free=True)
    assert kap.multiple_hungarian(costs, method, **options) == kap.k_assignment(costs, algo=method, **options)


@pytest.mark.parametrize("function,args", [(kap.ils, (1,)), (kap.memetic, (1,)),
                          (kap.local_search, ("sdv",)), (kap.multiple_hungarian, ("Cm", 1))])
def test_keyword_only(costs, function, args):
    with pytest.raises(TypeError):
        function(costs, *args)


@pytest.mark.parametrize("function,options,name", [
    (kap.ils, {"n_chains": 0}, "n_chains"),
    (kap.ils, {"neighborhood": "x"}, "neighborhood"),
    (kap.ils, {"max_iter": -1}, "max_iter"),
    (kap.local_search, {"start": "Cm"}, "start"),
    (kap.local_search, {"max_sweeps": -1}, "max_sweeps"),
    (kap.memetic, {"generations": -1}, "generations"),
    (kap.memetic, {"population": 1}, "population"),
    (kap.ils, {"time_limit": 0}, "time_limit"),
    (kap.memetic, {"time_limit": float("nan")}, "time_limit"),
    (kap.multiple_hungarian, {"method": "ils"}, "method"),
])
def test_invalid(costs, function, options, name):
    with pytest.raises((ValueError, TypeError), match=name):
        function(costs, **options)


def test_timed_ils():
    costs = random_instance(3, sizes=[30, 30, 30], dtype=np.int64, seed=7)
    before = time.monotonic()
    assert kap.ils(costs, max_iter=0, time_limit=1e9, random_state=42) == \
        kap.ils(costs, max_iter=0, random_state=42)
    assert time.monotonic() - before < 5
    before = time.monotonic()
    kap.ils(costs, time_limit=0.2, random_state=42)
    assert time.monotonic() - before >= 0.15


def test_missing_core(monkeypatch, costs):
    expected = kap._reference.k_assignment(costs, algo="Cm", random_state=42)
    monkeypatch.setitem(sys.modules, "kap._core", None)
    for function in (kap.ils, kap.memetic, kap.local_search):
        with pytest.raises(RuntimeError, match="native core.*required"):
            function(costs)
    assert kap.multiple_hungarian(costs, "Cm", random_state=42) == expected
