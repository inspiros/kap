# Benchmarks

Two measurements: the [speed of the native core](#native-core-against-the-python-implementation)
against the Python implementation, and the [solution quality](#solution-quality-of-the-search-methods)
of ILS and memetic against the multiple Hungarian methods.

## Native core against the Python implementation

`kap.k_assignment` with the native core (`backend="kap"`, the default) against the
Python implementation of the same algorithms (`kap._reference.k_assignment`, which
runs the multiple Hungarian methods in Python but calls the same native LAP solver).
kap 0.2.x used other LAP backends, which are not measured here.

Setup: a 144-core lab server, otherwise idle, kap 0.3.0a1 at commit 7ef4e85, Python 3.12,
one pinned core per run, single-threaded (`RAYON_NUM_THREADS=1`). Costs are random
integers in [0, 100) plus uniform noise (float64); n × n matrices; median of 3 seeds;
`max_iter=100`, `n_trials=1`. Script: `develop/bench_kap.py` (the full results, including
int64 inputs, are in `develop/bench_kap_server4.csv`). From n = 50 up, int64 inputs run
up to 2.5x faster than float64 in the native core.

| Algo | k | n | native | Python | speedup | cost, native / Python |
|---|---|---|---|---|---|---|
| Am | 6 | 10 | 0.091 ms | 1.2 ms | 13x | = |
| Am | 6 | 100 | 1.8 ms | 7.2 ms | 4.0x | = |
| Am | 6 | 500 | 67 ms | 78 ms | 1.2x | = |
| Dm | 8 | 10 | 0.54 ms | 26 ms | 48x | = |
| Dm | 8 | 100 | 25 ms | 152 ms | 6.0x | = |
| Dm | 8 | 500 | 1.3 s | 2.4 s | 1.9x | = |
| Bm | 4 | 100 | 8.5 ms | 55 ms | 6.5x | = |
| Bm | 6 | 10 | 11 ms | 1.2 s | 111x | = |
| Bm | 6 | 100 | 694 ms | 8.7 s | 13x | = |
| Bm | 6 | 200 | 2.9 s | 20 s | 6.9x | = |
| Cm | 4 | 100 | 20 ms | 157 ms | 8.0x | 1.005 |
| Cm | 5 | 10 | 5.3 ms | 227 ms | 43x | 0.990 |
| Cm | 5 | 100 | 303 ms | 1.8 s | 5.8x | 0.997 |
| Em | 4 | 100 | 20 ms | 116 ms | 5.9x | 1.000 |
| Em | 5 | 100 | 240 ms | 1.8 s | 7.3x | 0.997 |
| Fm | 3 | 100 | 4.4 ms | 45 ms | 10x | 0.931 |
| Fm | 4 | 100 | 605 ms | 157 ms | 0.26x | 0.950 |
| Fm | 5 | 100 | 7.7 s | 2.2 s | 0.28x | 0.988 |

`=` marks identical results (Am, Bm and Dm are deterministic and agree exactly).

- **Where the gain comes from.** The native core removes the Python overhead of building
  quotient graphs, so the speedup is largest for small n and large k, where that overhead
  dominates (up to about 110x for Bm). As n grows the LAP solves dominate, and both versions
  use the same native LAP, so the gap narrows (Am and Dm at n = 500: 1.2-2x).
- **Bm** also solves fewer LAPs natively, because merges that commute are memoised
  (see [Algorithms](algorithms.md#cost)).
- **Cm and Em** follow the paper in the native core, so their results can differ from
  the Python version's; neither is consistently better at k ≥ 4. At k = 3 and n ≥ 50 the
  native results were 3-10% lower in cost.
- **Fm** in the native core is a tabu search that always runs all `max_iter` steps,
  while the Python version stops at the first plateau. It is therefore slower for k ≥ 4,
  but finds 1-7% lower costs. Lower `max_iter` to trade quality for time.
- The native Bm, Cm, Em and Fm also run their neighbourhoods (and Em its trials) in
  parallel when more than one thread is available; these numbers are single-threaded.

## Solution quality of the search methods

ILS and memetic (see [Algorithms](algorithms.md)) against Cm and Fm, at equal wall-time
budgets. Setup: the same 144-core server, kap 0.3.0a1 at commit 1192d61, 16 threads per
run (`RAYON_NUM_THREADS=16`, 16 pinned cores, 8 runs side by side), int64 costs,
minimisation, equal partite sizes. Two instance families after Karapetyan and Gutin:
*random* (every pair cost a uniform integer in 1..100) and *geometric* (rounded Euclidean
distances between uniform random points). 5 seeds per size for random, 2 for geometric.
Script: `develop/bench_search.py`; results in `develop/bench_search_server4.csv`.

The optima are not known, so each number is the **mean % above the best cost any method
found for that instance** (0.00 means the method found the best-seen cost on every seed).
"ILS × 16" is `n_trials=16`; ILS and memetic use `time_limit`; Cm and Fm run with their
defaults and take the time shown.

Random instances:

| k | n | Cm | Fm | ILS, 1 s | ILS × 16, 1 s | memetic, 1 s | ILS, 60 s | ILS × 16, 60 s | memetic, 60 s |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 40 | 18.5 (0.6 ms) | 18.4 (0.6 ms) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | 70 | 24.5 (1.5 ms) | 24.5 (1.4 ms) | 2.72 | 0.72 | 2.91 | 0.33 | 0.00 | 0.29 |
| 3 | 100 | 21.2 (2.8 ms) | 21.2 (2.8 ms) | 4.51 | 2.39 | 5.99 | 0.77 | 0.04 | 3.25 |
| 3 | 150 | 20.7 (6.4 ms) | 20.9 (6.5 ms) | 5.55 | 3.24 | 7.32 | 0.63 | 0.06 | 4.70 |
| 4 | 30 | 16.4 (1.8 ms) | 5.2 (28 ms) | 0.90 | 0.01 | 0.35 | 0.00 | 0.00 | 0.35 |
| 4 | 50 | 24.9 (3.0 ms) | 18.1 (51 ms) | 3.30 | 2.55 | 5.13 | 0.79 | 0.11 | 0.51 |
| 5 | 18 | 10.4 (4.2 ms) | 3.0 (96 ms) | 0.11 | 0.00 | 0.03 | 0.00 | 0.00 | 0.00 |
| 5 | 25 | 13.6 (6.8 ms) | 10.2 (120 ms) | 0.70 | 0.16 | 0.10 | 0.09 | 0.00 | 0.00 |
| 6 | 12 | - | - | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 18 | - | - | 0.53 | 0.09 | 0.13 | 0.00 | 0.00 | 0.04 |
| 8 | 10 | - | - | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| mean | | 18.8 | 15.2 | 1.67 | 0.83 | 2.00 | 0.24 | 0.02 | 0.83 |

Cm and Fm were not run for k ≥ 6, so their means cover k ≤ 5. At a 10 s budget the means
are 0.53 (ILS), 0.25 (ILS × 16) and 1.07 (memetic).

- **ILS finds much lower costs than the multiple Hungarian methods on random costs**: within
  a second its cost is 9-17 % below Cm's.
- **More chains help more than more time**: 16 chains for 1 s (0.83) are close to one chain
  for 10 s (0.53), and 16 chains for 60 s found the best-seen cost almost everywhere.
- **Memetic does not beat multi-chain ILS here.** It is close for k ≥ 4 and clearly behind
  for k = 3 with large n. Using sDV instead of 1DV inside either method changed the 60 s means
  by less than 0.1.
- **On geometric instances all methods are close**: ILS and memetic found the same cost on
  every instance at every budget (k = 3, n = 70 and 150; k = 4, n = 50; k = 6, n = 18), and
  Cm was within 0.33 % of it. The differences above are specific to non-metric costs.
- These are relative comparisons on synthetic instances with equal partite sizes, not gaps
  to the optimum.
