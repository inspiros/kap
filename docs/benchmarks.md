# Benchmarks

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
