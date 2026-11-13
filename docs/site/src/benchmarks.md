# Benchmarks

Measurements include the [speed of the native core](#native-core-against-the-python-implementation)
against the Python implementation, and the [solution quality](#solution-quality-of-the-search-methods)
of ILS and memetic with complete and optional matching, plus comparisons with proven optima and LAP backend timings.

The tables identify the measured source revision. They precede the later LAP
integer-overflow safeguards, which add arithmetic checks; they do not measure
that subsequent implementation.

## Native core against the Python implementation

`kap.k_assignment` with the native core (`backend="kap"`, the default) against the
Python implementation of the same algorithms (`kap._reference.k_assignment`, which
runs the multiple Hungarian methods in Python but calls the same native LAP solver).
kap 0.2.x used other LAP backends, which are not measured here.

Setup: 4 × Intel Xeon E7-8860 v4 at 2.20 GHz (72 cores, 144 threads),
503 GB RAM, kap 0.3.0a1 from d1877e0 with the explicit-float threshold validation
change, Python 3.12.16, and NumPy 2.5.4. Eight workers ran concurrently, each
pinned to one physical core, with `RAYON_NUM_THREADS=1`. Other users were active.
Costs are independent uniform integers in [0, 100) plus uniform noise (float64),
in n × n pair matrices; `max_iter=100`, `n_trials=1`, and no threshold.
Each cell reports the median time across three seeds; short calls use the minimum
of up to five repeats within each seed. Costs are means across those seeds.

| Algo | k | n | Native | Python | Speedup | Cost, native / Python |
|---|---|---|---|---|---|---|
| Am | 6 | 10 | 0.104 ms | 1.73 ms | 16.59× | = |
| Am | 6 | 100 | 1.81 ms | 7.22 ms | 4.00× | = |
| Am | 6 | 500 | 62.8 ms | 82.2 ms | 1.31× | = |
| Dm | 8 | 10 | 0.558 ms | 29.9 ms | 53.64× | = |
| Dm | 8 | 100 | 24.4 ms | 149 ms | 6.12× | = |
| Dm | 8 | 500 | 882 ms | 2.23 s | 2.52× | = |
| Bm | 4 | 100 | 8.39 ms | 56 ms | 6.68× | = |
| Bm | 6 | 10 | 10.5 ms | 1.2 s | 114.48× | = |
| Bm | 6 | 100 | 679 ms | 8.69 s | 12.79× | = |
| Bm | 6 | 200 | 2.89 s | 20.4 s | 7.04× | = |
| Cm | 4 | 100 | 19.5 ms | 158 ms | 8.13× | 1.005 |
| Cm | 5 | 10 | 5.37 ms | 242 ms | 45.13× | 0.990 |
| Cm | 5 | 100 | 296 ms | 1.64 s | 5.53× | 0.997 |
| Em | 4 | 100 | 19.6 ms | 113 ms | 5.76× | 0.999 |
| Em | 5 | 100 | 238 ms | 1.75 s | 7.36× | 0.997 |
| Fm | 3 | 100 | 4.45 ms | 46.7 ms | 10.49× | 0.931 |
| Fm | 4 | 100 | 620 ms | 151 ms | 0.24× | 0.950 |
| Fm | 5 | 100 | 7.71 s | 2.3 s | 0.30× | 0.988 |

`=` denotes equal mean objective costs. Am, Bm, and Dm had equal native/reference
mean costs at every measured size and dtype. Cm, Em, and Fm can differ because
the native implementation follows different search rules from the Python reference.

- Removing Python quotient-graph construction has the largest effect on small n
  and larger k: the displayed Bm k=6,n=10 speedup is about 114×. For Am at n=500,
  LAP work dominates and the displayed speedup is 1.31×.
- Native Bm also avoids repeated merges by memoizing equivalent merge forests;
  see [algorithm costs](algorithms.md#cost).
- Native Fm runs all 100 tabu iterations, while the Python reference can stop
  at a plateau. At k=4/5 it is slower here and returns lower mean costs.
- Separate int64 workloads, without the uniform noise, ran up to 2.50× faster
  than the float64 workloads for n≥50. This comparison changes both the dtype
  and the cost structure, so it does not isolate an arithmetic-only speedup.
- Bm, Cm, Em, and Fm can use parallel execution, but each run here used one thread.

## Solution quality of the search methods

This reruns the retained laptop comparison grid on 4 × Intel Xeon E7-8860 v4
at 2.20 GHz (72 cores, 144 threads), with Python 3.12.16, NumPy 2.5.4,
and kap 0.3.0a1 from d1877e0 with the explicit-float threshold validation change.
Each instance used 16 physical cores on one socket (`RAYON_NUM_THREADS=16`),
with four instances on disjoint core groups. Other users were active; the
machine was not reserved exclusively for these measurements.

Part sizes are equal and `threshold=None`. Random costs are independent uniform
integers in 1..100. Geometric costs are rounded Euclidean distances between
uniform points in [0, 1000]². Matrices use int64. There are three seeds per random
setting and one per geometric setting: 23 instances and 350 measured calls.

Values are mean percentages above the lowest cost observed for that instance
across all measured methods and budgets. These are not gaps to a known optimum.
The reference includes Shift, 1DV, Shift+sDV, Cm, Em with 16 trials, Fm, and
ILS/memetic with both 1DV and sDV at 1 s and 10 s. The table shows a subset:
ILS and memetic use 1DV; ILS × 16 uses 16 independent chains. Cm/Fm use their
default iteration limits; their parentheses report mean elapsed milliseconds.
Cm, Em, and Fm were omitted for k≥6.

| Kind | k | n | Seeds | Cm | Fm | ILS, 1 s | ILS × 16, 1 s | memetic, 1 s | ILS, 10 s | ILS × 16, 10 s | memetic, 10 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 3 | 40 | 3 | 15.88 (0.6 ms) | 15.88 (0.5 ms) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| random | 3 | 100 | 3 | 23.03 (2.6 ms) | 23.03 (2.5 ms) | 4.69 | 0.71 | 5.13 | 1.29 | 0.00 | 3.54 |
| random | 4 | 30 | 3 | 15.17 (2.3 ms) | 4.51 (20.4 ms) | 0.63 | 0.01 | 0.16 | 0.00 | 0.00 | 0.18 |
| random | 4 | 50 | 3 | 24.47 (3.1 ms) | 17.24 (45.0 ms) | 3.10 | 1.72 | 4.05 | 0.84 | 0.42 | 0.67 |
| random | 5 | 25 | 3 | 14.16 (8.7 ms) | 11.01 (107.9 ms) | 0.81 | 0.09 | 0.11 | 0.30 | 0.00 | 0.00 |
| random | 6 | 18 | 3 | — | — | 0.57 | 0.07 | 0.07 | 0.12 | 0.00 | 0.00 |
| random | 8 | 10 | 3 | — | — | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| geometric | 3 | 70 | 1 | 0.20 (1.9 ms) | 0.20 (1.9 ms) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| geometric | 4 | 50 | 1 | 0.09 (3.8 ms) | 0.09 (85.6 ms) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

At 10 s, 16-chain ILS had mean excess at most 0.42% in this grid. Memetic's
largest mean excess was 3.54% at random k=3,n=100. All displayed ILS/memetic
configurations reached the same cost on the two geometric settings. These
complete-matching results do not predict the optional-matching results below.

Time limits are checked between operations, so elapsed time can exceed the
requested budget. The raw measurements retain elapsed time for each call.

## Optional matching: ILS and memetic

Measured on 4 × Intel Xeon E7-8860 v4 at 2.20 GHz (72 cores, 144 threads),
Python 3.12.16, NumPy 2.5.4, and kap 0.3.0a1 from commit d1877e0 with the
explicit-float threshold validation change. Each run used 16 physical cores on
one socket and `RAYON_NUM_THREADS=16`; four instances ran concurrently on disjoint
core groups. Other users were active, so cache and memory bandwidth were shared.

These synthetic instances test unequal part sizes, forbidden pairs, and
[optional matching](optional-matching.md). The base size `n` gives part sizes
0.8n, n, 1.2n, repeated in that order. All methods used 1DV, minimization, and
float64 matrices so forbidden pairs could be represented by `+inf`.

- **Random:** independent uniform integer costs in 1..100, threshold 30, and
  20% of pairs forbidden independently.
- **Geometric:** rounded Euclidean distances between uniform points in
  [0, 1000]², threshold 150. Distances above 250 and 5% of the remaining pairs
  selected independently are forbidden; about 85% of all pairs are forbidden.

There are three seeds per setting, except n=1000 with one seed per family:
32 instances and 192 method/budget calls. Every output was checked for vertex
coverage, uniqueness, part membership, forbidden pairs, reported original costs,
and a shifted objective no worse than the all-singleton partition.

Let J be the shifted objective, the sum of (cost − threshold) within cliques.
The table reports mean `100 × (J − J_best) / abs(J_best)`, where J_best is the
lowest objective any measured method found at either budget on that instance.
All J_best values here are negative. Zero denotes the best observed value;
these are not gaps to a proven optimum. Original costs alone would omit the
threshold's dependence on the number of matched pairs.

| Kind | k | n | Seeds | ILS, 1 s | ILS × 16, 1 s | memetic, 1 s | ILS, 10 s | ILS × 16, 10 s | memetic, 10 s |
|---|---|---|---|---|---|---|---|---|---|
| geometric | 3 | 100 | 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| geometric | 3 | 300 | 3 | 0.05 | 0.01 | 0.03 | 0.00 | 0.00 | 0.00 |
| geometric | 3 | 1000 | 1 | 0.02 | 0.02 | 0.03 | 0.00 | 0.00 | 0.00 |
| geometric | 5 | 100 | 3 | 0.18 | 0.02 | 0.03 | 0.02 | 0.00 | 0.00 |
| geometric | 5 | 300 | 3 | 0.44 | 0.39 | 0.26 | 0.15 | 0.11 | 0.00 |
| geometric | 8 | 100 | 3 | 0.84 | 0.49 | 0.79 | 0.21 | 0.07 | 0.01 |
| random | 3 | 100 | 3 | 0.65 | 0.23 | 0.23 | 0.16 | 0.04 | 0.00 |
| random | 3 | 300 | 3 | 0.87 | 0.78 | 0.73 | 0.29 | 0.21 | 0.14 |
| random | 3 | 1000 | 1 | 0.37 | 0.43 | 0.54 | 0.00 | 0.21 | 0.01 |
| random | 5 | 100 | 3 | 4.55 | 3.00 | 2.39 | 2.60 | 1.20 | 0.00 |
| random | 5 | 300 | 3 | 3.49 | 3.70 | 3.90 | 2.07 | 1.54 | 0.39 |
| random | 8 | 100 | 3 | 8.97 | 6.80 | 6.40 | 5.34 | 3.33 | 0.00 |

At 10 s, memetic had a lower shifted objective than 16-chain ILS on 18 instances,
tied it on 9, and had a higher objective on 5. This differs from the complete-matching
comparison above; method selection depends on the instance family and objective.

Actual elapsed time includes initial construction and the final in-progress
operation. The next table reports the **maximum elapsed seconds across seeds**
for each method and requested budget. In particular, the 1 s request for
16-chain ILS took up to 3.83 s on the largest random instance.

| Kind | k | n | ILS, 1 s | ILS × 16, 1 s | memetic, 1 s | ILS, 10 s | ILS × 16, 10 s | memetic, 10 s |
|---|---|---|---|---|---|---|---|---|
| geometric | 3 | 100 | 1.00 | 1.01 | 1.03 | 10.00 | 10.01 | 10.07 |
| geometric | 3 | 300 | 1.06 | 1.06 | 1.06 | 10.04 | 10.05 | 10.08 |
| geometric | 3 | 1000 | 1.84 | 2.32 | 1.07 | 10.71 | 11.57 | 10.21 |
| geometric | 5 | 100 | 1.02 | 1.03 | 1.03 | 10.01 | 10.01 | 10.06 |
| geometric | 5 | 300 | 1.13 | 1.18 | 1.08 | 10.17 | 10.18 | 10.08 |
| geometric | 8 | 100 | 1.05 | 1.05 | 1.01 | 10.08 | 10.05 | 10.07 |
| random | 3 | 100 | 1.01 | 1.01 | 1.02 | 10.01 | 10.01 | 10.03 |
| random | 3 | 300 | 1.12 | 1.13 | 1.06 | 10.10 | 10.11 | 10.04 |
| random | 3 | 1000 | 1.73 | 3.83 | 1.55 | 10.85 | 12.40 | 10.73 |
| random | 5 | 100 | 1.04 | 1.02 | 1.02 | 10.02 | 10.02 | 10.04 |
| random | 5 | 300 | 1.20 | 1.22 | 1.10 | 10.20 | 10.21 | 10.08 |
| random | 8 | 100 | 1.09 | 1.07 | 1.04 | 10.07 | 10.06 | 10.03 |

## ILS and the lower bound against the proven optimum

The same 33 instances previously measured on an AMD Ryzen 7 7840H were rerun
on 4 × Intel Xeon E7-8860 v4 at 2.20 GHz (72 cores, 144 threads), using
Python 3.12.16, NumPy 2.5.4, **SciPy 1.18.1** (HiGHS through `scipy.optimize.milp`),
and kap 0.3.0a1 from d1877e0 with the explicit-float threshold validation change.
There are three seeds per size. Random and geometric costs follow the definitions
above, with int64 matrices, equal sizes, minimization, and no threshold.

Four instances ran concurrently on disjoint 16-physical-core groups, one per
socket. Rayon used 16 threads. The benchmark explicitly forwarded `threads=1`
to HiGHS for exact solving; ILS used one or eight chains as shown. Other users
were active. Exact elapsed time includes model construction and solving but
excludes the initial SciPy import. The solve limit was 300 s.

All 33 exact solves proved optimality, and every optimum agreed with the retained
laptop result. ILS columns show mean percentage above the optimum, followed by
optimum hits / seeds. The lower-bound column shows mean percentage below the
optimum. Exact time is the maximum across the three seeds, in seconds.

| Kind | k | n | Exact time max (s) | ILS, 100 iterations | ILS, 1 s, 1 chain | ILS, 1 s, 8 chains | lower_bound |
|---|---|---|---|---|---|---|---|
| random | 3 | 10 | 0.09 | 0.00 (3/3) | 0.00 (3/3) | 0.00 (3/3) | 0.88 |
| random | 3 | 20 | 1.39 | 1.84 (0/3) | 0.00 (3/3) | 0.00 (3/3) | 1.00 |
| random | 3 | 30 | 9.27 | 2.29 (0/3) | 0.00 (3/3) | 0.00 (3/3) | 1.82 |
| random | 3 | 40 | 40.50 | 1.91 (0/3) | 0.00 (3/3) | 0.00 (3/3) | 2.39 |
| random | 4 | 8 | 0.82 | 0.00 (3/3) | 0.00 (3/3) | 0.00 (3/3) | 13.71 |
| random | 4 | 12 | 14.54 | 0.76 (2/3) | 0.00 (3/3) | 0.00 (3/3) | 16.50 |
| random | 4 | 16 | 64.33 | 1.94 (0/3) | 0.00 (3/3) | 0.00 (3/3) | 20.30 |
| random | 5 | 8 | 24.51 | 0.43 (1/3) | 0.00 (3/3) | 0.00 (3/3) | 19.79 |
| random | 6 | 6 | 71.35 | 0.00 (3/3) | 0.00 (3/3) | 0.00 (3/3) | 18.78 |
| geometric | 3 | 30 | 3.42 | 0.00 (3/3) | 0.00 (3/3) | 0.00 (3/3) | 0.01 |
| geometric | 4 | 16 | 28.64 | 0.00 (3/3) | 0.00 (3/3) | 0.00 (3/3) | 0.59 |

Both 1 s ILS configurations reached the proven optimum on all 33 instances.
The default 100 iterations reached it on 18. The bound was within 2.39% for
these k=3 cases and within 0.59% for the geometric cases; the largest mean
random-instance gap was 20.30% at k=4.

An initial run with HiGHS's default thread count created 72 solver threads per
worker despite the 16-core affinity limit. It was stopped and retained separately;
its partial results are excluded from this table. The table uses the explicit
single-thread HiGHS configuration described above.

A separate one-seed prototype run at k=6,n=8 (262,144 variables) returned no
incumbent after 626 s despite a requested 120 s limit. That size exceeds the
public `kap.exact` default guard of 150,000 variables and is not included in the
table. The result illustrates why the guard and time-limit caveat remain.

## LAP backends

| Backend | Algorithm | Rectangular | Notes |
|---|---|---|---|
| `kap` | Jonker-Volgenant initialization (square) + shortest augmenting path (Crouse 2016) | yes | default; native, built in; optimal like `scipy` (ties may differ on square input) |
| `scipy` | modified Jonker-Volgenant | yes | |
| `lap` | Jonker-Volgenant | yes | |
| `lapjv` | Jonker-Volgenant | no | see measurements below |
| `munkres` | Hungarian | yes | pure Python, slower in these measurements |

Measured time per call in µs on 4 × Intel Xeon E7-8860 v4 at 2.20 GHz
(72 cores, 144 threads), Python 3.12.16, NumPy 2.5.4, kap 0.3.0a1 from
d1877e0 with the explicit-float threshold validation change, **SciPy 1.18.1**,
lap 0.5.14, and lapjv 1.3.29. Each worker was pinned to one physical core;
four input cells ran concurrently on separate sockets. Other users were active.

Uniform inputs are float64 values in [0, 1). `int100` contains integer-valued
float64 costs in [0, 100), with many ties. Here the input labeled `geometric`
is the structured cost `i*j + uniform noise`; it differs from the Euclidean
instances used in the search sections. The full grid also includes integer-valued
costs in [0, 10⁶), and both n × n and n × 2n shapes.

Each solver is warmed up, then timing uses the minimum of five repeated samples,
with repetition counts calibrated to about 0.3 s total for short calls. Before
timing, assignments are checked for cardinality, uniqueness, index range, and
objective agreement with SciPy (`rtol=1e-12`, `atol=1e-8`). Inputs are generated
from a separate deterministic seed for each distribution/size/shape cell.

| Input | n | kap | SciPy 1.18.1 | lap 0.5.14 | lapjv 1.3.29 |
|---|---|---|---|---|---|
| uniform | 10 | 1.83 | 2.13 | 2.48 | 2.24 |
| uniform | 100 | 84.68 | 299.18 | 93.92 | 52.91 |
| uniform | 300 | 2 166.93 | 3 835.16 | 2 614.57 | 1 663.76 |
| uniform | 500 | 4 546.05 | 11 694.46 | 4 330.73 | 2 631.76 |
| uniform | 1000 | 20 851.80 | 54 183.01 | 47 246.04 | 34 966.02 |
| uniform | 2000 | 87 057.39 | 288 626.79 | 339 781.58 | 253 528.86 |
| int100 | 1000 | 7 788.47 | 26 058.69 | 5 476.26 | 5 780.74 |
| geometric | 1000 | 657 132.46 | 773 656.64 | 2 321 726.41 | cost mismatch |
| uniform, n × 2n | 1000 | 20 078.80 | 19 147.33 | 29 458.68 | — |

Kap is about 2.6–3.3 times faster than SciPy on the displayed uniform square
inputs at n=1000 and n=2000. At n=1000 with tied costs, kap is about 1.35–1.42 times slower than
lap/lapjv. For the displayed uniform rectangular case, kap takes about 1.05 times
SciPy's time. These ratios describe these inputs and versions.

`cost mismatch` means a feasible assignment had a higher cost than SciPy's and
failed the objective-agreement check. With lapjv 1.3.29, the difference was
0.02962184 at structured n=1000 and 2.5415404 at n=2000. Kap and lap agreed
with SciPy on those inputs. These lapjv timings are omitted from equivalent-solution
comparisons. The discrepancy was reproduced in the second run at n=1000.

The benchmark runner was also checked on identical serial and four-worker grids
(84 solver/input cells, n=10,100,300). Input hashes and returned objectives matched
exactly. The median parallel/serial timing ratio was 1.00, with a range of
0.85–1.06; timing equality is not expected on shared hardware.
