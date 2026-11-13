# Algorithms

`kap.multiple_hungarian` exposes Am–Fm and Shift. The six Am–Fm methods come from Gabrovšek et al. (2020) and build the solution by
**merging parts**. Solving the linear assignment problem (LAP) between
parts `i` and `j` and contracting each matched pair into one vertex gives
the *quotient graph* `G / M_ij`, which has one part fewer. The cost row of
a merged vertex is the sum of the rows of its two members, so the cost of the
final matches is the sum of all pairwise edges inside each match (the clique
cost). After k − 1 merges a single part remains, and each of its vertices
is a match.

Am, Bm and Dm differ in which pairs they merge. Cm, Em and Fm start from Bm
and improve the result by local search. Their neighborhood is the same:
take the current solution, fix the pairwise matching it induces between two
parts `i` and `j`, contract, and re-solve the remaining (k − 1)-part
problem with Bm. A solution has k(k − 1)/2 such neighbors.

| Algo | Strategy |
|---|---|
| Am | Merge the first two parts, in input order, until one is left. k − 1 LAPs. |
| Bm | At each level try every pair, solve the rest recursively with Bm, and keep the pair whose final cost is lowest. |
| Dm | Greedy: at each level merge the pair whose LAP is cheapest. |
| Cm | Steepest descent from Bm: evaluate all neighbors of the current solution, move to the best one while it improves. Deterministic. `max_iter` caps the moves. |
| Em | First improvement from Bm: visit the neighbors in random order and move to the first that improves. `n_trials` independent randomized runs; the best is kept. `max_iter` caps the moves per run. |
| Fm | Tabu search from Bm: at each of `max_iter` steps move to a random best neighbor, even if it is worse, never back to the current or previous solution. Returns the best solution seen. |

The paper's experiments (k = 3, 4; n up to 100) rank solution quality as
Fm ≈ Em ≈ Cm > Bm > Dm ≈ Am. It recommends Em overall (about 0.1 % above the
optimum on benchmarks with known optima) and Cm among the deterministic
methods.

## Shift

For each of the `k` cyclic orders of the parts, Shift constructs one Am
solution and keeps the first best on score ties. This uses `k(k − 1)` LAPs. Under clique-sum
costs, Shift is equivalent to Shift-ROM of Gutin, Goldengorin and Huang and
of Karapetyan and Gutin.

## Dimensionwise variation (DV)

DV re-matching exactly re-solves the assignment between a chosen subset of
parts and its complement. Each move is nonworsening; the two sides may
have unequal sizes, so incomplete matches are supported. 1DV re-matches one
part against the remaining parts, cycling through them; each sweep uses
`k` LAPs (`k = 2` needs one). sDV visits canonical subsets, choosing one of
each subset/complement pair. It uses cyclic first improvement: apply an
improving move immediately and continue the sweep. A full sweep considers
`2^(k−1) − 1` subsets; sDV is limited to `k ≤ 16`. `max_iter` caps DV
sweeps, and a complete cycle without an accepted move ends the search.

`kap.local_search(costs, start="Am", neighborhood="1dv")` builds a start and applies DV.
The dispatcher option `refine` can also apply 1DV or sDV after the main `algo` finishes. Shift, 1DV and
sDV as `algo` choices, and `refine`, require `backend="kap"` (the native core).

References: H.-J. Bandelt, A. Maas and F. C. R. Spieksma, “Local search for
multidimensional assignment problems with decomposable costs,” *European
Journal of Operational Research* 158, 2004; Karapetyan and Gutin,
“Local search heuristics for the multidimensional assignment problem,” *Journal of Heuristics* 17, 2011.

## Iterated local search (ILS)

`kap.ils` (or `algo="ils"` in the dispatcher) starts with Am followed by DV to convergence. Each step perturbs
some nodes of the current solution, runs DV to convergence again, and moves
to the candidate only if its score strictly improves. After 50 consecutive
steps without an improvement it moves to the latest candidate anyway, to
leave a region where it is stuck; the best solution seen is what is returned.
ILS is the default `algo` with `backend="kap"`. The perturbation chooses
`ceil(N/25) + 1` distinct nodes, clamped to `2..N`, where `N` is the number of
solution nodes. It fixes the first largest original part and independently
shuffles every other part's entries, including absent vertices, among the
chosen nodes. This preserves a valid partition for unequal sizes. With fewer
than two nodes, ILS returns the start.

`max_iter` caps perturbations per chain (zero returns the start), `n_trials`
selects independent chains from that same start, and `random_state` seeds them.
Chains run in parallel in the native Python core, or in Rust with the `rayon`
feature; the first best chain wins ties. `refine` selects the neighborhood
inside ILS: 1DV by default, or sDV with `refine="sdv"` (`k <= 16`). ILS requires
`backend="kap"`.

Optional `time_limit` is a finite positive budget in seconds (an explicit
`max_iter` also applies; without one the iteration limit is unlimited), checked before
each perturbation across all chains, after computing the full start solution.
An in-progress DV descent finishes even if the deadline passes. Only iteration
budgets are reproducible; a time limit makes results depend on machine speed.

In the [benchmarks](benchmarks.md#solution-quality-of-the-search-methods), ILS with several chains gave the lowest costs of the methods tested.

This is the “Chain” method of Karapetyan and Gutin, “Local search heuristics for the multidimensional assignment problem,” *Journal of Heuristics* 17, 2011.

## Memetic algorithm

`kap.memetic` (or `algo="memetic"` in the dispatcher) maintains a population of DV local optima. It starts from
Shift followed by DV, mutates members by swapping non-anchor coordinates,
and crosses parents while preserving their common cliques. Crossover repairs
duplicate and missing vertices, including absent vertices for unequal sizes.
Each new candidate passes through DV; selection keeps the distinct best
solutions and preserves the elite. `refine` selects 1DV (default) or sDV
inside the algorithm; `n_trials` is ignored. This requires `backend="kap"`.

Without `time_limit`, `population` sets the target size (default 12, at least
2), and `max_iter` caps generations; zero returns the best first-generation
member. An integer `random_state` makes this mode reproducible, including
across parallel execution. With a finite positive `time_limit` in seconds,
the deadline counts from entry, and population size is estimated from the
budget and measured local-search time unless explicitly supplied. An explicit generation limit also applies alongside the time limit; without
one the generation limit is unlimited. An in-progress batch or generation finishes after
the deadline, so time budgets make results depend on machine speed.

In the [benchmarks](benchmarks.md#solution-quality-of-the-search-methods), memetic did not beat ILS with several chains at equal time.

kap starts from Shift instead of the paper's Greedy construction and includes
the unperturbed local optimum in the first generation. Reference: Gutin and
Karapetyan, “A memetic algorithm for the multidimensional assignment problem,”
and Karapetyan and Gutin, “A New Approach to Population Sizing for Memetic
Algorithms: A Case Study for the Multidimensional Assignment Problem,”
*Evolutionary Computation* 19, 2011,
[arXiv:1003.4314](https://arxiv.org/abs/1003.4314).

## Cost

LAP calls made by one run of Bm, counting the intermediate levels too. The
native core memoizes merges that commute (merging {0,1} then {2,3} reaches the
same quotient as {2,3} then {0,1}), so it solves fewer LAPs than the Python
implementation:

| k | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| LAPs, native | 6 | 39 | 310 | 3 000 | 34 566 | 463 981 |
| LAPs, Python (`backend` other than `"kap"`) | 6 | 42 | 430 | 6 465 | 135 786 | 3 802 036 |

Cm, Em and Fm run Bm once per neighbor per step, so they cost a multiple of
this. Beyond about k = 6, use Am or Dm.

## Differences from the paper

- The paper assumes all parts have the same size. kap also accepts
  unequal sizes; the vertices left over then form incomplete matches.
- The paper breaks ties at random; kap's Bm takes the first best pair.
- With `backend="kap"` (native, default), Cm, Em and Fm follow the paper as
  in the table above. Other backends run the Python implementation, which keeps
  kap 0.2.3's behavior: Cm updates the current solution during a pass, and Fm
  accepts only equal-or-better moves.

## References

- B. Gabrovšek, T. Novak, J. Povh, D. Rupnik Poklukar, and J. Žerovnik, “Multiple Hungarian Method for k-Assignment Problem,” *Mathematics* 8(11), 2050, 2020. [doi:10.3390/math8112050](https://doi.org/10.3390/math8112050). Methods Am–Fm.
- H.-J. Bandelt, A. Maas, and F. C. R. Spieksma, “Local search for multidimensional assignment problems with decomposable costs,” *European Journal of Operational Research* 158, 2004. Dimensionwise variation.
- D. Karapetyan and G. Gutin, “Local search heuristics for the multidimensional assignment problem,” *Journal of Heuristics* 17(3), 201–249, 2011. [arXiv:0806.3258](https://arxiv.org/abs/0806.3258). 1DV, sDV, and the Chain method used by ILS.
- D. Karapetyan, G. Gutin, and B. Goldengorin, “Empirical evaluation of construction heuristics for the multidimensional assignment problem,” *Text in Algorithmics* 11, 107–122, 2009. [arXiv:0906.2960](https://arxiv.org/abs/0906.2960). Shift-ROM.
- G. Gutin and D. Karapetyan, “A memetic algorithm for the multidimensional assignment problem.” Memetic search.
- D. Karapetyan and G. Gutin, “A New Approach to Population Sizing for Memetic Algorithms: A Case Study for the Multidimensional Assignment Problem,” *Evolutionary Computation* 19(3), 345–371, 2011. [arXiv:1003.4314](https://arxiv.org/abs/1003.4314). Memetic population sizing.
