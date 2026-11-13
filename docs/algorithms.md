# Algorithms

The six Am–Fm methods come from Gabrovšek et al. (2020) and build the solution by
**merging partites**. Solving the linear assignment problem (LAP) between
partites `i` and `j` and contracting each matched pair into one vertex gives
the *quotient graph* `G / M_ij`, which has one partite fewer. The cost row of
a merged vertex is the sum of the rows of its two members, so the cost of the
final matches is the sum of all pairwise edges inside each match (the clique
cost). After k − 1 merges a single partite remains, and each of its vertices
is a match.

Am, Bm and Dm differ in which pairs they merge. Cm, Em and Fm start from Bm
and improve the result by local search. Their neighbourhood is the same:
take the current solution, fix the pairwise matching it induces between two
partites `i` and `j`, contract, and re-solve the remaining (k − 1)-partite
problem with Bm. A solution has k(k − 1)/2 such neighbours.

| Algo | Strategy |
|---|---|
| Am | Merge the first two partites, in input order, until one is left. k − 1 LAPs. |
| Bm | At each level try every pair, solve the rest recursively with Bm, and keep the pair whose final cost is lowest. |
| Dm | Greedy: at each level merge the pair whose LAP is cheapest. |
| Cm | Steepest descent from Bm: evaluate all neighbours of the current solution, move to the best one while it improves. Deterministic. `max_iter` caps the moves. |
| Em | First improvement from Bm: visit the neighbours in random order and move to the first that improves. `n_trials` independent randomised runs; the best is kept. `max_iter` caps the moves per run. |
| Fm | Tabu search from Bm: at each of `max_iter` steps move to a random best neighbour, even if it is worse, never back to the current or previous solution. Returns the best solution seen. |

The paper's experiments (k = 3, 4; n up to 100) rank solution quality as
Fm ≈ Em ≈ Cm > Bm > Dm ≈ Am. It recommends Em overall (about 0.1 % above the
optimum on benchmarks with known optima) and Cm among the deterministic
methods.

## Shift

For each of the `k` cyclic orders of the partites, Shift constructs one Am
solution and keeps the first best on score ties. This uses `k(k − 1)` LAPs. Under clique-sum
costs, Shift is equivalent to Shift-ROM of Gutin, Goldengorin and Huang and
of Karapetyan and Gutin.

## Dimensionwise variation (DV)

DV re-matching exactly re-solves the assignment between a chosen subset of
partites and its complement. Each move is nonworsening; the two sides may
have unequal sizes, so incomplete matches are supported. 1DV re-matches one
partite against the remaining partites, cycling through them; each sweep uses
`k` LAPs (`k = 2` needs one). sDV visits canonical subsets, choosing one of
each subset/complement pair. It uses cyclic first improvement: apply an
improving move immediately and continue the sweep. A full sweep considers
`2^(k−1) − 1` subsets; sDV is limited to `k ≤ 16`. `max_iter` caps DV
sweeps, and a complete cycle without an accepted move ends the search.

`refine` can apply 1DV or sDV after the main `algo` finishes. Shift, 1DV and
sDV as `algo` choices, and `refine`, require `backend="kap"` (the native core).

References: H.-J. Bandelt, A. Maas and F. C. R. Spieksma, “Local search for
multidimensional assignment problems with decomposable costs,” *European
Journal of Operational Research* 158, 2004; Karapetyan and Gutin,
“Efficient local search algorithms for known and new neighborhoods for the
multidimensional assignment problem,” *Journal of Heuristics* 17, 2011.

## Iterated local search (ILS)

`algo="ils"` starts with Am followed by DV to convergence. Each step perturbs
some nodes of the incumbent, runs DV to convergence again, and keeps the
candidate only if its score strictly improves. The perturbation chooses
`ceil(N/25) + 1` distinct nodes, clamped to `2..N`, where `N` is the number of
solution nodes. It fixes the first largest original partite and independently
shuffles every other partite's entries, including absent vertices, among the
chosen nodes. This preserves a valid partition for unequal sizes. With fewer
than two nodes, ILS returns the start.

`max_iter` caps perturbations per chain (zero returns the start), `n_trials`
selects independent chains from that same start, and `random_state` seeds them.
Chains run in parallel in the native Python core, or in Rust with the `rayon`
feature; the first best chain wins ties. `refine` selects the neighbourhood
inside ILS: 1DV by default, or sDV with `refine="sdv"` (`k <= 16`). ILS requires
`backend="kap"`.

Optional `time_limit` is a finite positive budget in seconds, checked before
each perturbation across all chains, after computing the full start solution.
An in-progress DV descent finishes even if the deadline passes. Only iteration
budgets are reproducible; a time limit makes results depend on machine speed.

This is the “Chain” method of Karapetyan and Gutin, “Efficient local search
algorithms for known and new neighborhoods for the multidimensional assignment
problem,” *Journal of Heuristics* 17, 2011.

## Cost

LAP calls made by one run of Bm, counting the intermediate levels too. The
native core memoises merges that commute (merging {0,1} then {2,3} reaches the
same quotient as {2,3} then {0,1}), so it solves fewer LAPs than the Python
implementation:

| k | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| LAPs, native | 6 | 39 | 310 | 3 000 | 34 566 | 463 981 |
| LAPs, Python (`backend` other than `"kap"`) | 6 | 42 | 430 | 6 465 | 135 786 | 3 802 036 |

Cm, Em and Fm run Bm once per neighbour per step, so they cost a multiple of
this. Beyond about k = 6, use Am or Dm.

## Differences from the paper

- The paper assumes all partites have the same size. kap also accepts
  unequal sizes; the vertices left over then form incomplete matches.
- The paper breaks ties at random; kap's Bm takes the first best pair.
- With `backend="kap"` (native, default), Cm, Em and Fm follow the paper as
  in the table above. Other backends run the Python implementation, which keeps
  kap 0.2.3's behaviour: Cm updates the current solution during a pass, and Fm
  accepts only equal-or-better moves.

## Reference

B. Gabrovšek, T. Novak, J. Povh, D. Rupnik Poklukar, J. Žerovnik,
"Multiple Hungarian Method for k-Assignment Problem", *Mathematics* 8(11),
2050, 2020. doi:10.3390/math8112050
