# Algorithms

All six methods come from Gabrovšek et al. (2020) and build the solution by
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
| Cm | Steepest descent from Bm: move to the best neighbour while it improves. Deterministic. |
| Em | First improvement from Bm: visit the neighbours in random order and move to the first that improves. `n_trials` runs it several times and keeps the best. |
| Fm | Plateau search from Bm: at each step move to a random best neighbour, even if it only ties, without stepping straight back. `max_iter` caps the number of steps. |

The paper's experiments (k = 3, 4; n up to 100) rank solution quality as
Fm ≈ Em ≈ Cm > Bm > Dm ≈ Am. It recommends Em overall (about 0.1 % above the
optimum on benchmarks with known optima) and Cm among the deterministic
methods.

## Cost

LAP calls made by one run of Bm, counting the intermediate levels too:

| k | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| LAPs | 6 | 42 | 430 | 6 465 | 135 786 | 3 802 036 |

Cm, Em and Fm run Bm once per neighbour per step, so they cost a multiple of
this. Beyond about k = 6, use Am or Dm.

## Differences from the paper

- The paper assumes all partites have the same size. kap also accepts
  unequal sizes; the vertices left over then form incomplete matches.
- The paper breaks ties at random; kap's Bm takes the first best pair.
- Known issues in kap 0.2.x, to be fixed: Em's `n_trials` has no effect, and
  Fm lacks the paper's rule against stepping back, so it keeps returning to
  the same solution until `max_iter`.

## Reference

B. Gabrovšek, T. Novak, J. Povh, D. Rupnik Poklukar, J. Žerovnik,
"Multiple Hungarian Method for k-Assignment Problem", *Mathematics* 8(11),
2050, 2020. doi:10.3390/math8112050
