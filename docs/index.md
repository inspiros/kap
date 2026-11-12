# kap documentation

`kap` solves the **k-assignment problem** (k-partite graph matching): given
k vertex sets and a cost matrix for every pair of sets, group the vertices
into k-tuples (at most one vertex per set) so that the total cost is
minimal. A tuple's cost is the sum of the pairwise costs among all of its
members.

`kap` implements the six *multiple Hungarian* heuristics of
Gabrovšek (2020), which reduce the problem to a sequence of linear
assignment problems (LAPs).

- [Installation](installation.md)
- [Usage](usage.md): `linear_assignment` and `k_assignment`
- [Algorithms](algorithms.md): what Am to Fm do and what they cost
- [Backends](backends.md): the LAP solvers kap can call
- [Benchmarks](benchmarks.md): native core vs the Python implementation

## Reference

B. Gabrovšek, T. Novak, J. Povh, D. Rupnik Poklukar, J. Žerovnik,
"Multiple Hungarian Method for k-Assignment Problem",
*Mathematics* 8(11), 2050, 2020. doi:10.3390/math8112050
