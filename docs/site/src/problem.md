# Problem definition


Let V₀, …, Vₖ₋₁ be disjoint parts. For each i &lt; j, Cᵢⱼ[u,v] gives the cost of pairing vertex u in Vᵢ with vertex v in Vⱼ. A solution partitions the vertices into N = maxᵢ |Vᵢ| cliques, counting singletons, with at most one vertex from each part in each clique.

For a clique Q, define its cost as:

<p><code>cost(Q) = Σ<sub>i&lt;j; (i,u),(j,v)∈Q</sub> Cᵢⱼ[u,v]</code></p>

The objective is:

<p><code>minimize Σ<sub>Q in the partition</sub> cost(Q)</code></p>

Use maximization for weights instead of costs. Equal-size parts give full k-vertex cliques; unequal sizes can leave parts absent from a clique or produce free singleton vertices.

With `threshold=tau`, [optional matching](optional-matching.md) instead optimizes the sum of (cost − tau) over pairs sharing a clique, over all partitions into cliques with at most one vertex per part, singletons allowed; returned costs stay in original units.

See [input and results](input.md) for matrix order and the result representation.

## Reference

B. Gabrovšek, T. Novak, J. Povh, D. Rupnik Poklukar, and J. Žerovnik, “Multiple Hungarian Method for k-Assignment Problem,” *Mathematics* 8(11), 2050, 2020. [doi:10.3390/math8112050](https://doi.org/10.3390/math8112050).
