import itertools
import math
from collections import UserList, namedtuple
from typing import List, Sequence, Tuple, Union, Optional

import numpy as np

from ._types import Array2DLike

__all__ = ["linear_assignment", "k_assignment"]

AssignmentResult = namedtuple("AssignmentResult",
                              ["matches", "matching_costs", "free"],
                              defaults=[None, None, None])


def _compute_n_partites(cost_matrices: Sequence[Array2DLike]) -> Optional[int]:
    r"""Return number of partites if number of cost_matrices is valid."""
    if not len(cost_matrices):
        return None
    n_partites = (1 + math.sqrt(1 + 8 * len(cost_matrices))) / 2
    return int(n_partites) if n_partites.is_integer() else None


def _check_partites(p1: int, p2: int, M: Optional[Array2DLike] = None):
    if p1 == p2:
        raise ValueError("No diagonal elements in condensed matrix.")
    if p1 > p2:
        p1, p2 = p2, p1
        if M is not None:
            M = np.asarray(M)[:, ::-1]
    if M is not None:
        return p1, p2, np.asarray(M)
    return p1, p2


def _calc_row_idx(k, n):
    return int(math.ceil((1 / 2.) * (- (-8 * k + 4 * n ** 2 - 4 * n - 7) ** 0.5 + 2 * n - 1) - 1))


def _elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i * (i + 1)) // 2


def _calc_col_idx(k, i, n):
    return int(n - _elem_in_i_rows(i + 1, n) + k)


def _square_to_condensed(p1, p2, n_partites):
    p1, p2 = _check_partites(p1, p2)
    return n_partites * p1 + p2 - p1 * (p1 + 1) // 2 - p1 - 1


def _condensed_to_square(k, n_partites):
    i = _calc_row_idx(k, n_partites)
    j = _calc_col_idx(k, i, n_partites)
    return i, j


class KPartiteNamedNode:
    def __init__(self, partite_indices, node_indices, vertices_cost=None):
        if isinstance(partite_indices, Sequence):
            partite_indices, node_indices = zip(*sorted(zip(partite_indices, node_indices)))

        self.partite_indices = partite_indices
        self.node_indices = node_indices
        self.vertices_cost = vertices_cost

    @property
    def is_quotient(self):
        return isinstance(self.partite_indices, Sequence)

    @property
    def quotient_nodes(self):
        if self.is_quotient:
            return [(p, n) for p, n in zip(self.partite_indices, self.node_indices)]
        return self.partite_indices, self.node_indices

    def associate(self, other, vertices_cost):
        merged_partite_indices = []
        merged_node_indices = []
        if self.is_quotient:
            merged_partite_indices.extend(self.partite_indices)
            merged_node_indices.extend(self.node_indices)
        else:
            merged_partite_indices.append(self.partite_indices)
            merged_node_indices.append(self.node_indices)
        if other.is_quotient:
            merged_partite_indices.extend(other.partite_indices)
            merged_node_indices.extend(other.node_indices)
        else:
            merged_partite_indices.append(other.partite_indices)
            merged_node_indices.append(other.node_indices)

        merged_vertices_cost = vertices_cost
        if merged_vertices_cost is not None:
            if self.vertices_cost is not None:
                merged_vertices_cost += self.vertices_cost
            if other.vertices_cost is not None:
                merged_vertices_cost += other.vertices_cost
        return KPartiteNamedNode(partite_indices=merged_partite_indices,
                                 node_indices=merged_node_indices,
                                 vertices_cost=merged_vertices_cost)

    def __add__(self, other):
        return self.associate(other, None)

    def __repr__(self):
        ret = f"{self.__class__.__name__}(nodes={self.quotient_nodes}"
        if self.vertices_cost is not None:
            ret += f", cost={self.vertices_cost}"
        ret += ")"
        return ret


class KPartiteNamedNodeList(UserList):
    def __getitem__(self, item):
        if isinstance(item, Sequence):
            ret = self.data
            for query in item:
                ret = ret[query]
            return ret
        return self.data[item]


class KPartiteGraph:
    r"""
    k-Partite Graph.
    This class serves as internal struct of k-Assignment solvers and
    does not perform sanity checks on input arguments.
    """

    def __init__(self, cost_matrices, n_partites=None, node_indices=None):
        self.cost_matrices = cost_matrices
        self.n_partites = _compute_n_partites(self.cost_matrices) if n_partites is None else n_partites
        if node_indices is not None:
            self.node_indices = KPartiteNamedNodeList(node_indices)
        else:
            self.node_indices = KPartiteNamedNodeList()
            for pi, n_node in enumerate(self.n_nodes):
                self.node_indices.append([KPartiteNamedNode(pi, _) for _ in range(n_node)])

    def quotient(self, p1: int, p2: int, M12: Array2DLike) -> 'KPartiteGraph':
        p1, p2, M12 = _check_partites(p1, p2, M12)
        n_nodes = self.n_nodes
        free_1 = list(set(range(n_nodes[p1])).difference(M12[:, 0].tolist()))
        free_2 = list(set(range(n_nodes[p2])).difference(M12[:, 1].tolist()))

        n_matches_12 = len(M12)
        n_free_1 = len(free_1)
        n_free_2 = len(free_2)
        n_free_12 = n_free_1 + n_free_2
        n_nodes_12 = n_matches_12 + n_free_12

        def preq_ind(new_ind):
            if new_ind == p1:
                return p1, p2
            elif new_ind < p2:
                return new_ind
            return new_ind + 1

        quotient_cost_matrices = []
        for i in range(self.n_partites - 1):
            for j in range(i + 1, self.n_partites - 1):
                preq_i, preq_j = preq_ind(i), preq_ind(j)
                if preq_i == (p1, p2):
                    cost_1j = self[p1, preq_j]
                    cost_2j = self[p2, preq_j]

                    cost_ij = np.empty((n_nodes_12, n_nodes[preq_j]), dtype=self.dtype)
                    cost_ij[:n_matches_12, :] = cost_1j[M12[:, 0].tolist(), :] + cost_2j[M12[:, 1].tolist(), :]
                    cost_ij[n_matches_12:n_matches_12 + n_free_1, :] = cost_1j[free_1, :]
                    cost_ij[n_matches_12 + n_free_1:, :] = cost_2j[free_2, :]
                    quotient_cost_matrices.append(cost_ij)
                elif preq_j == (p1, p2):
                    cost_i1 = self[preq_i, p1]
                    cost_i2 = self[preq_i, p2]

                    cost_ij = np.empty((n_nodes[preq_i], n_nodes_12), dtype=self.dtype)
                    cost_ij[:, :n_matches_12] = cost_i1[:, M12[:, 0].tolist()] + cost_i2[:, M12[:, 1].tolist()]
                    cost_ij[:, n_matches_12:n_matches_12 + n_free_1] = cost_i1[:, free_1]
                    cost_ij[:, n_matches_12 + n_free_1:] = cost_i2[:, free_2]
                    quotient_cost_matrices.append(cost_ij)
                else:
                    quotient_cost_matrices.append(self[i, j].clone())

        quotient_node_indices = KPartiteNamedNodeList()
        for i in range(self.n_partites - 1):
            preq_i = preq_ind(i)
            if preq_i == (p1, p2):
                node_indices_i = [
                    self.node_indices[p1, k].associate(self.node_indices[p2, l], self[p1, p2][k, l])
                    for k, l in M12]
                node_indices_i.extend([self.node_indices[p1, k] for k in free_1])
                node_indices_i.extend([self.node_indices[p2, k] for k in free_2])
                quotient_node_indices.append(node_indices_i)
            else:
                quotient_node_indices.append(self.node_indices[preq_i].copy())
        return KPartiteGraph(quotient_cost_matrices,
                             n_partites=self.n_partites - 1,
                             node_indices=quotient_node_indices)

    def vertices(self):
        vertices = []
        for node_indices_i in self.node_indices:
            for node in node_indices_i:
                if node.is_quotient:
                    vertices.extend(list(itertools.combinations(node.quotient_nodes, r=2)))
        vertices.sort(key=lambda v: (v[0][0], v[1][0], v[0][1], v[1][1]))
        return vertices

    def vertices_cost(self) -> Optional[float]:
        acc_cost = None
        for node_indices_i in self.node_indices:
            for node in node_indices_i:
                if node.vertices_cost is not None:
                    if acc_cost is None:
                        acc_cost = 0
                    acc_cost += node.vertices_cost
        return acc_cost

    def reconstruct(self, return_free: bool = False) -> AssignmentResult:
        matches, free, matching_costs = [], [] if return_free else None, []
        for node_indices_i in self.node_indices:
            for node in node_indices_i:
                if node.is_quotient:
                    matches.append(node.quotient_nodes)
                    matching_costs.append(node.vertices_cost)
                elif return_free:
                    free.append(node.quotient_nodes)
        # matches = np.array(matches)
        # free = np.array(free)
        # matching_costs = np.array(matching_costs)
        return AssignmentResult(matches, matching_costs, free)

    def reconstruct_from(self, node_indices, return_free: bool = False) -> AssignmentResult:
        matches, free, matching_costs = [], [] if return_free else None, []
        for node_indices_i in node_indices:
            for node in node_indices_i:
                if node.is_quotient:
                    quotient_nodes = node.quotient_nodes
                    for k in range(len(quotient_nodes)):
                        for l in range(k + 1, len(quotient_nodes)):
                            node_k = quotient_nodes[k]
                            node_l = quotient_nodes[l]
                            matches.append([node_k, node_l])
                            matching_costs.append(self[node_k[0], node_l[0]][node_k[1], node_l[1]])
                    # matches.append(node.quotient_nodes)
                elif return_free:
                    free.append(node.quotient_nodes)
        # matches = np.array(matches)
        # free = np.array(free)
        # matching_costs = np.array(matching_costs)
        return AssignmentResult(matches, matching_costs, free)

    def clone(self) -> 'KPartiteGraph':
        return KPartiteGraph(self.cost_matrices, self.n_partites, self.node_indices)

    @property
    def n_nodes(self) -> List[int]:
        return [self[0].shape[0]] + [self[0, pi].shape[1] for pi in range(1, self.n_partites)]

    @property
    def dtype(self):
        return self[0].dtype

    def __getitem__(self, item):
        if isinstance(item, Sequence):
            p1, p2 = item
            cost = self.cost_matrices[_square_to_condensed(p1, p2, self.n_partites)]
            return cost if p1 < p2 else cost.T
        return self.cost_matrices[item]

    def __repr__(self):
        rep = f"{self.__class__.__name__}("
        rep += f"n_partites={self.n_partites}, "
        rep += f"n_vertices={self.n_nodes}"
        rep += ")"
        return rep


# noinspection PyPackageRequirements
def linear_assignment(cost_matrix: Array2DLike,
                      maximize: bool = False,
                      return_free: bool = False,
                      backend: str = "scipy") -> AssignmentResult:
    r"""
    Solve the Linear Sum Assignment Problem. This function is a wrapper around
    backend solvers, which implement the following algorithms:
    - :func:`scipy.optimize.linear_sum_assignment`: modified Jonker-Volgenant
    - :func:`lap.lapjv`: Jonker-Volgenant
    - :func:`lapjv.lapjv`: Jonker-Volgenant
    - :func:`lapsolver.solve_dense`: Jonker-Volgenant
    - :meth:`munkres.Munkres.compute`: Hungarian (or Kuhn-Munkres)

    For a detailed performance comparisons between these solvers, see ref. [1]_.

    Args:
        cost_matrix (ndarray): 2D matrix of costs.
        maximize (bool): Calculates a maximum weight matching if true.
            Defaults to False.
        return_free (bool): Return unmatched vertices if true.
            Defaults to False.
        backend (str): The backend used to solve the linear assignment problem.
            Supported backends are "scipy", "lap", "lapjv", "lapsolver", "munkres".
            Defaults to "scipy".

    Returns:
        matching_results: A namedtuple of ``matches``, ``matching_costs``,
            and optionally ``free`` vertices if ``return_free`` is True.

    Notes:
        This function currently only accepts dense matrix.

    References:
        [1] https://github.com/berhane/LAP-solvers
    """
    error_msg = ("backend \"{backend}\" is selected but not installed. "
                 "Please install with: pip install {backend}")
    if backend == "scipy":
        try:
            from scipy.optimize import linear_sum_assignment
        except ModuleNotFoundError:
            raise ModuleNotFoundError(error_msg.format(backend=backend))
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=maximize)
        matches = np.stack([row_ind, col_ind]).T
    elif backend == "lap":
        try:
            import lap
        except ModuleNotFoundError:
            raise ModuleNotFoundError(error_msg.format(backend=backend))
        row_ind, col_ind = lap.lapjv(cost_matrix if not maximize else -cost_matrix,
                                     extend_cost=True, return_cost=False)
        matches = np.array([[col_ind[i], i] for i in row_ind if i >= 0])
    elif backend == "lapjv":
        try:
            import lapjv
        except ModuleNotFoundError:
            raise ModuleNotFoundError(error_msg.format(backend=backend))
        # cost_matrix must be squared
        row_ind, col_ind, _ = lapjv.lapjv(cost_matrix if not maximize else -cost_matrix)
        matches = np.array([[col_ind[i], i] for i in row_ind])
    elif backend == "lapsolver":
        try:
            from lapsolver import solve_dense
        except ModuleNotFoundError:
            raise ModuleNotFoundError(error_msg.format(backend=backend))
        row_ind, col_ind = solve_dense(cost_matrix if not maximize else -cost_matrix)
        matches = np.stack([row_ind, col_ind]).T
    elif backend == "munkres":
        try:
            from munkres import Munkres
        except ModuleNotFoundError:
            raise ModuleNotFoundError(error_msg.format(backend=backend))
        m = Munkres()
        matches = np.array(m.compute(cost_matrix.copy() if not maximize else -cost_matrix))
    else:
        raise ValueError(f"backend must be either scipy | lap | lapjv | lapsolver | munkres. "
                         f"Got {backend}.")
    matching_costs = cost_matrix[tuple(matches.T.tolist())]

    if return_free:
        free = []
        free += [(0, _) for _ in list(set(range(cost_matrix.shape[0])).difference(matches[:, 0].tolist()))]
        free += [(1, _) for _ in list(set(range(cost_matrix.shape[1])).difference(matches[:, 1].tolist()))]
    else:
        free = None
    return AssignmentResult(matches, matching_costs, free)


# noinspection DuplicatedCode
def solve_Am(G: KPartiteGraph,
             maximize: bool = False,
             matching_filter=None,
             backend: str = "scipy") -> KPartiteGraph:
    Gq = G.clone()
    for _ in range(G.n_partites - 1):
        # construct quotient graph by merging the two first partites
        Gq = Gq.quotient(0, 1, linear_assignment(Gq[0, 1], maximize, matching_filter,
                                                 backend=backend)[0])
    return Gq


# noinspection DuplicatedCode
def solve_Bm(G: KPartiteGraph,
             maximize: bool = False,
             matching_filter=None,
             return_partites_choices: bool = False,
             backend: str = "scipy") -> Union[KPartiteGraph, Tuple[KPartiteGraph, List[Tuple[int, int]]]]:
    best_Gq = None
    best_score = np.inf if not maximize else -np.inf
    best_choices = None

    for i, j in itertools.combinations(range(G.n_partites), r=2):
        choices = [(i, j)]
        Gq = G.quotient(i, j, linear_assignment(G[i, j], maximize, matching_filter,
                                                backend=backend)[0])
        if Gq.n_partites > 1:
            recursive_result = solve_Bm(Gq, maximize, matching_filter,
                                        return_partites_choices=return_partites_choices,
                                        backend=backend)
            if not return_partites_choices:
                Gq = recursive_result
            else:
                Gq, trailing_choices = recursive_result
                choices += trailing_choices
        score = Gq.vertices_cost()
        if (score < best_score and not maximize) or (score > best_score and maximize):
            best_Gq = Gq
            best_score = score
            best_choices = choices
    if not return_partites_choices:
        return best_Gq
    return best_Gq, best_choices


# noinspection DuplicatedCode
def solve_Cm(G: KPartiteGraph,
             maximize: bool = False,
             max_iter: int = 100,
             backend: str = "scipy") -> KPartiteGraph:
    # init with Bm
    Gq, choices = solve_Bm(G, maximize,
                           return_partites_choices=True,
                           backend=backend)
    best_Gq = Gq
    best_score = Gq.vertices_cost()
    best_vertices = Gq.vertices()
    best_first_partites_pair = choices[0]

    # steepest descent
    for iteration in range(max_iter):
        improvement_achieved = False
        for p1, p2 in filter(lambda _: _ != best_first_partites_pair,
                             itertools.combinations(range(G.n_partites), r=2)):
            fixed_vertices = list(filter(lambda _: (_[0][0], _[1][0]) == (p1, p2), best_vertices))
            M = [[_[0][1], _[1][1]] for _ in fixed_vertices]
            Gq = G.quotient(p1, p2, M)
            Gq, choices = solve_Bm(Gq, maximize,
                                   return_partites_choices=True,
                                   backend=backend)
            score = Gq.vertices_cost()
            if (score < best_score and not maximize) or (score > best_score and maximize):
                best_Gq = Gq
                best_score = score
                best_vertices = Gq.vertices()
                best_first_partites_pair = choices[0]
                improvement_achieved = True
        if not improvement_achieved:
            break
    return best_Gq


# noinspection DuplicatedCode
def solve_Dm(G: KPartiteGraph,
             maximize: bool = False,
             return_partites_choices: bool = False,
             backend: str = "scipy") -> Union[KPartiteGraph, Tuple[KPartiteGraph, List[Tuple[int, int]]]]:
    last_Gq = G.clone()
    choices = []
    while last_Gq.n_partites > 1:
        best_Gq = None
        best_score = np.inf if not maximize else -np.inf
        best_choices = None

        for i, j in itertools.combinations(range(last_Gq.n_partites), r=2):
            Gq = last_Gq.quotient(i, j, linear_assignment(last_Gq[i, j], maximize,
                                                          backend=backend)[0])
            score = Gq.vertices_cost()
            if (score < best_score and not maximize) or (score > best_score and maximize):
                best_Gq = Gq
                best_score = score
                best_choices = [(i, j)]
        # continue next iteration from best state
        last_Gq = best_Gq
        choices += best_choices
        del best_Gq, best_score, best_choices
    if not return_partites_choices:
        return last_Gq
    return last_Gq, choices


# noinspection DuplicatedCode
def _Em_loop(G: KPartiteGraph,
             G_init: KPartiteGraph,
             first_partite_pair: Tuple[int, int],
             maximize: bool = False,
             max_iter: int = 100,
             backend: str = "scipy") -> Tuple[KPartiteGraph, float]:
    best_Gq = G_init
    best_score = G_init.vertices_cost()
    best_vertices = G_init.vertices()
    best_first_partites_pair = first_partite_pair

    # undeterministic steepest descent
    for iteration in range(max_iter):
        improvement_achieved = False
        for p1, p2 in np.random.permutation(list(filter(lambda _: _ != best_first_partites_pair,
                                                        itertools.combinations(range(G.n_partites), r=2)))):
            fixed_vertices = list(filter(lambda _: (_[0][0], _[1][0]) == (p1, p2), best_vertices))
            M = [[_[0][1], _[1][1]] for _ in fixed_vertices]
            Gq = G.quotient(p1, p2, M)
            Gq, choices = solve_Bm(Gq, maximize,
                                   return_partites_choices=True,
                                   backend=backend)
            score = Gq.vertices_cost()
            if (score < best_score and not maximize) or (score > best_score and maximize):
                best_Gq = Gq
                best_score = score
                best_vertices = Gq.vertices()
                best_first_partites_pair = choices[0]
                improvement_achieved = True
                break
        if not improvement_achieved:
            break
    return best_Gq, best_score


# noinspection DuplicatedCode
def solve_Em(G: KPartiteGraph,
             maximize: bool = False,
             max_iter: int = 100,
             n_trials: int = 1,
             backend: str = "scipy") -> KPartiteGraph:
    if max_iter <= 0:
        raise ValueError(f"max_iter must be a positive integer. Got {max_iter}.")
    if n_trials <= 0:
        raise ValueError(f"n_trials must be a positive integer. Got {n_trials}.")

    # init with Bm
    Gq, choices = solve_Bm(G, maximize,
                           return_partites_choices=True,
                           backend=backend)
    best_Gq = Gq
    best_score = Gq.vertices_cost()
    best_first_partites_pair = choices[0]

    for ti in range(n_trials):
        Gq, score = _Em_loop(G, Gq, best_first_partites_pair, maximize, max_iter,
                             backend=backend)
        if (score < best_score and not maximize) or (score > best_score and maximize):
            best_Gq = Gq
            best_score = score
    return best_Gq


# noinspection DuplicatedCode
def solve_Fm(G: KPartiteGraph,
             maximize: bool = False,
             max_iter: int = 100,
             backend: str = "scipy") -> KPartiteGraph:
    if max_iter <= 0:
        raise ValueError(f"max_iter must be a positive integer. Got {max_iter}.")

    # init with Bm
    Gq, choices = solve_Bm(G, maximize,
                           return_partites_choices=True,
                           backend=backend)
    best_Gq = None
    best_Gq_list = [Gq]
    best_score = Gq.vertices_cost()
    best_edges_list = [Gq.vertices()]
    best_first_partites_pair_list = [choices[0]]

    # steepest descent
    for iteration in range(max_iter):
        improvement_achieved = False

        rand_ind = np.random.randint(0, len(best_Gq_list))
        best_Gq = best_Gq_list[rand_ind]
        best_edges = best_edges_list[rand_ind]
        best_first_partites_pair = best_first_partites_pair_list[rand_ind]
        best_Gq_list.clear()
        best_edges_list.clear()
        best_first_partites_pair_list.clear()

        for p1, p2 in filter(lambda _: _ != best_first_partites_pair,
                             itertools.combinations(range(G.n_partites), r=2)):
            fixed_vertices = list(filter(lambda _: (_[0][0], _[1][0]) == (p1, p2), best_edges))
            M = [[_[0][1], _[1][1]] for _ in fixed_vertices]
            Gq = G.quotient(p1, p2, M)
            Gq, choices = solve_Bm(Gq, maximize,
                                   return_partites_choices=True,
                                   backend=backend)
            score = Gq.vertices_cost()
            if score == best_score:
                best_Gq_list.append(Gq)
                best_edges_list.append(Gq.vertices())
                best_first_partites_pair_list.append(choices[0])
                improvement_achieved = True
            elif (score < best_score and not maximize) or (score > best_score and maximize):
                best_Gq_list = [Gq]
                best_score = score
                best_edges_list = [Gq.vertices()]
                best_first_partites_pair_list = [choices[0]]
                improvement_achieved = True
        if not improvement_achieved:
            break
    if len(best_Gq_list):
        rand_ind = np.random.randint(0, len(best_Gq_list))
        best_Gq = best_Gq_list[rand_ind]
    return best_Gq


def k_assignment(cost_matrices: Sequence[Array2DLike],
                 algo: str = "Cm",
                 max_iter: int = 100,
                 n_trials: int = 1,
                 maximize: bool = False,
                 return_free: bool = False,
                 backend: str = "scipy") -> AssignmentResult:
    r"""
    Solve the k-Assignment Problem using Gabrovšek's multiple Hungarian methods,
    described in ref. [1]_. This function call the Linear Assignment Problem solver
    multiple times. See :func:`linear_assignment` for details about the available backends.

    Args:
        cost_matrices (sequence of ndarray): List of pairwise cost matrices,
            ordered as in ``itertools.combination(n, 2)``.
        algo (str): One of the 6 named algorithms proposed in the paper:
            "Am", "Bm", "Cm", "Dm", "Em", "Fm". Defaults to "Cm".
        max_iter (int): Maximum number of iterations for Em and Fm.
            This is ignored if other algorithm is used. Defaults to 100.
        n_trials (int): Number of trials for Em.
            This is ignored if other algorithm is used. Defaults to 1.
        maximize (bool): Calculates a maximum weight matching if true.
            Defaults to False.
        return_free (bool): Return unmatched vertices if true.
            Defaults to False.
        backend (str): The backend used to solve the linear assignment problem.
            Supported backends are "scipy", "lap", "lapjv", "lapsolver", "munkres".
            Defaults to "scipy".

    Returns:
        matching_results: A namedtuple of ``matches``, ``matching_costs``,
            and optionally ``free`` vertices if ``return_free`` is True.

    References:
        [1] Multiple Hungarian Method for k-Assignment Problem.
            *Mathematics*, 8(11), 2050, November 2020, :doi:`10.3390/math8112050`
    """
    # check inputs
    if not len(cost_matrices):
        raise ValueError("cost_matrices is empty.")
    if not isinstance(cost_matrices, np.ndarray):
        cost_matrices = [np.asarray(cost_matrix) for cost_matrix in cost_matrices]
    if not all(cost_matrix.ndim == 2 for cost_matrix in cost_matrices):
        raise ValueError("cost_matrices must be sequence of 2D arrays.")
    # check number of cost_matrices
    n_partites = _compute_n_partites(cost_matrices)
    if not n_partites:
        raise ValueError("Invalid number of cost matrices. Expected len(cost_matrices) = n * (n - 1) / 2,"
                         f" where n is number of partites. Got {len(cost_matrices)}.")
    # check dimensions of cost_matrices
    n_vertices = np.full((n_partites,), -1, np.int64)
    for cid, (p0, p1) in enumerate(itertools.combinations(range(n_partites), r=2)):
        if n_vertices[p0] == -1:
            n_vertices[p0] = cost_matrices[cid].shape[0]
        elif n_vertices[p0] != cost_matrices[cid].shape[0]:
            raise ValueError("Number of vertices do not match the dimension of cost matrices. "
                             f"Expected cost_matrices[{cid}].shape[0]={n_vertices[p0]}. Got {n_vertices[p0]}.")
        if n_vertices[p1] == -1:
            n_vertices[p1] = cost_matrices[cid].shape[1]
        elif n_vertices[p1] != cost_matrices[cid].shape[1]:
            raise ValueError("Number of vertices do not match the dimension of cost matrices. "
                             f"Expected cost_matrices[{cid}].shape[1]={n_vertices[p1]}. Got {n_vertices[p1]}.")

    algo = algo.lower()
    if len(algo) == 2 and algo[1] == "m":
        algo = algo[0]

    G = KPartiteGraph(cost_matrices, n_partites)
    if algo == "a":
        Gq = solve_Am(G, maximize,
                      backend=backend)
        return Gq.reconstruct(return_free=return_free)
    elif algo == "b":
        Gq = solve_Bm(G, maximize,
                      backend=backend)
        return Gq.reconstruct(return_free=return_free)
    elif algo == "c":
        Gq = solve_Cm(G, maximize, max_iter,
                      backend=backend)
        return Gq.reconstruct(return_free=return_free)
    elif algo == "d":
        Gq = solve_Dm(G, maximize,
                      backend=backend)
        return Gq.reconstruct(return_free=return_free)
    elif algo == "e":
        Gq = solve_Em(G, maximize, max_iter, n_trials,
                      backend=backend)
        return Gq.reconstruct(return_free=return_free)
    elif algo == "f":
        Gq = solve_Fm(G, maximize, max_iter,
                      backend=backend)
        return Gq.reconstruct(return_free=return_free)
    else:
        raise ValueError("algo must be either Am | Bm | Cm | Dm | Em | Fm "
                         f"(codenames according to the paper). Got {algo}.")
