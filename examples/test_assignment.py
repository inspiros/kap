import numpy as np

from kap import linear_assignment, k_assignment
from kap.assignment import _condensed_to_square


def test_linear_assignment():
    print("\n[Linear Assignment Problem]")
    cost = np.array([
        [9, 6, 4],
        [4, 4, 6],
        [3, 9, 2]
    ])
    print("Cost matrix:")
    print(cost)

    matching_result = linear_assignment(cost, return_free=True)
    print("Result:", matching_result)


def test_k_assignment():
    print("\n[k-Assignment Problem]")
    n_nodes = [6, 5, 7]
    n_partites = len(n_nodes)
    # np.random.seed(1)
    cost_matrices = []
    for i in range(n_partites):
        for j in range(i + 1, n_partites):
            cost_matrices.append(np.random.randint(0, 11, (n_nodes[i], n_nodes[j])))

    # Example in paper
    # cost_matrices = [
    #     np.array([
    #         [9, 6, 4],
    #         [4, 4, 6],
    #         [3, 9, 2]
    #     ]),
    #     np.array([
    #         [8, 2, 9],
    #         [5, 0, 4],
    #         [8, 7, 4]
    #     ]),
    #     np.array([
    #         [4, 0, 9],
    #         [9, 9, 6],
    #         [8, 9, 5]
    #     ]),
    # ]
    print("Cost matrices:")
    for index, cost in enumerate(cost_matrices):
        print("- partites", _condensed_to_square(index, n_partites))
        print(cost)
    print()

    print("--- Am ---")
    matching_result = k_assignment(cost_matrices,
                                   algo="Am",
                                   return_free=True)
    print("Result:", matching_result)
    print("Total cost:", sum(matching_result[1]))

    print("--- Bm ---")
    matching_result = k_assignment(cost_matrices,
                                   algo="Bm",
                                   return_free=True)
    print("Result:", matching_result)
    print("Total cost:", sum(matching_result[1]))

    print("--- Cm ---")
    matching_result = k_assignment(cost_matrices,
                                   algo="Cm",
                                   return_free=True)
    print("Result:", matching_result)
    print("Total cost:", sum(matching_result[1]))

    print("--- Dm ---")
    matching_result = k_assignment(cost_matrices,
                                   algo="Dm",
                                   return_free=True)
    print("Result:", matching_result)
    print("Total cost:", sum(matching_result[1]))

    print("--- Em ---")
    matching_result = k_assignment(cost_matrices,
                                   algo="Em",
                                   n_trials=10,
                                   return_free=True)
    print("Result:", matching_result)
    print("Total cost:", sum(matching_result[1]))

    print("--- Fm ---")
    matching_result = k_assignment(cost_matrices,
                                   algo="Fm",
                                   return_free=True)
    print("Result:", matching_result)
    print("Total cost:", sum(matching_result[1]))


if __name__ == "__main__":
    test_linear_assignment()
    test_k_assignment()
