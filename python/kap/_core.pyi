import numpy as np
import numpy.typing as npt

__version__: str

def linear_sum_assignment(
    cost: np.ndarray, maximize: bool = False
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...


def k_assignment(
    cost_matrices: list[npt.NDArray[np.float64]] | list[npt.NDArray[np.int64]],
    algo: str,
    maximize: bool = False,
    max_iter: int = 100,
    n_trials: int = 1,
    seed: int = 0,
    refine: str | None = None,
    time_limit: float | None = None,
    population: int | None = None,
) -> tuple[list[list[tuple[int, int]]], list[float] | list[int], list[tuple[int, int]]]: ...
