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
) -> tuple[list[list[tuple[int, int]]], list[float] | list[int], list[tuple[int, int]]]: ...
