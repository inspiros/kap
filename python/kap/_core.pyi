import numpy as np
import numpy.typing as npt

__version__: str

def linear_sum_assignment(
    cost: np.ndarray, maximize: bool = False
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...
