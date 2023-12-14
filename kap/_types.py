from typing import Sequence, Union

import numpy as np

Real = Union[int, float]
sr1 = Sequence[Real]
sr2 = Sequence[sr1]
sr3 = Sequence[sr2]

SequenceNDReals = Union[sr1, sr2, sr3]

VectorLike = Union[np.ndarray, sr1]
Array2DLike = Union[np.ndarray, sr2]
Array3DLike = Union[np.ndarray, sr3]
ArrayLike = Union[np.ndarray, SequenceNDReals]
