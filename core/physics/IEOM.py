from abc import ABC, abstractmethod

import numpy as np


class IEOM(ABC):
    @abstractmethod
    def __call__(self, t: float, state: np.ndarray) -> np.ndarray:
        r"""
        Compute the time derivative of the state at time t.
        ```
        \partial_t \Phi = IEOM(t, state)
        ```

        Args:
            t (float): Current time.
            state (np.ndarray): Current state vector.

        Returns:
            np.ndarray: Time derivative of the state vector.
        """
        pass
