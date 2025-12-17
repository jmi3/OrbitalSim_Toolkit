import numpy as np
from abc import ABC, abstractmethod


class IState(ABC):
    @abstractmethod
    def to_array(self) -> np.ndarray:
        """
        Return array form of state.

        Returns:
            np.ndarray: Array representation of the state.
        """
        pass

    @staticmethod
    @abstractmethod
    def from_array(arr: np.ndarray) -> "IState":
        """
        Create state from array.
        Args:
            arr (np.ndarray): Array representation of the state.
        """
        pass
