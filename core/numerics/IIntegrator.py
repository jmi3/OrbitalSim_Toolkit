import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable


class IIntegrator(ABC):
    """Interface for integrators"""

    @abstractmethod
    def next_step(self, y: np.ndarray) -> np.ndarray:
        """
        Takes in the state, steps it using whatever method implemented

        :param y: state
        :type y: np.ndarray

        :return: Next state
        :rtype: ndarray
        """
        pass

    @abstractmethod
    def integrate(self, y0: np.ndarray, t_max: float) -> np.ndarray:
        pass
