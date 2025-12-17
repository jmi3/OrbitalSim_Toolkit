import numpy as np
from abc import ABC, abstractmethod
from typing import List

from core.physics.states.IState import IState

class IStateVisualizer(ABC):
    @abstractmethod
    def visualize(self, state: IState) -> None:
        """
        Visualize the given state.
        
        Args:
            state (np.ndarray): The state to visualize.
        """
        pass
    
    @abstractmethod
    def update(self, state: IState) -> None:
        """
        Update the visualization with the new state.
        
        Args:
            state (np.ndarray): The new state to visualize.
        """
        pass
    
    @abstractmethod
    def animate(self, states: List[IState], interval: float) -> None:
        """
        Animate a sequence of states.
        
        Args:
            states (np.ndarray): Sequence of states to animate.
            interval (float): Time interval between frames in the animation.
        """
        pass