import numpy as np
from abc import ABC, abstractmethod



class Loss(ABC):
    """
    """

    def __init__(self, d_in: int) -> None:
        """
        """

        super().__init__()

        self.d_in, self.d_out, = d_in, 1

        return None

    
    abstractmethod
    def forward(self, x: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        """
        raise NotImplementedError

    
    @abstractmethod
    def backward(self) -> None:
        """
        """
        raise NotImplementedError
    

    def __call__(self, x: np.ndarray[float], y: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        """
        return self.forward(x, y, track)


class MSE(Loss):
    """
    """

    def __init__(self, d_in: int) -> None:
        """
        """
        
        super().__init__(d_in)

        return None
    
    
    def forward(self, x: np.ndarray[float], y: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        """

        match track:

            case True:

                self.N: int = x.shape[0]
                self.x: np.ndarray[float] = np.copy(x)
                self.y: np.ndarray[float] = np.copy(y)
                self.value: np.ndarray[float] = np.mean((y - x)**2)

                return self.value
            
            case False:

                return np.mean((y - x)**2)
    

    def backward(self) -> None:
        """
        dim(grad_out_in) = (N, d_in)
        """
        assert all(hasattr(self, attr) for attr in ("N", "x", "y", "value")), "The forward pass tracked must be computed before the gradient."
        
        self.grad_out_in: np.ndarray[float] = ((1/self.N)*(1/self.d_in)*(-2)*(self.y - self.x))
        return self.grad_out_in