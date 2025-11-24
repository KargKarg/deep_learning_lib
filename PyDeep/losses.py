import numpy as np
from abc import ABC, abstractmethod



class Loss(ABC):
    """
    """

    def __init__(self, d_in: int | None, d_out: int | None) -> None:
        """
        """
        super().__init__()
        self.d_in, self.d_out = d_in, d_out
        return None

    
    abstractmethod
    def forward(self, x: np.ndarray[float], y: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
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

    def __init__(self, d_in: int | None = None, d_out: int | None = None) -> None:
        """
        """
        super().__init__(d_in, d_out)
        return None
    
    
    def forward(self, x: np.ndarray[float], y: np.ndarray[float], track: bool = True) -> float:
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
    

    def backward(self) -> np.ndarray[float]:
        """
        dim(grad_out_in) = (N, d_in)
        """
        assert all(hasattr(self, attr) for attr in ("N", "x", "y", "value")), "The forward pass tracked must be computed before the gradient."
        self.grad_out_in: np.ndarray[float] = ((1/self.N)*(1/self.y.shape[1])*(-2)*(self.y - self.x))
        return self.grad_out_in
    

class BCE(Loss):
    """
    """


    def __init__(self, d_in: int | None = None, d_out: int | None = None) -> None:
        """
        """
        super().__init__(d_in, d_out)
        return None
    
    
    def forward(self, x: np.ndarray[float], y: np.ndarray[float], track: bool = True) -> float:
        """
        """

        match track:

            case True:

                self.N: int = x.shape[0]
                self.x: np.ndarray[float] = np.copy(x)
                self.y: np.ndarray[float] = np.copy(y)
                self.value: np.ndarray[float] = np.mean(np.sum(-(
                    y*np.log(x + 1e-5) + (1 - y)*np.log(1 - x + 1e-5)
                    ), axis=1))

                return self.value
            
            case False:

                return np.mean(np.sum(-(
                    y*np.log(x + 1e-5) + (1 - y)*np.log(1 - x + 1e-5)
                    ), axis=1))
            
    
    def backward(self) -> np.ndarray[float]:
        """
        dim(grad_out_in) = (N, d_in)
        """
        assert all(hasattr(self, attr) for attr in ("N", "x", "y", "value")), "The forward pass tracked must be computed before the gradient."
        
        self.grad_out_in: np.ndarray[float] = - (
            (self.y/(self.x + 1e-5)) - ((1 - self.y)/(1 - self.x + 1e-5))
        )

        return self.grad_out_in
    

class CE(Loss):
    """
    """

    def __init__(self, d_in: int | None = None, d_out: int | None = None) -> None:
        """
        """
        super().__init__(d_in, d_out)
        return None
    

    def forward(self, x: np.ndarray[float], y: np.ndarray[float], track = True) -> float:
        """
        """

        match track:

            case True:

                self.N: int = x.shape[0]
                self.x: np.ndarray[float] = np.copy(x)
                self.y: np.ndarray[float] = np.copy(y)
                self.value: np.ndarray[float] = np.mean(np.sum(-y*np.log(x + 1e-5), axis=1))

                return self.value
            
            case False:

                return np.mean(np.sum(-y*np.log(x + 1e-5), axis=1))
            

    def backward(self) -> np.ndarray[float]:
        """
        dim(grad_out_in) = (N, d_in)
        """
        assert all(hasattr(self, attr) for attr in ("N", "x", "y", "value")), "The forward pass tracked must be computed before the gradient."
        
        self.grad_out_in: np.ndarray[float] = -(self.y/self.x)

        return self.grad_out_in
    

class LogCE(Loss):
    """
    """

    def __init__(self, d_in: int | None = None, d_out: int | None = None) -> None:
        """
        """
        super().__init__(d_in, d_out)
        return None
    

    def forward(self, x: np.ndarray[float], y: np.ndarray[float], track = True) -> float:
        """
        """

        match track:

            case True:

                self.N: int = x.shape[0]
                self.x: np.ndarray[float] = np.copy(x)
                self.y: np.ndarray[float] = np.copy(y)
                self.value: np.ndarray[float] = np.mean(np.sum(-y*x, axis=1))

                return self.value
            
            case False:

                return np.mean(np.sum(-y*x, axis=1))
            

    def backward(self) -> np.ndarray[float]:
        """
        dim(grad_out_in) = (N, d_in)
        """
        assert all(hasattr(self, attr) for attr in ("N", "x", "y", "value")), "The forward pass tracked must be computed before the gradient."
        
        self.grad_out_in: np.ndarray[float] = -self.y

        return self.grad_out_in


class SummedLoss(Loss):
    """
    """

    def __init__(self, *modules) -> None:
        """
        """
        self.modules: list[Loss] = [module[1] for module in modules]
        self.weights: list[float] = [module[0] for module in modules]
        super().__init__(sum([module.d_in for module in self.modules]), 1)
        return None

    
    def forward(self, x: np.ndarray[float], y: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        """
        assert x.shape[-1] == sum([module.d_in for module in self.modules]), ""
        return np.sum([weight*module(x[:, (dim_past := sum([m.d_in for m in self.modules[:i]])) : dim_past + module.d_in], y[:, dim_past : dim_past + module.d_in], track) for i, (module, weight) in enumerate(zip(self.modules, self.weights))])/sum(self.weights)
    

    def backward(self) -> np.ndarray[float]:
        """
        """
        return np.concatenate([(weight/sum(self.weights))*module.backward() for module, weight in zip(self.modules, self.weights)], axis=1)
