import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Iterator


class Module(ABC):
    """
    """

    def __init__(self, d_in: int, d_out: int, seed: int = 12) -> None:
        """
        """

        super().__init__()

        self.d_in, self.d_out, self.seed = d_in, d_out, seed


    @abstractmethod
    def forward(self, x: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        """
        raise NotImplementedError

    
    @abstractmethod
    def backward(self) -> None:
        """
        """
        raise NotImplementedError


    @abstractmethod
    def grad_loss_parameters(self) -> None:
        """
        """
        raise NotImplementedError


    def __call__(self, x: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        """
        return self.forward(x, track)
    

class Sequential(Module):
    """
    """

    def __init__(self, *modules) -> None:
        """
        """

        self.modules = [module for module in modules]

        super().__init__(self.modules[0].d_in, self.modules[-1].d_out)

        return None
    

    def forward(self, x: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        """
        input = x

        for module in self.modules:
            input = module.forward(input, track)

        return input
    

    def backward(self, grad_loss_out: np.ndarray[float]) -> np.ndarray[float]:
        """
        """
        deltas = grad_loss_out
        for module in self.modules[::-1]:
            deltas = module.backward(deltas)
        return deltas
    

    def grad_loss_parameters(self) -> list[dict[str: np.ndarray[float] |float] | None]:
        """
        """
        return [module.grad_loss_parameters() for module in self.modules[::-1]]


    def __iter__(self) -> Iterator[Module]:
        """
        """
        for module in self.modules:
            yield module

    
    def __getitem__(self, index: int | slice) -> Module:
        """
        """
        return self.modules[index]


    @property
    def dimensions(self) -> tuple[int, int]:
        """
        """
        return (self.modules[0].d_in, self.modules[-1].d_out)
    
    
class Linear(Module):
    """
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = True) -> None:
        """"
        """

        super().__init__(d_in, d_out)
        np.random.seed(self.seed)

        self.has_bias = bias

        match bias:

            case True:

                self.parameters = {
                    "W": np.random.normal(size=(d_in, d_out)),
                    "b": np.random.normal(size=(1,))[0]
                }

            case False:

                self.parameters = {
                    "W": np.random.normal(size=(d_in, d_out)),
                    "b": 0
                }

        return None
    

    def forward(self, x: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        dim(X) = (N, d_in)
        dim(W) = (d_in, d_out)
        """

        match track:

            case True:

                self.N: int = x.shape[0]
                self.x: np.ndarray[float] = np.copy(x)
                self.value: np.ndarray[float] = np.dot(x, self.parameters["W"]) + self.parameters["b"]

                return self.value
            
            case False:

                return np.dot(x, self.parameters["W"]) + self.parameters["b"]


    def backward(self, grad_loss_out: np.ndarray[float]) -> None:
        """
        dim(grad_out_in) = (N, d_in, d_out)
        dim(grad_out_W) = (N, d_in, d_out, d_out)
        dim(grad_out_bias) = (N, d_out)
        dim(grad_loss_out) = (N, d_out)
        """
        assert all(hasattr(self, attr) for attr in ("N", "x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = np.tile(np.copy(self.parameters["W"]), (self.N, 1, 1))

        self.grad_out_W: np.ndarray[float] = self.x[..., None, None] * np.eye(self.d_out)
        self.grad_out_bias: np.ndarray[float] = np.ones(shape=(self.N, self.d_out))

        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        self.grad_loss_out = np.copy(grad_loss_out)

        return self.grad_loss_in


    def grad_loss_parameters(self) -> dict[str: np.ndarray[float] | float]:
        """
        """
        return {
            "W": np.mean(np.sum((self.grad_out_W*self.grad_loss_out[:, None, None, :]), axis=3), axis=0),
            "b": int(self.has_bias)*np.mean(np.sum(self.grad_loss_out*self.grad_out_bias, axis=1), axis=0)
        }


class Sigmoid(Module):
    """
    """

    def __init__(self, d_in: int) -> None:
        """"
        """
        super().__init__(d_in, d_in)
        self.sigm: Callable[[np.ndarray[float]], np.ndarray[float]] = lambda x: 1 / (1 + np.exp(-x))
        return None
    

    def forward(self, x: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        dim(X) = (N, d_in)
        """

        match track:

            case True:

                self.N: int = x.shape[0]
                self.x: np.ndarray[float] = np.copy(x)
                self.value: np.ndarray[float] = self.sigm(x)

                return self.value
            
            case False:

                return self.sigm(x)


    def backward(self, grad_loss_out: np.ndarray[float]) -> None:
        """
        dim(grad_out_in) = (N, d_in, d_out)
        dim(grad_out_W) = (N, d_in, d_out, d_out)
        dim(grad_out_bias) = (N, d_out)
        dim(grad_loss_out) = (N, d_out)
        """
        assert all(hasattr(self, attr) for attr in ("N", "x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = ((1 - self.sigm(self.x))*self.sigm(self.x))[:, :, None] * np.eye(self.d_in)

        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        return self.grad_loss_in


    def grad_loss_parameters(self) -> None:
        """
        """
        return None


class Tanh(Module):
    """
    """

    def __init__(self, d_in: int) -> None:
        """"
        """
        super().__init__(d_in, d_in)
        return None
    

    def forward(self, x: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        dim(X) = (N, d_in)
        """

        match track:

            case True:

                self.N: int = x.shape[0]
                self.x: np.ndarray[float] = np.copy(x)
                self.value: np.ndarray[float] = np.tanh(x)

                return self.value
            
            case False:

                return np.tanh(x)


    def backward(self, grad_loss_out: np.ndarray[float]) -> None:
        """
        dim(grad_out_in) = (N, d_in, d_out)
        dim(grad_out_W) = (N, d_in, d_out, d_out)
        dim(grad_out_bias) = (N, d_out)
        dim(grad_loss_out) = (N, d_out)
        """
        assert all(hasattr(self, attr) for attr in ("N", "x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = (1 - self.value**2)[:, :, None] * np.eye(self.d_in)

        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        return self.grad_loss_in


    def grad_loss_parameters(self) -> None:
        """
        """
        return None
    

class Softmax(Module):
    """
    """

    def __init__(self, d_in: int, tau: int = 1) -> None:
        """"
        """
        super().__init__(d_in, d_in)
        self.tau = tau
        return None
    

    def forward(self, x: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        dim(X) = (N, d_in)
        """

        match track:

            case True:

                self.N: int = x.shape[0]
                self.x: np.ndarray[float] = np.copy(x)
                self.value: np.ndarray[float] = np.exp(self.tau * x)/np.sum(np.exp(self.tau * x), axis=-1, keepdims=True)

                return self.value
            
            case False:

                return np.exp(self.tau * x)/np.sum(np.exp(self.tau * x), axis=-1, keepdims=True)


    def backward(self, grad_loss_out: np.ndarray[float]) -> None:
        """
        dim(grad_out_in) = (N, d_in, d_out)
        dim(grad_out_W) = (N, d_in, d_out, d_out)
        dim(grad_out_bias) = (N, d_out)
        dim(grad_loss_out) = (N, d_out)
        """
        assert all(hasattr(self, attr) for attr in ("N", "x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = (self.tau*self.value*(1 - self.value))[:, :, None] * np.eye(self.d_in) + (np.ones(shape=(self.d_in, self.d_in)) - np.eye(self.d_in))*(-self.tau*self.value[:, None, :]*self.value[:, :, None])
        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        return self.grad_loss_in


    def grad_loss_parameters(self) -> None:
        """
        """
        return None