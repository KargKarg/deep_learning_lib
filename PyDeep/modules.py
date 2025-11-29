import numpy as np
from scipy.linalg import block_diag
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
        return [module.grad_loss_parameters() for module in self.modules]


    def __iter__(self) -> Iterator[Module]:
        """
        """
        for module in self.modules:
            yield module

    
    def __getitem__(self, index: int | slice) -> Module:
        """
        """
        elems = self.modules[index]

        if not isinstance(index, slice):
            return elems
        
        return Sequential(*elems)


    @property
    def dimensions(self) -> tuple[int, int]:
        """
        """
        return (self.modules[0].d_in, self.modules[-1].d_out)
    

class Concat(Module):
    """
    """

    def __init__(self, *modules) -> None:
        """
        """
        self.modules: list[Module] = [module for module in modules]
        super().__init__(sum([module.d_in for module in self.modules]), sum([module.d_out for module in self.modules]))
        return None

    def forward(self, x: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        """
        assert x.shape[-1] == sum([module.d_in for module in self.modules]), ""
        return np.concatenate([module(x[:, (dim_past := sum([m.d_in for m in self.modules[:i]])) : dim_past + module.d_in], track) for i, module in enumerate(self.modules)], axis=1)
    

    def backward(self, grad_loss_out: np.ndarray[float]) -> np.ndarray[float]:
        """
        """
        return np.concatenate([module.backward(grad_loss_out[:, (dim_past := sum([m.d_out for m in self.modules[:i]])) : dim_past + module.d_out]) for i, module in enumerate(self.modules)], axis=1)
    

    def grad_loss_parameters(self) -> list[dict[str: np.ndarray[float] |float] | None]:
        """
        """
        return [module.grad_loss_parameters() for module in self.modules]
    

    def __iter__(self) -> Iterator[Module]:
        """
        """
        for module in self.modules:
            yield module


    def __getitem__(self, index: int | slice) -> Module:
        """
        """
        return self.modules[index]
    

class Linear(Module):
    """
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = True, drop: float = 0.) -> None:
        """"
        """

        super().__init__(d_in, d_out)
        np.random.seed(self.seed)

        self.has_bias = bias
        self.drop = drop

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
        assert all(hasattr(self, attr) for attr in ("x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = np.tile(np.copy(self.parameters["W"]), (self.x.shape[0], 1, 1))

        self.grad_out_W: np.ndarray[float] = self.x[..., None, None] * np.eye(self.d_out)
        self.grad_out_bias: np.ndarray[float] = np.ones(shape=(self.x.shape[0], self.d_out))
        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        self.grad_loss_out = np.copy(grad_loss_out)

        return self.grad_loss_in


    def grad_loss_parameters(self) -> dict[str: np.ndarray[float] | float]:
        """
        """
        dropped_weights = np.tile(np.random.binomial(n=1, p=(1 - self.drop), size=(self.d_out,)), (self.d_in, 1)).copy()
        return {
            "W": np.mean(np.sum((self.grad_out_W*self.grad_loss_out[:, None, None, :]), axis=3), axis=0)*dropped_weights,
            "b": int(self.has_bias)*np.mean(np.sum(self.grad_loss_out*self.grad_out_bias, axis=1), axis=0)
        }


class Embedding(Module):
    """
    """


    def __init__(self, d_in: int, d_out: int, K: int = 1, bias: bool = True, drop: float = 0.) -> None:
        """"
        """
        
        super().__init__(d_in*K, d_out*K)
        np.random.seed(self.seed)

        self.has_bias = bias
        self.drop = drop
        self.K = K

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
        dim(X) = (N, d_in*K)
        dim(W) = (d_in, d_out)
        """
        match track:

            case True:

                self.x: np.ndarray[float] = np.copy(x)
                self.value: np.ndarray[float] = (np.dot(x.reshape(x.shape[0], self.K, self.d_in//self.K), self.parameters["W"]) + self.parameters["b"]).reshape(x.shape[0], self.d_out)

                return self.value.reshape(x.shape[0], self.d_out)
            
            case False:

                return (np.dot(x.reshape(x.shape[0], self.K, self.d_in//self.K), self.parameters["W"]) + self.parameters["b"]).reshape(x.shape[0], self.d_out)


    def backward(self, grad_loss_out: np.ndarray[float]) -> None:
        """
        dim(grad_out_in) = (N, d_in, d_out)
        dim(grad_out_W) = (N, d_in, d_out, d_out)
        dim(grad_out_bias) = (N, d_out)
        dim(grad_loss_out) = (N, d_out)
        """
        assert all(hasattr(self, attr) for attr in ("x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = block_diag(*([self.parameters["W"]] * self.K))

        d_in: int = self.parameters["W"].shape[0]
        d_out: int = self.parameters["W"].shape[1]

        self.grad_out_W: np.ndarray[float] = (self.x.reshape(-1, self.K, d_in)[..., None, None] * np.concatenate([np.concatenate([np.eye(d_out)[None, ...] for _ in range(d_in)], axis=0)[None, ...] for _ in range(self.K)], axis=0)[None, ...]).transpose(0, 2, 3, 1, 4).reshape(-1, d_in, d_out, d_out*self.K)

        self.grad_out_bias: np.ndarray[float] = np.ones(shape=(self.x.shape[0], d_out*self.K))
        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        self.grad_loss_out = np.copy(grad_loss_out)

        return self.grad_loss_in


    def grad_loss_parameters(self) -> dict[str: np.ndarray[float] | float]:
        """
        """
        d_in: int = self.parameters["W"].shape[0]
        d_out: int = self.parameters["W"].shape[1]

        dropped_weights = np.tile(np.random.binomial(n=1, p=(1 - self.drop), size=(d_out,)), (d_in, 1)).copy()
        return {
            "W": np.mean(np.sum((self.grad_out_W*self.grad_loss_out[:, None, None, :]), axis=3), axis=0)*dropped_weights,
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
        assert all(hasattr(self, attr) for attr in ("x", "value")), "The forward pass tracked must be computed before the gradient."

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
        assert all(hasattr(self, attr) for attr in ("x", "value")), "The forward pass tracked must be computed before the gradient."

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

    def __init__(self, d_in: int, tau: float = 1) -> None:
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
        assert all(hasattr(self, attr) for attr in ("x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = (self.tau*self.value*(1 - self.value))[:, :, None] * np.eye(self.d_in) + (np.ones(shape=(self.d_in, self.d_in)) - np.eye(self.d_in))*(-self.tau*self.value[:, None, :]*self.value[:, :, None])
        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        return self.grad_loss_in


    def grad_loss_parameters(self) -> None:
        """
        """
        return None
    


class LogSoftmax(Module):
    """
    """


    def __init__(self, d_in: int, tau: float = 1) -> None:
        """
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

                self.x: np.ndarray[float] = np.copy(x)
                self.value: np.ndarray[float] = self.tau*x - np.log(np.sum(np.exp(self.tau * x), axis=-1, keepdims=True))

                return self.value
            
            case False:

                return self.tau*x - np.log(np.sum(np.exp(self.tau * x), axis=-1, keepdims=True))
    

    def backward(self, grad_loss_out: np.ndarray[float]) -> None:
        """
        dim(grad_out_in) = (N, d_in, d_out)
        dim(grad_out_W) = (N, d_in, d_out, d_out)
        dim(grad_out_bias) = (N, d_out)
        dim(grad_loss_out) = (N, d_out)
        """
        assert all(hasattr(self, attr) for attr in ("x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = ((self.tau * np.eye(self.d_in))[None, :, :] - (self.tau*np.exp(self.tau * self.x)/np.sum(np.exp(self.tau * self.x), axis=-1, keepdims=True))[:, :, None] * np.ones(shape=(self.value.shape[0], 1, self.d_in)))
        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        return self.grad_loss_in


    def grad_loss_parameters(self) -> None:
        """
        """
        return None
    

class Identity(Module):
    """
    """

    def __init__(self, d_in) -> None:
        """
        """
        super().__init__(d_in, d_in)
        return None
    

    def forward(self, x, track = True):
        """
        """

        match track:

            case True:

                self.x: np.ndarray[float] = np.copy(x)
                self.value: np.ndarray[float] = np.copy(x)

                return self.value
            
            case False:

                return x
            
    def backward(self, grad_loss_out: np.ndarray[float]) -> None:
        """
        dim(grad_out_in) = (N, d_in, d_out)
        dim(grad_out_W) = (N, d_in, d_out, d_out)
        dim(grad_out_bias) = (N, d_out)
        dim(grad_loss_out) = (N, d_out)
        """
        assert all(hasattr(self, attr) for attr in ("x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = np.eye(self.d_in)[None, :, :].repeat(self.x.shape[0], axis=0)
        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        return self.grad_loss_in
    
            
    def grad_loss_parameters(self) -> None:
        """
        """
        return None
    

class Duplicate(Module):
    """
    """


    def __init__(self, d_in: int, times: int = 2) -> None:
        """
        """
        super().__init__(d_in, d_in*times)
        self.times = times


    def forward(self, x: np.ndarray[float], track: float = True) -> np.ndarray[float]:
        """
        """
        match track:

            case True:

                self.x: np.ndarray[float] = np.copy(x)
                self.value: np.ndarray[float] = np.repeat(x, self.times, axis=1).copy()

                return self.value
            
            case False:

                return np.repeat(x, self.times, axis=1)
            

    def backward(self, grad_loss_out: np.ndarray[float]) -> None:
        """
        dim(grad_out_in) = (N, d_in, d_out)
        dim(grad_out_W) = (N, d_in, d_out, d_out)
        dim(grad_out_bias) = (N, d_out)
        dim(grad_loss_out) = (N, d_out)
        """
        assert all(hasattr(self, attr) for attr in ("x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = np.concatenate([np.eye(self.d_in) for _ in range(self.times)], axis=1)[None, :, :].repeat(self.x.shape[0], axis=0)
        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        return self.grad_loss_in


    def grad_loss_parameters(self) -> None:
        """
        """
        return None
    

class ReLU(Module):
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

                self.x: np.ndarray[float] = np.copy(x)
                self.value: np.ndarray[float] = np.where(x > 0, x, 0)

                return self.value
            
            case False:

                return np.where(x > 0, x, 0)


    def backward(self, grad_loss_out: np.ndarray[float]) -> None:
        """
        dim(grad_out_in) = (N, d_in, d_out)
        dim(grad_out_W) = (N, d_in, d_out, d_out)
        dim(grad_out_bias) = (N, d_out)
        dim(grad_loss_out) = (N, d_out)
        """
        assert all(hasattr(self, attr) for attr in ("x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = np.where(self.x > 0, 1, 0)[:, :, None] * np.eye(self.d_in)

        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        return self.grad_loss_in


    def grad_loss_parameters(self) -> None:
        """
        """
        return None
    

class LeakyReLU(Module):
    """
    """

    def __init__(self, d_in: int, alpha: float = 0) -> None:
        """"
        """
        super().__init__(d_in, d_in)
        self.alpha = alpha
        return None
    

    def forward(self, x: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        dim(X) = (N, d_in)
        """

        match track:

            case True:

                self.x: np.ndarray[float] = np.copy(x)
                self.value: np.ndarray[float] = np.where(x > 0, x, self.alpha*x)

                return self.value
            
            case False:

                return np.where(x > 0, x, self.alpha*x)


    def backward(self, grad_loss_out: np.ndarray[float]) -> None:
        """
        dim(grad_out_in) = (N, d_in, d_out)
        dim(grad_out_W) = (N, d_in, d_out, d_out)
        dim(grad_out_bias) = (N, d_out)
        dim(grad_loss_out) = (N, d_out)
        """
        assert all(hasattr(self, attr) for attr in ("x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = np.where(self.x > 0, 1, self.alpha)[:, :, None] * np.eye(self.d_in)

        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        return self.grad_loss_in


    def grad_loss_parameters(self) -> None:
        """
        """
        return None
    

class PReLU(Module):
    """
    """

    def __init__(self, d_in: int, alpha: float = 0.) -> None:
        """"
        """
        super().__init__(d_in, d_in)
        self.parameters: dict[str: float] = {"alpha": alpha}
        return None
    

    def forward(self, x: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        dim(X) = (N, d_in)
        """

        match track:

            case True:

                self.x: np.ndarray[float] = np.copy(x)
                self.value: np.ndarray[float] = np.where(x > 0, x, self.parameters["alpha"]*x)

                return self.value
            
            case False:

                return np.where(x > 0, x, self.parameters["alpha"]*x)


    def backward(self, grad_loss_out: np.ndarray[float]) -> None:
        """
        dim(grad_out_in) = (N, d_in, d_out)
        dim(grad_out_W) = (N, d_in, d_out, d_out)
        dim(grad_out_bias) = (N, d_out)
        dim(grad_loss_out) = (N, d_out)
        """
        assert all(hasattr(self, attr) for attr in ("x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = np.where(self.x > 0, 1, self.parameters["alpha"])[:, :, None] * np.eye(self.d_in)

        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        self.grad_out_alpha: np.ndarray[float] = np.where(self.x <= 0, self.x, 0)

        self.grad_loss_out = grad_loss_out.copy()

        return self.grad_loss_in


    def grad_loss_parameters(self) -> dict[str: np.ndarray[float] | float]:
        """
        """
        return {
            "alpha": np.mean(np.sum(self.grad_out_alpha*self.grad_loss_out, axis=1), axis=0)
        }
    

class ELU(Module):
    """
    """

    def __init__(self, d_in: int, alpha: float = 0.) -> None:
        """"
        """
        super().__init__(d_in, d_in)
        self.alpha = alpha
        return None
    

    def forward(self, x: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        dim(X) = (N, d_in)
        """

        match track:

            case True:

                self.x: np.ndarray[float] = np.copy(x)
                self.value: np.ndarray[float] = np.where(x > 0, x, self.alpha*(np.exp(x) - 1))

                return self.value
            
            case False:

                return np.where(x > 0, x, self.alpha*(np.exp(x) - 1))


    def backward(self, grad_loss_out: np.ndarray[float]) -> None:
        """
        dim(grad_out_in) = (N, d_in, d_out)
        dim(grad_out_W) = (N, d_in, d_out, d_out)
        dim(grad_out_bias) = (N, d_out)
        dim(grad_loss_out) = (N, d_out)
        """
        assert all(hasattr(self, attr) for attr in ("x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = np.where(self.x > 0, 1, self.alpha*np.exp(self.x))[:, :, None] * np.eye(self.d_in)

        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        return self.grad_loss_in


    def grad_loss_parameters(self) -> None:
        """
        """
        return None
    

class PELU(Module):
    """
    """

    def __init__(self, d_in: int, alpha: float = 0.) -> None:
        """"
        """
        super().__init__(d_in, d_in)
        self.parameters: dict[str: float] = {"alpha": alpha}
        return None
    

    def forward(self, x: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        dim(X) = (N, d_in)
        """

        match track:

            case True:

                self.x: np.ndarray[float] = np.copy(x)
                self.value: np.ndarray[float] = np.where(x > 0, x, self.parameters["alpha"]*(np.exp(x) - 1))

                return self.value
            
            case False:

                return np.where(x > 0, x, self.parameters["alpha"]*(np.exp(x) - 1))


    def backward(self, grad_loss_out: np.ndarray[float]) -> None:
        """
        dim(grad_out_in) = (N, d_in, d_out)
        dim(grad_out_W) = (N, d_in, d_out, d_out)
        dim(grad_out_bias) = (N, d_out)
        dim(grad_loss_out) = (N, d_out)
        """
        assert all(hasattr(self, attr) for attr in ("x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = np.where(self.x > 0, 1, self.parameters["alpha"]*np.exp(self.x))[:, :, None] * np.eye(self.d_in)

        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        self.grad_out_alpha: np.ndarray[float] = np.where(self.x <= 0, np.exp(self.x) - 1, 0)

        self.grad_loss_out = grad_loss_out.copy()

        return self.grad_loss_in


    def grad_loss_parameters(self) -> dict[str: float]:
        """
        """
        return {
            "alpha": np.mean(np.sum(self.grad_out_alpha*self.grad_loss_out, axis=1), axis=0)
        }
    

class Softplus(Module):
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

                self.x: np.ndarray[float] = np.copy(x)
                self.value: np.ndarray[float] = np.log(1 + np.exp(x))

                return self.value
            
            case False:

                return np.log(1 + np.exp(x))


    def backward(self, grad_loss_out: np.ndarray[float]) -> None:
        """
        dim(grad_out_in) = (N, d_in, d_out)
        dim(grad_out_W) = (N, d_in, d_out, d_out)
        dim(grad_out_bias) = (N, d_out)
        dim(grad_loss_out) = (N, d_out)
        """
        assert all(hasattr(self, attr) for attr in ("x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = self.sigm(self.x)[:, :, None] * np.eye(self.d_in)

        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        return self.grad_loss_in


    def grad_loss_parameters(self) -> None:
        """
        """
        return None
    

class MatMul(Module):
    """
    """

    def __init__(self, N: int, M: int, P: int, left_first: bool = True) -> None:
        """
        """
        self.N, self.M, self.P = N, M, P
        self.left_first = left_first
        super().__init__((N*M) + (M*P), (N*P))
        return None
    

    def forward(self, x: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        dim(X) = (N, d_in)
        """

        match track:

            case True:
                
                self.x = np.copy(x)

                if self.left_first:
                    self.left: np.ndarray[float] = np.copy(x)[:, :self.N*self.M].reshape(-1, self.N, self.M)
                    self.right: np.ndarray[float] = np.copy(x)[:, self.N*self.M:].reshape(-1, self.M, self.P)

                else:
                    self.left: np.ndarray[float] = np.copy(x)[:, self.M*self.P:].reshape(-1, self.N, self.M)
                    self.right: np.ndarray[float] = np.copy(x)[:, :self.M*self.P].reshape(-1, self.M, self.P)

                self.value: np.ndarray[float] = np.matmul(self.left, self.right).reshape(-1, self.N*self.P)

                return self.value
            
            case False:

                return np.matmul(x[:, :self.N*self.M].reshape(-1, self.N, self.M), x[:, self.N*self.M:].reshape(-1, self.M, self.P)).reshape(-1, self.N*self.P)


    def backward(self, grad_loss_out: np.ndarray[float]) -> None:
        """
        dim(grad_out_in) = (N, d_in, d_out)
        dim(grad_out_W) = (N, d_in, d_out, d_out)
        dim(grad_out_bias) = (N, d_out)
        dim(grad_loss_out) = (N, d_out)
        """
        assert all(hasattr(self, attr) for attr in ("x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_left_in: np.ndarray[float] = (self.right[:, None, :, None, :] * np.eye(self.N)[None, :, None, :, None]).reshape(-1, self.N*self.M, self.N*self.P)
        self.grad_out_right_in: np.ndarray[float] = (self.left.transpose(0, 2, 1)[:, :, None, :, None] * np.eye(self.P)[None, None, :, None, :]).reshape(-1, self.M*self.P, self.N*self.P)

        self.grad_out_in: np.ndarray[float] = np.concatenate([self.grad_out_left_in, self.grad_out_right_in], axis=1) if self.left_first else np.concatenate([self.grad_out_right_in, self.grad_out_left_in], axis=1)

        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        return self.grad_loss_in


    def grad_loss_parameters(self) -> None:
        """
        """
        return None
    

class Transposition(Module):
    """
    """

    def __init__(self, N: int, M: int) -> None:
        """
        """
        self.N, self.M = N, M
        super().__init__(N*M, N*M)
        return None
    

    def forward(self, x: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        dim(X) = (N, d_in)
        """

        match track:

            case True:
                
                self.x = np.copy(x)
                self.value: np.ndarray[float] = self.x.reshape(-1, self.N, self.M).transpose(0, 2, 1).reshape(-1, self.M*self.N)

                return self.value
            
            case False:

                return x.reshape(-1, self.N, self.M).transpose(0, 2, 1).reshape(-1, self.M*self.N)
            

    def backward(self, grad_loss_out: np.ndarray[float]) -> None:
        """
        dim(grad_out_in) = (N, d_in, d_out)
        dim(grad_out_W) = (N, d_in, d_out, d_out)
        dim(grad_out_bias) = (N, d_out)
        dim(grad_loss_out) = (N, d_out)
        """
        assert all(hasattr(self, attr) for attr in ("x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = np.eye(self.N*self.M)[None, :, :].repeat(self.x.shape[0], axis=0).reshape(-1, self.N*self.M, self.N, self.M).transpose(0, 1, 3, 2).reshape(-1, self.N*self.M, self.N*self.M)

        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        return self.grad_loss_in
    

    def grad_loss_parameters(self) -> None:
        """
        """
        return None


class ConstMul(Module):
    """
    """

    def __init__(self, d_in: int, b: float) -> None:
        """
        """
        super().__init__(d_in, d_in)
        self.b = b
        return None
    

    def forward(self, x: np.ndarray[float], track: bool = True) -> np.ndarray[float]:
        """
        dim(X) = (N, d_in)
        """

        match track:

            case True:
                
                self.x: np.ndarray[float] = np.copy(x)
                self.value: np.ndarray[float] = self.x*self.b

                return self.value
            
            case False:

                return x*self.b


    def backward(self, grad_loss_out: np.ndarray[float]) -> None:
        """
        dim(grad_out_in) = (N, d_in, d_out)
        dim(grad_out_W) = (N, d_in, d_out, d_out)
        dim(grad_out_bias) = (N, d_out)
        dim(grad_loss_out) = (N, d_out)
        """
        assert all(hasattr(self, attr) for attr in ("x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = np.concatenate([np.eye(self.d_in)[None, ...]*self.b for _ in range(self.x.shape[0])], axis=0)
        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        return self.grad_loss_in


    def grad_loss_parameters(self) -> None:
        """
        """
        return None
    


class Attention(Sequential):
    """
    """

    def __init__(self, L: int, d_in: int, embedding_size: int, bias: bool = True, drop: float = 0.) -> None:
        """
        """
        self.triple: Duplicate = Duplicate(d_in=d_in*L, times=3)

        self.Q: Embedding = Embedding(d_in=d_in, d_out=embedding_size, K=L, bias=bias, drop=drop)
        self.K: Embedding = Embedding(d_in=d_in, d_out=embedding_size, K=L, bias=bias, drop=drop)
        self.V: Embedding = Embedding(d_in=d_in, d_out=embedding_size, K=L, bias=bias, drop=drop)

        super().__init__(
            self.triple,

            Concat(
                self.Q,
                self.K,
                self.V
            ),

            Concat(
                Identity(L*embedding_size),
                Transposition(N=L, M=embedding_size),
                Identity(L*embedding_size)
            ),

            Concat(
                MatMul(N=L, M=embedding_size, P=L),
                Identity(L*embedding_size)
            ),

            Concat(
                ConstMul(L*L, 1/np.sqrt(embedding_size)),
                Identity(L*embedding_size)
            ),

            Concat(
                *[Softmax(L) for _ in range(L)],
                Identity(L*embedding_size)
            ),

            MatMul(N=L, M=L, P=embedding_size)
        )

        return None
    
    @property
    def attention_matrix(self) -> np.ndarray[float]:
        """
        """
        return self[-1].left
    

# class MultiHeadAttention(Concat):
#     """
#     """

#     def __init__(self, L: int, d_in: int, parameters: list[tuple[int, bool, float]]) -> None:
#         """
#         """
#         super(MultiHeadAttention, self).__init__(
#             *[Attention(L=L, d_in=d_in, embedding_size=embedding_size, bias=bias, drop=drop) for (embedding_size, bias, drop) in parameters]
#         )