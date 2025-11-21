import numpy as np
from abc import ABC, abstractmethod


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
    def step(self, eps: float) -> None:
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
    

    def step(self, eps: float) -> None:
        [module.step(eps) for module in self.modules[::-1]]
        return None


class Linear(Module):
    """
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = True) -> None:
        """"
        """

        super().__init__(d_in, d_out)
        np.random.seed(self.seed)

        self.has_bias = bias

        self.W: np.ndarray[float] = np.random.normal(size=(d_in, d_out))

        match bias:

            case True:
                self.bias: float = np.random.normal(size=(1,))[0]

            case False:
                self.bias: float = 0

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
                self.value: np.ndarray[float] = np.dot(x, self.W) + self.bias

                return self.value
            
            case False:

                return np.dot(x, self.W) + self.bias


    def backward(self, grad_loss_out: np.ndarray[float]) -> None:
        """
        dim(grad_out_in) = (N, d_in, d_out)
        dim(grad_out_W) = (N, d_in, d_out, d_out)
        dim(grad_out_bias) = (N, d_out)
        dim(grad_loss_out) = (N, d_out)
        """
        assert all(hasattr(self, attr) for attr in ("N", "x", "value")), "The forward pass tracked must be computed before the gradient."

        self.grad_out_in: np.ndarray[float] = np.tile(np.copy(self.W), (self.N, 1, 1))

        self.grad_out_W: np.ndarray[float] = self.x[..., None, None] * np.eye(self.d_out)
        self.grad_out_bias: np.ndarray[float] = np.ones(shape=(self.N, self.d_out))

        self.grad_loss_in: np.ndarray[float] = np.sum(grad_loss_out[:, None, :] * self.grad_out_in, axis=2)

        self.grad_loss_out = np.copy(grad_loss_out)

        return self.grad_loss_in


    def step(self, eps: float) -> None:
        """
        """
        
        self.W -= eps*np.mean(np.sum((self.grad_out_W*self.grad_loss_out[:, None, None, :]), axis=3), axis=0)
        self.bias -= int(self.has_bias)*eps*np.mean(np.sum(self.grad_loss_out*self.grad_out_bias, axis=1), axis=0)

        return None
    

if __name__ == "__main__":

    from sklearn.datasets import make_regression
    from loss import MSE

    N, D, K = 100, 10, 1

    X, Y = make_regression(n_samples=N, n_features=D, n_targets=K)

    if K == 1:  Y = Y[..., None]

    NN = Sequential(Linear(D, K), Linear(K, K))
    loss = MSE(K)

    for _ in range(1000):

        loss(NN(X), Y)
        NN.backward(loss.backward())
        
        NN.step(1e-1)

        print(loss.value)