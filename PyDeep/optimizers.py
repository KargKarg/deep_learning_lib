import numpy as np
from abc import ABC, abstractmethod
from .modules import Module, Concat


class Optimizer(ABC):
    """
    """

    def __init__(self, model: Module) -> None:
        """
        """
        super().__init__()
        self.model = model
        return None
    
    
    @abstractmethod
    def step(self) -> None:
        """
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    """

    def __init__(self, model: Module, lr: float) -> None:
        """"""
        super().__init__(model)
        self.lr = lr
        return None
    

    def step(self) -> None:
        """
        """
            
        grad_loss_parameters = self.model.grad_loss_parameters()

        for module, grad in zip(self.model[::-1], grad_loss_parameters):
            
            if grad is None:    continue

            if module.__class__.__name__ == "Concat":

                for sub_module, sub_grad in zip(module, grad):

                    if sub_grad is None:    continue
                    
                    sub_module.parameters = {p_name: (p_value - self.lr*sub_grad[p_name]) for p_name, p_value in sub_module.parameters.items()}

                continue

            module.parameters = {p_name: (p_value - self.lr*grad[p_name]) for p_name, p_value in module.parameters.items()}