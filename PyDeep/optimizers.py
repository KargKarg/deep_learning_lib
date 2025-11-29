from abc import ABC, abstractmethod
from .modules import Module


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

        def update(module, grad) -> None:
            """
            """
            if grad is None:    return None

            if type(grad) == dict:
                module.parameters = {p_name: (p_value - self.lr*grad[p_name]) for p_name, p_value in module.parameters.items()}
                return None
            
            if type(grad) == list:
                for sub_module, sub_grad in zip(module[::-1], grad[::-1]):
                    update(sub_module, sub_grad)
                return None

            
        grad_loss_parameters = self.model.grad_loss_parameters()

        for module, grad in zip(self.model[::-1], grad_loss_parameters[::-1]):
            
            update(module, grad)

        return None