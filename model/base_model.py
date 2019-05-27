import numpy as np
import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Model base class
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
