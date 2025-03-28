from torch import nn
from collections import OrderedDict
from typing import List
import numpy as np
import torch

def set_parameters(
    model: nn.Module, 
    parameters: List[np.ndarray]
):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]