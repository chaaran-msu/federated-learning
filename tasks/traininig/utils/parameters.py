from torch import nn
from collections import OrderedDict
from typing import List
import numpy as np
import torch
import io

def set_parameters(
    model: nn.Module, 
    params_list: List[np.ndarray]
):
    # Convert back to PyTorch tensors
    params = [torch.tensor(params_list[f'arr_{i}']) for i in range(len(params_list.files))]

    # Load into model
    state_dict = model.state_dict()
    for key, param in zip(state_dict.keys(), params):
        state_dict[key] = param

    model.load_state_dict(state_dict)
    
def get_parameters(
    model: nn.Module
) -> List[np.ndarray]:
    params = model.state_dict()
    params_list = [p.cpu().numpy() for p in params.values()]

    return params_list

def serialize_parameters(
    params_list
):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, *params_list)  # Compress to reduce size
    serialized_params = buffer.getvalue()

    return serialized_params

def get_serialized_parameters(
    model: nn.Module
):
    params_list = get_parameters(model)
    serialized_params = serialize_parameters(params_list)
    
    return serialized_params

def deserialize_parameters(
    serialized_params,
):
    buffer = io.BytesIO(serialized_params)
    params_list = np.load(buffer)

    return params_list

def load_serialized_parameters(
    model,
    serialized_params,
):
    # Deserialize received parameters
    params_list = deserialize_parameters(serialized_params)

    set_parameters(model, params_list)