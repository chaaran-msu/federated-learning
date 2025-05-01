from torch import nn
from collections import OrderedDict
from typing import List
import numpy as np
import torch
import io
import base64

def set_parameters(
    model: nn.Module, 
    params_list: List[np.ndarray]
):
    # Convert back to PyTorch tensors
    params = [torch.tensor(params_list[i]) for i in range(len(params_list))]

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
    serialized_params = base64.b64encode(serialized_params).decode('utf-8')

    return serialized_params

def get_serialized_parameters(
    model: nn.Module
):
    params_list = get_parameters(model)
    serialized_params = serialize_parameters(params_list)
    
    return serialized_params

def deserialize_parameters(
    serialized_params
):
    binary_data = base64.b64decode(serialized_params)
    buffer = io.BytesIO(binary_data)

    # Load the compressed .npz archive
    data = np.load(buffer)

    # Retrieve arrays in order using the default keys 'arr_0', 'arr_1', ...
    params_list = [data[f'arr_{i}'] for i in range(len(data.files))]

    return params_list

def load_serialized_parameters(
    model,
    serialized_params,
):
    # Deserialize received parameters
    params_list = deserialize_parameters(serialized_params)

    set_parameters(model, params_list)