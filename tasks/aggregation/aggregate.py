import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname))
sys.path.append(os.path.join(dirname, '../'))

import numpy as np

from strategies.fed_avg import FedeartedAveraging
from traininig.utils.parameters import deserialize_parameters, serialize_parameters

# Initialize the strategies
fed_avg = FedeartedAveraging()

def aggregate(
    round_data,
    algorithm: str,
):
    if algorithm == 'fed_avg':
        num_samples = np.array([client_data['num_samples'] for client_data in round_data])
        parameters = [deserialize_parameters(client_data['parameters']) for client_data in round_data]

        # Get the aggregated parameters
        aggregated_parameters = fed_avg.aggregate(
            num_samples=num_samples,
            parameters=parameters,
        )

    # Serialize the parameters
    serialized_parameters = serialize_parameters(aggregated_parameters)

    return serialized_parameters