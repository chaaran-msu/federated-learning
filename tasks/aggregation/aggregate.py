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
    logger
):
    if algorithm == 'fed_avg':
        num_samples = np.array([round_data[device_id]['num_samples'] for device_id in round_data])
        parameters = [deserialize_parameters(round_data[device_id]['parameters']) for device_id in round_data]

        # Get the aggregated parameters
        total_num_samples, aggregated_parameters = fed_avg.aggregate(
            num_samples=num_samples,
            parameters=parameters,
            logger=logger
        )

    # Serialize the parameters
    serialized_parameters = serialize_parameters(aggregated_parameters)

    return total_num_samples, serialized_parameters