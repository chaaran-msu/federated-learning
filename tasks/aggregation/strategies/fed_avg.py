import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname, '../../'))

from typing import List
import numpy as np

class FedeartedAveraging():
    def __init__(self):
        pass

    def weighted_average(
        self,
        weights: np.ndarray,
        values: np.ndarray
    ):
        return np.tensordot(weights, values, axes=(0,0)) / np.sum(weights)

    def aggregate(
        self,
        num_samples: np.ndarray, 
        parameters: List
    ):
        total_num_samples = np.sum(num_samples)

        aggregated_parameters = []

        for elements in zip(*parameters):
            aggregated_parameters.append(
                self.weighted_average(
                    weights=num_samples, 
                    values=elements
                )
            )

        return total_num_samples, aggregated_parameters