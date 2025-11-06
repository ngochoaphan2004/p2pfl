import numpy as np
from functools import reduce
import math
from typing import Any
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class FedAdp(Aggregator):

    SUPPORTS_PARTIAL_AGGREGATION: bool = True

    def __init__(self, disable_partial_aggregation: bool = False) -> None:
        """Initialize the aggregator."""
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)
        self.timer = 1
        self.global_model_params: list[np.ndarray] = []
        self.node_correlation : dict[str, float] = {}

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate the models.

        Args:
            models: Dictionary with the models (node: model,num_samples).

        Returns:
            A P2PFLModel with the aggregated.

        Raises:
            NoModelsToAggregateError: If there are no models to aggregate.

        """
        # Check if there are models to aggregate
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) Trying to aggregate models when there is no models")
        
        # TEMPORARY
        # Set init global model
        if not self.global_model_params:
            self.global_model_params = self._init_global_model(models)

        # Total Samples
        total_samples = sum([m.get_num_samples() for m in models])

        # Create a Zero Model using numpy
        accum = [np.copy(p) for p in self.global_model_params]

        # Get weighted models
        strategy_weights = []
        g_vec = np.concatenate([p.ravel() for p in self.global_model_params])
        g_norm = np.linalg.norm(g_vec)
        t = self.timer
        for m in models:
            l_vec = np.concatenate([p.ravel() for p in m.get_parameters()])
            l_norm = np.linalg.norm(l_vec)

            if g_norm == 0 or l_norm == 0:
                cos_sim = 0.0
            else:
                cos_sim = float(np.dot(g_vec, l_vec) / (g_norm * l_norm))
                cos_sim = np.clip(cos_sim, -1.0, 1.0)
            angle = float(np.arccos(cos_sim))

            address = m.get_info("address")["self"]
            pre_arccos_i = self.node_correlation.get(address)
            arccos_i = angle if (self.timer == 1 or pre_arccos_i is None) else (((t-1)/t)*pre_arccos_i + (1/t)*angle)
            self.node_correlation[address] = arccos_i

            gompertz_value = self._gompertz_function(arccos_i)
            strategy_weights.append(m.get_num_samples() * gompertz_value)
            
        
        # Normalize weights
        total_strategy_weights = sum(strategy_weights)
        strategy_weights = [w/total_strategy_weights for w in strategy_weights]

        # Normalize accum
        for index, m in enumerate(models):
            for i, layer in enumerate(m.get_parameters()):
                accum[i] = np.add(accum[i], np.subtract(layer,self.global_model_params[i]) * strategy_weights[index])

        # Get contributors
        contributors: list[str] = []
        for m in models:
            contributors = contributors + m.get_contributors()

        # Increase timer value
        self.timer += 1

        # Change global parameter with newest
        self.global_model_params = accum

        # Return an aggregated p2pfl model
        return models[0].build_copy(params=accum, num_samples=total_samples, contributors=contributors)


    def _init_global_model(self, models: list[P2PFLModel]):
        # Total Samples
        total_samples = sum([m.get_num_samples() for m in models])

        # Create a Zero Model using numpy
        first_model_weights = models[0].get_parameters()
        accum = [np.zeros_like(layer) for layer in first_model_weights]

        # Add weighted models
        for m in models:
            for i, layer in enumerate(m.get_parameters()):
                accum[i] = np.add(accum[i], layer * m.get_num_samples())

        # Normalize Accum
        accum = [np.divide(layer, total_samples) for layer in accum]

        return accum
    
    def _gompertz_function(self, angle: float):
        return 5*(1-math.exp(-(math.exp(-5*(angle - 1)))))