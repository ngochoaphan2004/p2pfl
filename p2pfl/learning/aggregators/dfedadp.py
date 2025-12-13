import numpy as np
import math
from collections import defaultdict
from typing import Any, List, Dict
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger

class DFedAdp(Aggregator):
    SUPPORTS_PARTIAL_AGGREGATION: bool = True
    REQUIRED_INFO_KEYS = ["delta", "degrees"] 
    ALPHA = 5.0 

    def __init__(self, disable_partial_aggregation: bool = False, learning_rate: float = 0.1) -> None:
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)
        self.global_model_params: List[np.ndarray] = []
        # Map contributor_id -> smoothed_angle history
        self.node_correlation: Dict[str, float] = defaultdict(lambda: 0.0)
        self.learning_rate = learning_rate
        # Store previous local gradient for Gradient Tracking
        self.prev_local_gradient: List[np.ndarray] = [] 

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        # Validate input
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) No models to aggregate")

        # 1. Basic setup: total samples and contributors
        total_samples = sum([m.get_num_samples() for m in models])
        contributors = []
        for m in models:
            contributors.extend(m.get_contributors())
        
        current_round = self.each_trained_round.get(self.addr, 0)
        self_model = models[0] # Assuming models[0] is self
        
        # 2. Initial Round (Round 0): Setup params and initial tracking gradient
        if not self.global_model_params:
            self.global_model_params = [p.copy() for p in self_model.get_parameters()]
            info = self._get_and_validate_model_info(self_model)
            delta = info["delta"]
            # Initial tracking gradient is just the local gradient
            self.prev_local_gradient = [-d / self.learning_rate for d in delta]
            
            return self_model.build_copy(
                params=self.global_model_params, 
                num_samples=total_samples, 
                contributors=contributors, 
                gradients_estimate=self.prev_local_gradient
            )

        # 3. Calculate Metropolis-Hastings Weights (Topology-based)
        degrees = [int(m.degrees) for m in models]
        weight_metro = [0.0] * len(degrees)
        my_degree = degrees[0]
        
        for i in range(1, len(degrees)):
            weight_metro[i] = 1.0 / (1 + max(my_degree, degrees[i]))
        weight_metro[0] = 1.0 - sum(weight_metro[1:])

        # 4. Compute Current Local Gradient
        self_info = self._get_and_validate_model_info(self_model)
        self_delta = self_info["delta"]
        curr_local_gradient = [-d / self.learning_rate for d in self_delta]

        # 5. Gradient Tracking: Estimate Global Gradient
        # 5a. Weighted sum of neighbors' previous tracking gradients
        weighted_neighbor_tracking = [np.zeros_like(p) for p in self.global_model_params]
        for idx, m in enumerate(models):
            # Retrieve tracking gradient from previous round
            if hasattr(m, 'gradients_estimate') and m.gradients_estimate:
                g_j_prev = m.gradients_estimate
            else:
                 d = self._get_and_validate_model_info(m)["delta"]
                 g_j_prev = [-x / self.learning_rate for x in d]

            w_ij = weight_metro[idx]
            weighted_neighbor_tracking = [acc + w_ij * g for acc, g in zip(weighted_neighbor_tracking, g_j_prev)]

        # 5b. Add Gradient Drift (Current - Previous Local Gradient)
        if not self.prev_local_gradient:
             self.prev_local_gradient = [np.zeros_like(p) for p in curr_local_gradient]
             
        tracking_gradient = [
            wn + curr - prev 
            for wn, curr, prev in zip(weighted_neighbor_tracking, curr_local_gradient, self.prev_local_gradient)
        ]

        # Update previous local gradient for the next round
        self.prev_local_gradient = [g.copy() for g in curr_local_gradient]

        # 6. Calculate FedAdp Scores (Contribution Measurement)
        fedadp_scores = []
        g_vec = np.concatenate([p.ravel() for p in tracking_gradient])
        g_norm = np.linalg.norm(g_vec)

        for idx, m in enumerate(models):
            ctrb = m.get_contributors()
            node_id = ctrb[0] if ctrb else f"unknown_{idx}"
            
            # Reconstruct neighbor's local gradient from delta
            m_delta = self._get_and_validate_model_info(m)["delta"]
            neigh_local_grad = [-d / self.learning_rate for d in m_delta]
            
            l_vec = np.concatenate([p.ravel() for p in neigh_local_grad])
            l_norm = np.linalg.norm(l_vec)

            # Compute Angle (Cosine Similarity)
            if g_norm == 0 or l_norm == 0:
                cos_sim = 1.0
            else:
                cos_sim = np.dot(g_vec, l_vec) / (g_norm * l_norm)
                cos_sim = np.clip(cos_sim, -1.0, 1.0)
            angle = float(np.arccos(cos_sim))

            # Smooth Angle over rounds
            prev_angle = self.node_correlation[node_id]
            if current_round <= 1 or prev_angle == 0.0:
                smoothed_angle = angle
            else:
                smoothed_angle = ((current_round - 1)/current_round)*prev_angle + (1/current_round)*angle
            
            self.node_correlation[node_id] = smoothed_angle

            # Compute Score using Gompertz function
            f_val = self._gompertz_function(smoothed_angle)
            score = m.get_num_samples() * math.exp(f_val)
            fedadp_scores.append(score)

        # 7. Calculate Adaptive Mixing Matrix (Metropolis + FedAdp)
        total_score = sum(fedadp_scores)
        psi = [s / total_score if total_score > 0 else 1.0/len(models) for s in fedadp_scores]
        
        unnormalized_mix = [p * w for p, w in zip(psi, weight_metro)]
        sum_mix = sum(unnormalized_mix)
        final_mixing_weights = [u / sum_mix if sum_mix > 0 else 1.0/len(models) for u in unnormalized_mix]

        # 8. Aggregation Step (Consensus)
        w_half = [np.zeros_like(p, dtype=np.float64) for p in self.global_model_params]
        for idx, m in enumerate(models):
            w = final_mixing_weights[idx]
            for i, layer in enumerate(m.get_parameters()):
                w_half[i] += layer * w

        # 9. Final Update (Apply Gradient Tracking correction)
        self.global_model_params = [
            wh - self.learning_rate * tg 
            for wh, tg in zip(w_half, tracking_gradient)
        ]

        logger.info(self.addr, f"DFedAdp Round {current_round}: Aggregated {len(models)} models.")

        return models[0].build_copy(
            params=self.global_model_params, 
            num_samples=total_samples, 
            contributors=contributors, 
            gradients_estimate=tracking_gradient
        )

    def _gompertz_function(self, angle: float):
        # Non-linear Gompertz mapping function
        return self.ALPHA * (1 - math.exp(-math.exp(-self.ALPHA * (angle - 1))))
    
    def _get_and_validate_model_info(self, model: P2PFLModel) -> dict[str, Any]:
        # Validate model info (must contain 'delta')
        info = model.get_info("dfedadp")
        if "delta" not in info: 
             if "delta" in model.additional_info:
                 return model.additional_info
             raise ValueError(f"Model missing 'delta' information required for DFedAdp.")
        return info

    def get_required_callbacks(self) -> list[str]:
        return ["dfedadp"]