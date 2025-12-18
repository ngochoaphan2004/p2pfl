import numpy as np
import math
from collections import defaultdict
from typing import Any, List, Dict
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger

class DFedAdp(Aggregator):
    SUPPORTS_PARTIAL_AGGREGATION: bool = False
    REQUIRED_INFO_KEYS = ["delta", "degrees"] 
    

    def __init__(self, disable_partial_aggregation: bool = False, learning_rate: float = 0.001, log_dfedadp_params: bool = False, decay_rate: float = 0.98, min_learning_rate: float = 0.0001,alpha : float = 5.0) -> None:
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)
        self.global_model_params: List[np.ndarray] = []
        # Map contributor_id -> smoothed_angle history
        self.node_correlation: Dict[str, float] = defaultdict(lambda: 0.0)
        # Learning rate
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.decay_rate = decay_rate
        # Store previous local gradient for Gradient Tracking
        self.prev_local_gradient: List[np.ndarray] = []
        # Control logging of dfedadp parameters
        self.log_dfedadp_params = log_dfedadp_params
        # ALPHA for glompetz function
        self.ALPHA = alpha

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

        if self.log_dfedadp_params:
            logger.info(self.addr, f"DFedAdp Round {current_round}: Learning rate = {self.learning_rate}")
        
        # 2. Initial Round (Round 0): Setup params and initial tracking gradient
        if not self.global_model_params:
            self.global_model_params = [p.copy() for p in self_model.get_parameters()]
            for m in models:
                info = self._get_and_validate_model_info(m)
                delta = info["delta"]
                # Initial tracking gradient is just the local gradient
                self.prev_local_gradient = [-d / self.learning_rate for d in delta]
                # Then set the gradients_estimate attribute directly
                m.gradients_estimate = self.prev_local_gradient
                
                # Log initial round information if logging is enabled
                if self.log_dfedadp_params:
                    node_id = m.get_contributors()[0] if m.get_contributors() else "initial_node"
                    logger.info(self.addr, f"DFedAdp Round {current_round}: Initial round - Node {node_id}, gradient estimate norm = {np.linalg.norm(np.concatenate([g.ravel() for g in self.prev_local_gradient])):.4f}")

        # 3. Calculate Metropolis-Hastings Weights (Topology-based)
        degrees = [int(self._get_and_validate_model_info(m)["degrees"]) for m in models]
        weight_metro = [0.0] * len(degrees)
        my_degree = degrees[0]
        for i in range(1, len(degrees)):
            weight_metro[i] = 1.0 / (1 + max(my_degree, degrees[i]))
        weight_metro[0] = 1.0 - sum(weight_metro[1:])
        
        # Log Metropolis-Hastings weights if logging is enabled
        if self.log_dfedadp_params:
            logger.info(self.addr, f"DFedAdp Round {current_round}: Metro weights = {weight_metro}")

        # 4. Compute Current Local Gradient
        self_info = self._get_and_validate_model_info(self_model)
        self_delta = self_info["delta"]
        curr_local_gradient = [-d / self.learning_rate for d in self_delta]
        
        # Log current node's local gradient if logging is enabled
        if self.log_dfedadp_params:
            self_node_id = self_model.get_contributors()[0] if self_model.get_contributors() else "self_node"
            logger.info(self.addr, f"DFedAdp Round {current_round}, Node {self_node_id}: Local gradient norm = {np.linalg.norm(np.concatenate([g.ravel() for g in curr_local_gradient])):.4f}")

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
            
            # Log gradient estimate for this node if logging is enabled
            if self.log_dfedadp_params:
                node_id = m.get_contributors()[0] if m.get_contributors() else f"node_{idx}"
                logger.info(self.addr, f"DFedAdp Round {current_round}, Node {node_id}: Gradient estimate norm = {np.linalg.norm(np.concatenate([g.ravel() for g in g_j_prev])):.4f}")

            w_ij = weight_metro[idx]
            weighted_neighbor_tracking = [acc + w_ij * g for acc, g in zip(weighted_neighbor_tracking, g_j_prev)]

        # 5b. Add Gradient Drift (Current - Previous Local Gradient)
        if not self.prev_local_gradient:
            self.prev_local_gradient = [np.zeros_like(p) for p in curr_local_gradient]
             
        tracking_gradient = [
            wn + curr - prev
            for wn, curr, prev in zip(weighted_neighbor_tracking, curr_local_gradient, self.prev_local_gradient)
        ]

        # Log the tracking gradient if logging is enabled
        if self.log_dfedadp_params:
            logger.info(self.addr, f"DFedAdp Round {current_round}: Tracking gradient norm = {np.linalg.norm(np.concatenate([tg.ravel() for tg in tracking_gradient])):.4f}")

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
            
            # Log detailed information for this node if logging is enabled
            if self.log_dfedadp_params:
                logger.info(self.addr, f"DFedAdp Round {current_round}, Node {node_id}: Angle={angle:.4f}, Smoothed Angle={smoothed_angle:.4f}, Gompertz f_val={f_val:.4f}, Score={score:.4f}, Local grad norm={l_norm:.4f}")

        # Log the mixing weights and scores if logging is enabled
        if self.log_dfedadp_params:
            logger.info(self.addr, f"DFedAdp Round {current_round}: FedAdp scores = {fedadp_scores}")
            logger.info(self.addr, f"DFedAdp Round {current_round}: Normalized psi weights = {psi}")
            
        # 8. Consensus step (pure Metropolis, NO adaptive weighting)
        w_half = [np.zeros_like(p, dtype=np.float64) for p in self.global_model_params]

        for idx, m in enumerate(models):
            w = weight_metro[idx]   # <-- CHỈ Metropolis
            for i, layer in enumerate(m.get_parameters()):
                w_half[i] += layer * w

        # 9. Final Update (Adaptive step-size, NOT adaptive mixing)
        total_score = sum(fedadp_scores)
        psi = [s / total_score if total_score > 0 else 1.0/len(models) for s in fedadp_scores]

        self.global_model_params = [
            wh - self.learning_rate * tg
            for wh, tg in zip(w_half, tracking_gradient)
        ]



        if self.log_dfedadp_params:
            logger.info(self.addr, f"DFedAdp Round {current_round}: Aggregated {len(models)} models.")
        
        # Log final model parameters information if logging is enabled
        if self.log_dfedadp_params:
            param_norm = np.linalg.norm(np.concatenate([p.ravel() for p in self.global_model_params]))
            logger.info(self.addr, f"DFedAdp Round {current_round}: Global model param norm = {param_norm:.4f}")
        
        # Log node correlation information if logging is enabled
        if self.log_dfedadp_params:
            logger.info(self.addr, f"DFedAdp Round {current_round}: Node correlations = {dict(list(self.node_correlation.items()))}")

        # Create the model copy without gradients_estimate first
        result_model = models[0].build_copy(
            params=self.global_model_params,
            num_samples=total_samples,
            contributors=contributors
        )
        # Then set the gradients_estimate attribute directly
        result_model.gradients_estimate = tracking_gradient
        
        # Log the final gradient estimate that will be sent to other nodes if logging is enabled
        if self.log_dfedadp_params:
            result_node_id = result_model.get_contributors()[0] if result_model.get_contributors() else "result_node"
            logger.info(self.addr, f"DFedAdp Round {current_round}, Node {result_node_id}: Final gradient estimate norm = {np.linalg.norm(np.concatenate([tg.ravel() for tg in tracking_gradient])):.4f}")
        # Update learning_rate
        self.learning_rate = max(self.learning_rate*self.decay_rate, self.min_learning_rate)
        return result_model

    def _gompertz_function(self, angle: float):
        # Non-linear Gompertz mapping function
        return self.ALPHA * (1 - math.exp(-math.exp(-self.ALPHA * (angle - 1))))
    
    def _get_and_validate_model_info(self, model: P2PFLModel) -> dict[str, Any]:
        try:
            info = model.get_info("dfedadp")
        except KeyError:
            info = model.get_info()
        
        if "delta" not in info:
            raise ValueError(f"Model missing 'delta' information required for DFedAdp.")
        return info

    def get_required_callbacks(self) -> list[str]:
        return ["dfedadp"]