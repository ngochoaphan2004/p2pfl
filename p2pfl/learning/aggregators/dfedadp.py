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
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) No models to aggregate")

        total_samples = sum(m.get_num_samples() for m in models)
        contributors = [c for m in models for c in m.get_contributors()]
        current_round = self.each_trained_round.get(self.addr, 0)
        self_model = models[0]

        # ===== 0. INIT GLOBAL MODEL & GRADIENT TRACK =====
        if not self.global_model_params:
            self.global_model_params = [p.copy() for p in self_model.get_parameters()]
            info = self._get_and_validate_model_info(self_model)
            delta = info["delta"]
            self.prev_local_gradient = [-d / self.learning_rate for d in delta]
            self_model.gradients_estimate = self.prev_local_gradient

        # ===== 1. METROPOLIS WEIGHTS =====
        degrees = [int(self._get_and_validate_model_info(m)["degrees"]) for m in models]
        my_degree = degrees[0]
        W = [0.0]*len(models)
        for i in range(1, len(models)):
            W[i] = 1.0/(1 + max(my_degree, degrees[i]))
        W[0] = 1 - sum(W[1:])

        # ===== 2. CURRENT LOCAL GRADIENT (pseudo from delta) =====
        info_self = self._get_and_validate_model_info(self_model)
        curr_local_grad = [-d/self.learning_rate for d in info_self["delta"]]

        # ===== 3. GRADIENT TRACKING =====
        g_mix = [np.zeros_like(p) for p in self.global_model_params]
        for idx, m in enumerate(models):
            if hasattr(m,"gradients_estimate"):
                prev_hat = m.gradients_estimate
            else:
                delta_i = self._get_and_validate_model_info(m)["delta"]
                prev_hat = [-x/self.learning_rate for x in delta_i]
            for k in range(len(g_mix)):
                g_mix[k] += W[idx]*prev_hat[k]

        g_prev = self.prev_local_gradient
        g_hat = [gm + c - p for gm,c,p in zip(g_mix, curr_local_grad, g_prev)]
        self.prev_local_gradient = curr_local_grad

        # ===== 4. COMPUTE ψ (score, softmax) =====
        # cycle through nodes → use pseudo-gradients for angle
        g_vec = np.concatenate([g.ravel() for g in g_hat])
        g_norm = np.linalg.norm(g_vec)

        psi = []
        for idx, m in enumerate(models):
            info_i = self._get_and_validate_model_info(m)
            neigh_grad = [-d/self.learning_rate for d in info_i["delta"]]
            l_vec = np.concatenate([g.ravel() for g in neigh_grad])
            l_norm = np.linalg.norm(l_vec)

            if g_norm == 0 or l_norm == 0:
                cos = 1.0
            else:
                cos = np.clip(np.dot(g_vec, l_vec)/(g_norm*l_norm), -1,1)

            angle = float(np.arccos(cos))
            prev_angle = self.node_correlation[m.get_contributors()[0]]
            smoothed = angle if (current_round<=1 or prev_angle==0) else ((current_round-1)/current_round)*prev_angle + (1/current_round)*angle
            self.node_correlation[m.get_contributors()[0]] = smoothed

            f_val = self._gompertz_function(smoothed)
            psi.append(math.exp(f_val))

        # normalize psi
        total = sum(psi)
        psi = [p/total if total>0 else 1.0/len(models) for p in psi]

        # ===== 5. ADAPTIVE MIXING MATRIX =====
        # build row-stochastic \tilde W for this aggregator
        # for each neighbor j → weight = psi_j * Metropolis
        neighbor_ids = range(len(models))
        Wi_tilde = []
        for i in neighbor_ids:
            row = []
            denom = sum(psi[j]*W[j] for j in neighbor_ids)
            for j in neighbor_ids:
                row.append((psi[j]*W[j])/denom if denom>0 else W[j])
            Wi_tilde.append(row)  # note: symmetric only if undirected

        # ===== 6. CONSENSUS with adaptive mixing =====
        w_adapt = [np.zeros_like(p) for p in self.global_model_params]
        for j in neighbor_ids:
            for k in range(len(w_adapt)):
                w_adapt[k] += Wi_tilde[j][0] * models[j].get_parameters()[k]

        # ===== 7. FINAL UPDATE (no scaling psi) =====
        self.global_model_params = [
            wj - self.learning_rate*gh
            for wj,gh in zip(w_adapt, g_hat)
        ]

        # ===== 8. Build model copy and return =====
        result = self_model.build_copy(
            params=self.global_model_params,
            num_samples=total_samples,
            contributors=contributors
        )
        result.gradients_estimate = g_hat

        return result


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