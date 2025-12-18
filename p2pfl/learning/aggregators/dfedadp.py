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


    def __init__(self, disable_partial_aggregation: bool = False, learning_rate: float = 0.001, log_dfedadp_params: bool = False, decay_rate: float = 0.98, min_learning_rate: float = 0.0001, alpha : float = 5.0) -> None:
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
        # Store previous local gradients (new for gradient collection)
        self.prev_collected_gradients: Dict[str, List[np.ndarray]] = {}
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
        self_info = self._get_and_validate_model_info(self_model)

        # ===== 0. INIT =====
        if not self.global_model_params:
            self.global_model_params = [p.copy() for p in self_model.get_parameters()]
            # Use collected gradients if available, otherwise calculate from delta
            if "local_gradients" in self_info:
                self.prev_local_gradient = self_info["local_gradients"]
            else:
                # Calculate gradient from delta and learning rate
                delta = self_info["delta"]
                self.prev_local_gradient = [-d / self.learning_rate for d in delta]
            self_model.gradients_estimate = self.prev_local_gradient

        # ===== 1. METROPOLIS WEIGHTS =====
        degrees = [int(self._get_and_validate_model_info(m)["degrees"]) for m in models]
        my_degree = degrees[0]

        W = [0.0] * len(models)
        for i in range(1, len(models)):
            W[i] = 1.0 / (1 + max(my_degree, degrees[i]))
        W[0] = 1.0 - sum(W[1:])

        # ===== 2. CURRENT LOCAL GRADIENT (TRUE GRADIENT) =====
        if "local_gradients" in self_info:
            g_curr = self_info["local_gradients"]
        else:
            # Calculate gradient from delta and learning rate
            delta = self_info["delta"]
            g_curr = [-d / self.learning_rate for d in delta]

        # ===== 3. GRADIENT TRACKING (DIGing) =====
        g_mix = [np.zeros_like(p) for p in self.global_model_params]

        for idx, m in enumerate(models):
            if hasattr(m, "gradients_estimate"):
                g_prev_hat = m.gradients_estimate
            else:
                info = self._get_and_validate_model_info(m)
                if "local_gradients" in info:
                    g_prev_hat = info["local_gradients"]
                else:
                    # Calculate gradient from delta and learning rate
                    delta = info["delta"]
                    g_prev_hat = [-d / self.learning_rate for d in delta]

            for k in range(len(g_mix)):
                g_mix[k] += W[idx] * g_prev_hat[k]

        g_prev = self.prev_local_gradient
        g_hat = [gm + gc - gp for gm, gc, gp in zip(g_mix, g_curr, g_prev)]

        self.prev_local_gradient = g_curr

        # ===== 4. FEDADP SCORES (ONLY FOR LOGGING) =====
        g_vec = np.concatenate([g.ravel() for g in g_hat])
        g_norm = np.linalg.norm(g_vec)

        fedadp_scores = []

        for idx, m in enumerate(models):
            info = self._get_and_validate_model_info(m)
            node_id = m.get_contributors()[0]

            # Use collected gradients if available, otherwise calculate from delta
            if "local_gradients" in info:
                l_grad = info["local_gradients"]
            else:
                # Calculate gradient from delta and learning rate
                delta = info["delta"]
                l_grad = [-d / self.learning_rate for d in delta]

            l_vec = np.concatenate([g.ravel() for g in l_grad])
            l_norm = np.linalg.norm(l_vec)

            if g_norm == 0 or l_norm == 0:
                angle = 0.0
            else:
                cos = np.clip(np.dot(g_vec, l_vec) / (g_norm * l_norm), -1, 1)
                angle = float(np.arccos(cos))

            prev_angle = self.node_correlation[node_id]
            if current_round <= 1:
                smoothed = angle
            else:
                smoothed = ((current_round - 1)/current_round)*prev_angle + (1/current_round)*angle

            self.node_correlation[node_id] = smoothed
            f_val = self._gompertz_function(smoothed)
            fedadp_scores.append(math.exp(f_val))

        # ===== 5. CONSENSUS (PURE METROPOLIS) =====
        w_half = [np.zeros_like(p) for p in self.global_model_params]
        for idx, m in enumerate(models):
            for k, layer in enumerate(m.get_parameters()):
                w_half[k] += W[idx] * layer

        # ===== 6. FINAL UPDATE (NO ψ) =====
        self.global_model_params = [
            wh - self.learning_rate * gh
            for wh, gh in zip(w_half, g_hat)
        ]

        # ===== 7. BUILD RESULT =====
        result = self_model.build_copy(
            params=self.global_model_params,
            num_samples=total_samples,
            contributors=contributors
        )
        result.gradients_estimate = g_hat

        self.learning_rate = max(self.learning_rate * self.decay_rate, self.min_learning_rate)
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
        return ["dfedadp", "gradient_collection"]