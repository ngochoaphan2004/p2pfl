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

    def __init__(
        self,
        disable_partial_aggregation: bool = False,
        learning_rate: float = 0.001,
        log_dfedadp_params: bool = False,
        decay_rate: float = 0.98,
        min_learning_rate: float = 0.0001,
        alpha: float = 5.0,
    ) -> None:
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)

        self.global_model_params: List[np.ndarray] = []

        # node_id -> smoothed angle
        self.node_correlation: Dict[str, float] = defaultdict(lambda: 0.0)

        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.decay_rate = decay_rate

        # gradient tracking
        self.prev_local_gradient: List[np.ndarray] = []

        self.log_dfedadp_params = log_dfedadp_params
        self.ALPHA = alpha

    # ============================================================
    # Main aggregation
    # ============================================================
    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) No models to aggregate")

        total_samples = sum(m.get_num_samples() for m in models)
        contributors = []
        for m in models:
            contributors.extend(m.get_contributors())

        current_round = self.each_trained_round.get(self.addr, 0)
        self_model = models[0]  # self node

        if self.log_dfedadp_params:
            logger.info(
                self.addr,
                f"DFedAdp Round {current_round}: learning_rate={self.learning_rate}",
            )

        # ------------------------------------------------------------
        # Round 0 initialization
        # ------------------------------------------------------------
        if not self.global_model_params:
            self.global_model_params = [
                p.copy() for p in self_model.get_parameters()
            ]

            for m in models:
                info = self._get_and_validate_model_info(m)
                delta = info["delta"]
                local_grad = [-d / self.learning_rate for d in delta]

                # each node keeps its own gradient estimate
                m.gradients_estimate = local_grad

                # only SELF initializes prev_local_gradient
                if m is self_model:
                    self.prev_local_gradient = [g.copy() for g in local_grad]

        # ------------------------------------------------------------
        # Metropolis weights (fixed, topology-based)
        # ------------------------------------------------------------
        degrees = [
            int(self._get_and_validate_model_info(m)["degrees"]) for m in models
        ]
        weight_metro = [0.0] * len(models)
        my_degree = degrees[0]

        for i in range(1, len(models)):
            weight_metro[i] = 1.0 / (1 + max(my_degree, degrees[i]))

        weight_metro[0] = 1.0 - sum(weight_metro[1:])

        # ------------------------------------------------------------
        # Current local gradient (self)
        # ------------------------------------------------------------
        self_info = self._get_and_validate_model_info(self_model)
        self_delta = self_info["delta"]
        curr_local_gradient = [-d / self.learning_rate for d in self_delta]

        # ------------------------------------------------------------
        # Gradient Tracking
        # ------------------------------------------------------------
        weighted_neighbor_tracking = [
            np.zeros_like(p) for p in self.global_model_params
        ]

        for idx, m in enumerate(models):
            if hasattr(m, "gradients_estimate") and m.gradients_estimate:
                g_prev = m.gradients_estimate
            else:
                d = self._get_and_validate_model_info(m)["delta"]
                g_prev = [-x / self.learning_rate for x in d]

            w_ij = weight_metro[idx]
            weighted_neighbor_tracking = [
                acc + w_ij * g for acc, g in zip(weighted_neighbor_tracking, g_prev)
            ]

        if not self.prev_local_gradient:
            self.prev_local_gradient = [
                np.zeros_like(g) for g in curr_local_gradient
            ]

        tracking_gradient = [
            wn + curr - prev
            for wn, curr, prev in zip(
                weighted_neighbor_tracking,
                curr_local_gradient,
                self.prev_local_gradient,
            )
        ]

        self.prev_local_gradient = [g.copy() for g in curr_local_gradient]

        # ------------------------------------------------------------
        # FedAdp scoring (cosine with tracking gradient)
        # ------------------------------------------------------------
        fedadp_scores = []

        g_vec = np.concatenate([p.ravel() for p in tracking_gradient])
        g_norm = np.linalg.norm(g_vec)

        for idx, m in enumerate(models):
            node_id = (
                m.get_contributors()[0]
                if m.get_contributors()
                else f"node_{idx}"
            )

            d = self._get_and_validate_model_info(m)["delta"]
            local_grad = [-x / self.learning_rate for x in d]

            l_vec = np.concatenate([p.ravel() for p in local_grad])
            l_norm = np.linalg.norm(l_vec)

            if g_norm == 0 or l_norm == 0:
                cos_sim = 1.0
            else:
                cos_sim = np.dot(g_vec, l_vec) / (g_norm * l_norm)
                cos_sim = np.clip(cos_sim, -1.0, 1.0)

            angle = float(np.arccos(cos_sim))

            prev_angle = self.node_correlation[node_id]
            if current_round <= 1 or prev_angle == 0.0:
                smoothed_angle = angle
            else:
                smoothed_angle = (
                    (current_round - 1) / current_round * prev_angle
                    + 1 / current_round * angle
                )

            self.node_correlation[node_id] = smoothed_angle

            f_val = self._gompertz_function(smoothed_angle)
            score = m.get_num_samples() * math.exp(f_val)
            fedadp_scores.append(score)

        # ------------------------------------------------------------
        # Consensus step (pure Metropolis)
        # ------------------------------------------------------------
        w_half = [
            np.zeros_like(p, dtype=np.float64)
            for p in self.global_model_params
        ]

        for idx, m in enumerate(models):
            w = weight_metro[idx]
            for i, layer in enumerate(m.get_parameters()):
                w_half[i] += layer * w

        # ------------------------------------------------------------
        # Adaptive step-size update
        # ------------------------------------------------------------
        total_score = sum(fedadp_scores)
        psi = (
            [s / total_score for s in fedadp_scores]
            if total_score > 0
            else [1.0 / len(models)] * len(models)
        )

        psi_self = psi[0]

        self.global_model_params = [
            wh - self.learning_rate * psi_self * tg
            for wh, tg in zip(w_half, tracking_gradient)
        ]

        # ------------------------------------------------------------
        # Build result model
        # ------------------------------------------------------------
        result_model = models[0].build_copy(
            params=self.global_model_params,
            num_samples=total_samples,
            contributors=contributors,
        )

        result_model.gradients_estimate = tracking_gradient

        self.learning_rate = max(
            self.learning_rate * self.decay_rate, self.min_learning_rate
        )

        return result_model

    # ============================================================
    # Utilities
    # ============================================================
    def _gompertz_function(self, angle: float) -> float:
        return self.ALPHA * (1 - math.exp(-math.exp(-self.ALPHA * (angle - 1))))

    def _get_and_validate_model_info(self, model: P2PFLModel) -> Dict[str, Any]:
        try:
            info = model.get_info("dfedadp")
        except KeyError:
            info = model.get_info()

        if "delta" not in info:
            raise ValueError("Model missing 'delta' information required for DFedAdp.")
        if "degrees" not in info:
            raise ValueError(
                "Model missing 'degrees' information required for DFedAdp."
            )

        return info

    def get_required_callbacks(self) -> List[str]:
        return ["dfedadp"]
