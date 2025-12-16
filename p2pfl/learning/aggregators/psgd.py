import numpy as np
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class PSGD(Aggregator):
    """
    Decentralized Parallel SGD (D-PSGD)
    """

    SUPPORTS_PARTIAL_AGGREGATION = True

    def __init__(self, W: dict[str, float] | None = None, lr: float = 0.01):
        """
        Args:
            W: mixing weights W_ij (neighbors + self), sum_j W_ij = 1
               If None, an empty dict is used (missing weights treated as 0).
            lr: step size γ (default 0.01)
        """
        super().__init__()
        self.W = W or {}
        self.lr = lr


    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        if not models:
            raise NoModelsToAggregateError("No models received")

        # Find the model that comes from this node (self node)
        # This is identified by the contributors list containing the current node's address
        self_model = None
        for m in models:
            contributors = m.get_contributors()
            if self.addr in contributors:
                self_model = m
                break

        if self_model is None:
            # If no model from self is found, use the first model as fallback
            self_model = models[0]

        # Get the parameters of the self model to match dimensions
        x_k_i = self_model.get_parameters()

        # -----------------------------
        # 1. Consensus step: x_{k+1/2,i} = Σ_j W_ij * x_{k,j}
        # -----------------------------
        mixed = [np.zeros_like(p) for p in x_k_i]

        for m in models:
            # Get the address of the model's contributors
            contributors = m.get_contributors()
            if contributors:
                node_addr = contributors[0]  # Take the first contributor
                w_ij = self.W.get(node_addr, 0.0)
            else:
                w_ij = 0.0

            if w_ij != 0.0:
                for l, param in enumerate(m.get_parameters()):
                    mixed[l] += w_ij * param

        # In a complete D-PSGD implementation, we would also need gradient information
        # to perform: x_new = x_half - lr * grad_f_i(x_half)
        # However, in the P2PFL framework, gradients are typically computed during
        # local training phase, not during aggregation. The aggregation only does
        # the mixing/consensus step. The gradient step would happen in the learner
        # before or after gossiping models with neighbors.

        # Return the mixed/consensus model
        return self_model.build_copy(
            params=mixed,
            num_samples=self_model.get_num_samples(),
            contributors=self_model.get_contributors(),
        )
