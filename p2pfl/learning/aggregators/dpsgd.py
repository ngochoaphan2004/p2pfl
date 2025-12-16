import numpy as np
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class DPSGD(Aggregator):
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

        # -----------------------------
        # 1. Fetch local model (node i)
        # -----------------------------
        def _addr_of(model: P2PFLModel) -> str | None:
            # Try common accessors in preference order
            if hasattr(model, "get_addr"):
                try:
                    return model.get_addr()
                except Exception:
                    pass
            try:
                contribs = model.get_contributors()
                if contribs:
                    return contribs[0]
            except Exception:
                pass
            try:
                info = model.get_info("addr")
                if info:
                    return info
            except Exception:
                pass
            underlying = getattr(model, "model", None)
            if underlying is not None and hasattr(underlying, "addr"):
                return getattr(underlying, "addr")
            return None

        self_model = next(m for m in models if _addr_of(m) == self.addr)
        x_k_i = self_model.get_parameters()

        # -----------------------------
        # 2. Consensus step
        # x_{k+1/2,i} = Σ_j W_ij x_{k,j}
        # -----------------------------
        mixed = [np.zeros_like(p) for p in x_k_i]

        for m in models:
            addr = _addr_of(m)
            w_ij = self.W.get(addr, 0.0)
            if w_ij == 0.0:
                continue
            for l, param in enumerate(m.get_parameters()):
                mixed[l] += w_ij * param

        # -----------------------------
        # 3. Local stochastic gradient
        # ∇F_i(x_{k,i}; ξ_{k,i})
        # -----------------------------
        grads = self_model.get_gradients()

        # -----------------------------
        # 4. SGD update
        # x_{k+1,i}
        # -----------------------------
        updated = [
            x_half - self.lr * g
            for x_half, g in zip(mixed, grads)
        ]

        return self_model.build_copy(
            params=updated,
            num_samples=self_model.get_num_samples(),
            contributors=self_model.get_contributors(),
        )
