import copy
from typing import Any, List

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from p2pfl.learning.frameworks.callback import P2PFLCallback


class DFEDADPCallback(Callback, P2PFLCallback):
    """
    DFedAdp callback.
    - Lưu gradient thật g_i(w_i(t)) và g_i(w_i(t-1))
    - Lưu delta cho compatibility
    """

    def __init__(self) -> None:
        super().__init__()
        self.additional_info: dict[str, Any] = {}

        # DIGing states
        self.prev_local_gradients: List[np.ndarray] | None = None
        self.curr_local_gradients: List[np.ndarray] | None = None

    @staticmethod
    def get_name() -> str:
        return "dfedadp"

    # ==========================================================
    # 1. Store global model before local training
    # ==========================================================
    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        self.additional_info["global_model"] = copy.deepcopy(
            self._get_parameters(pl_module)
        )

    # ==========================================================
    # 2. Collect TRUE local gradient (after backward)
    # ==========================================================
    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        grads = [
            p.grad.detach().cpu().numpy().copy()
            for p in pl_module.parameters()
            if p.grad is not None
        ]
        self.curr_local_gradients = grads

    # ==========================================================
    # 3. End of local training round
    # ==========================================================
    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        # ---- delta (for compatibility) ----
        post_params = self._get_parameters(pl_module)
        pre_params = self.additional_info["global_model"]
        delta = [post - pre for post, pre in zip(post_params, pre_params)]
        self.additional_info["delta"] = [
            d.detach().cpu().numpy() for d in delta
        ]

        # ---- DIGing gradients ----
        self.additional_info["local_gradients"] = self.curr_local_gradients
        self.additional_info["prev_local_gradients"] = self.prev_local_gradients

        # shift gradient history
        self.prev_local_gradients = self.curr_local_gradients
        self.curr_local_gradients = None

    # ==========================================================
    def _get_parameters(self, pl_module: pl.LightningModule):
        return [p.detach().cpu() for _, p in pl_module.state_dict().items()]

    def get_info(self) -> dict[str, Any]:
        return self.additional_info

    def set_info(self, info: dict[str, Any]) -> None:
        self.additional_info.update(info)
