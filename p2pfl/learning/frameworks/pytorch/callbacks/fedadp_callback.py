"""Callback for FEDADP operations (PyTorch Lighting)."""

import copy
from typing import Any

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from p2pfl.learning.frameworks.callback import P2PFLCallback


class FedAdpCallback(Callback, P2PFLCallback):
    """
    Callback for fedadp operations to use with PyTorch Lightning.

    At the beginning of the training, the callback needs to store the global model and the initial learning rate. Then, after optimization,
    """

    def __init__(self) -> None:
        """Initialize the callback."""
        super().__init__()
        self.c_i: list[torch.Tensor] = []
        self.c: list[torch.Tensor] = []
        self.initial_model_params: list[torch.Tensor] = []
        self.saved_lr: float | None = None
        self.K: int = 0
        self.additional_info: dict[str, Any] = {}

    @staticmethod
    def get_name() -> str:
        """Return the name of the callback."""
        return "fedadp"

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Store the global model.

        Args:
            trainer: The trainer
            pl_module: The model.

        """
        if not self.c_i:
            self.c_i = [torch.zeros_like(param) for param in self._get_parameters(pl_module)]

        global_c = self.additional_info.get("global_c")
        if global_c is not None:
            self.c = [torch.from_numpy(c_np).to(pl_module.device) for c_np in global_c]
        else:
            if not self.c:
                self.c = [torch.zeros_like(param) for param in self._get_parameters(pl_module)]

        self.initial_model_params = copy.deepcopy(self._get_parameters(pl_module))
        self.K = 0  # reset local steps counter


    def on_before_zero_grad(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: torch.optim.Optimizer) -> None:
        """
        Modify model by applying control variate adjustment.
        """
        self.K += 1

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Restore the global model.

        Args:
            trainer: The trainer
            pl_module:code The model.

        """
        if not self.initial_model_params:
            raise AttributeError("Necessary attributes are not initialized.")

        y_i = self._get_parameters(pl_module)
        x_g = self.initial_model_params  # global model at the beginning of the training
        previous_c_i = [c.clone() for c in self.c_i]

        for idx, (c_i, x, y) in enumerate(zip(self.c_i, x_g, y_i, strict=False)):
            adjustment = (x - y) / (self.K * self.saved_lr)
            self.c_i[idx] = c_i + adjustment

        # Compute delta y_i and delta c_i
        delta_y_i = [y - x for y, x in zip(y_i, x_g, strict=False)]
        delta_c_i = [c_new - c_old for c_new, c_old in zip(self.c_i, previous_c_i, strict=False)]

        delta_y_i_np = [dyi.detach().cpu().numpy() for dyi in delta_y_i]  # to numpy for transmission
        delta_c_i_np = [dci.detach().cpu().numpy() for dci in delta_c_i]

        self.additional_info["delta_y_i"] = delta_y_i_np
        self.additional_info["delta_c_i"] = delta_c_i_np

    def _get_parameters(self, pl_module: pl.LightningModule) -> list[torch.Tensor]:
        return [param.cpu() for _, param in pl_module.state_dict().items()]

    def _set_parameters(self, pl_module: pl.LightningModule, parameters: list[np.ndarray]) -> None:
        """Set model parameters from a list of numpy arrays."""
        state_dict = pl_module.state_dict()
        for (name, _), param in zip(state_dict.items(), parameters, strict=False):
            state_dict[name] = torch.from_numpy(param).to(pl_module.device)
        pl_module.load_state_dict(state_dict)
