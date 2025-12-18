#
# Gradient Collection Callback for DFedADP
#

"""Callback for collecting gradients during training (PyTorch Lightning)."""

import copy
from typing import Any, List
import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from p2pfl.learning.frameworks.callback import P2PFLCallback


class GradientCollectionCallback(Callback, P2PFLCallback):
    """
    Callback to collect gradients ∇F_i(w_i(t)) during the training process.
    
    This callback hooks into the training loop to capture gradients at each iteration or epoch.
    """

    def __init__(self) -> None:
        """Initialize the callback."""
        super().__init__()
        self.gradients_history: List[List[np.ndarray]] = []  # Store gradients across training steps
        self.current_gradients: List[np.ndarray] | None = None  # Store current gradient
        self.additional_info: dict[str, Any] = {}

    @staticmethod
    def get_name() -> str:
        """Return the name of the callback."""
        return "gradient_collection"

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Initialize gradient collection at the start of training.

        Args:
            trainer: The trainer
            pl_module: The model.

        """
        self.gradients_history = []
        self.current_gradients = None
        self.additional_info = {}

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer) -> None:
        """
        Collect gradients before each optimizer step.
        This captures ∇F_i(w_i(t)) for the current iteration.

        Args:
            trainer: The trainer
            pl_module: The model.
            optimizer: The optimizer being used.
        """
        # Collect gradients from model parameters
        gradients = []
        for param in pl_module.parameters():
            if param.grad is not None:
                # Store the gradient as numpy array
                gradients.append(param.grad.detach().cpu().numpy())
            else:
                # If no gradient, create zero tensor with same shape
                gradients.append(np.zeros_like(param.detach().cpu().numpy()))
        
        # Store current gradients
        self.current_gradients = gradients
        # Optionally store in history for tracking over steps
        self.gradients_history.append(gradients)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Process and store collected gradients at the end of training.

        Args:
            trainer: The trainer
            pl_module: The model.

        """
        # Store the last gradients as additional info for DFedADP
        if self.current_gradients is not None:
            # Store the final gradients to be transmitted to other nodes
            self.additional_info["local_gradients"] = self.current_gradients
            # Also store gradient statistics for analysis
            grad_norms = [np.linalg.norm(grad) for grad in self.current_gradients]
            self.additional_info["gradient_norms"] = grad_norms
            # Store history for more complete analysis
            self.additional_info["gradient_history"] = self.gradients_history

    def get_info(self) -> dict[str, Any]:
        """
        Get the collected gradient information.
        
        Returns:
            Dictionary containing gradient information.
        """
        return self.additional_info

    def set_info(self, info: dict[str, Any]) -> None:
        """
        Set the gradient information.
        
        Args:
            info: Dictionary containing gradient information.
        """
        self.additional_info.update(info)