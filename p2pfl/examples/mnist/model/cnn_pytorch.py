"""CNN model for MNIST using PyTorch Lightning."""

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, Metric

from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed


class CNN(L.LightningModule):
    """Simple CNN for MNIST classification."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        lr_rate: float = 1e-3,
        metric: type[Metric] = Accuracy,
    ) -> None:
        super().__init__()
        set_seed(Settings.general.SEED, "pytorch")
        self.lr_rate = lr_rate
        # metric configuration
        if num_classes == 1:
            self.metric = metric(task="binary")
        else:
            self.metric = metric(task="multiclass", num_classes=num_classes)

        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expecting input shape [B, C, H, W]
        # Normalize if pixels are in 0..255
        if x.dtype != torch.float32:
            x = x.float()

        # Handle common shapes coming from the dataset pipeline:
        # - [B, H, W] (no channel) -> add channel dim
        # - accidental transpose where shape is [1, B, H, W] (batch and channel swapped)
        if x.dim() == 3:
            # [B, H, W] -> [B, 1, H, W]
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            # detect swapped (1, B, H, W) where B equals DEFAULT_BATCH_SIZE
            try:
                from p2pfl.settings import Settings

                batch_hint = Settings.training.DEFAULT_BATCH_SIZE
            except Exception:
                batch_hint = 128

            if x.size(0) == 1 and x.size(1) == batch_hint:
                # Permute to [B, C, H, W]
                x = x.permute(1, 0, 2, 3).contiguous()

        if x.max() > 1.0:
            x = x / 255.0

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def training_step(self, batch: dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        x = batch["image"].float()
        if x.max() > 1.0:
            x = x / 255.0
        y = batch["label"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        # Optional: implement if needed by pipeline
        x = batch["image"].float()
        if x.max() > 1.0:
            x = x / 255.0
        y = batch["label"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        metric = self.metric(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_metric", metric, prog_bar=True)
        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        x = batch["image"].float()
        if x.max() > 1.0:
            x = x / 255.0
        y = batch["label"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        metric = self.metric(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_metric", metric, prog_bar=True)
        return loss


def model_build_fn(*args, **kwargs) -> LightningModel:
    """Factory for creating the CNN wrapped in LightningModel."""
    compression = kwargs.pop("compression", None)
    return LightningModel(CNN(*args, **kwargs), compression=compression)
