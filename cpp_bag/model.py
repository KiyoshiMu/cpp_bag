from __future__ import annotations

from enum import Enum
from pathlib import Path

import torch
from torch import nn
from torch import Tensor

from .metric_trainer import TrainTask
from hflayers import HopfieldPooling


class Pooling(Enum):
    Hopfield = 1
    SetTrans = 2


class BagPooling(nn.Module):
    def __init__(self, in_dim=256, out_dim=256):
        """
        Initialize a new instance of a Hopfield-based pooling network.

        Note: all hyperparameters of the network are fixed for demonstration purposes.
        Morevover, most of the notation of the original implementation is kept in order
        to be easier comparable (partially ignoring PEP8).
        """
        super().__init__()
        self.L = 128
        self.pooling = HopfieldPooling(
            input_size=in_dim,
            hidden_size=32,
            output_size=self.L,
            num_heads=4,
            update_steps_max=3,
            scaling=0.25,
            dropout=0.2,
        )

        self.fc = nn.Sequential(
            nn.Linear(self.L, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, embeds: Tensor):
        """
        Compute result of Hopfield-based pooling network on specified data.

        :param input: data to be processed by the Hopfield-based pooling network
        :return: result as computed by the Hopfield-based pooling network
        """

        H = self.pooling(embeds)
        Y_logit = self.fc(H)

        return Y_logit

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    @classmethod
    def from_checkpoint(cls, p, in_dim=256, out_dim=256):
        m = cls(in_dim=in_dim, out_dim=out_dim)
        m.load_state_dict(torch.load(p, map_location=torch.device("cpu")))
        return m


def encoder_training(
    train_ds,
    in_dim=256,
    dst_dir=Path("."),
    num_epochs=50,
    num_workers=1,
):
    """
    Train the encoder model.
    """
    model = BagPooling(in_dim=in_dim, out_dim=256)
    task = TrainTask(
        train_ds,
        model,
        num_workers=num_workers,
        batch_size=64,
        patience=3,
    )
    task.run(num_epochs)
    model_path = str(dst_dir / f"pool-{task.timestamp}.pth")
    task.save_model(model_path)
    return model_path
