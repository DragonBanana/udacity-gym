from typing import Any

import lightning as pl
import torch
import torchinfo
import torchmetrics
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from udacity_gym.extras.model.segmentation.unet.module import UnetEncoder, UnetDecoder, PositionalEncoder


class SegmentationUnet(pl.LightningModule):

    def __init__(
            self,
            hidden_dims: list[int],
            num_groups: int = 32,
            in_channels: int = 3,
            out_channels: int = 1,
            input_shape: tuple[int, int, int] = (3, 160, 320),
            learning_rate: float = 3e-4,
            *args: Any,
            **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims
        self.input_shape = input_shape
        self.patch_size = (2 ** len(hidden_dims), 2 ** len(hidden_dims))
        self.seq_len = (input_shape[1] * input_shape[2]) // (self.patch_size[0] * self.patch_size[1])
        self.dim = hidden_dims[-1]
        self.learning_rate = learning_rate

        self._metric = torchmetrics.classification.BinaryJaccardIndex()

        self.encoder = UnetEncoder(hidden_dims=hidden_dims, num_groups=num_groups, in_channels=in_channels,
                                   input_shape=input_shape)
        self.positional_encoder = PositionalEncoder(dim=self.dim, seq_len=self.seq_len)
        self.transformer = nn.Transformer(d_model=self.dim,
                                          batch_first=True).encoder
        self.decoder = UnetDecoder(hidden_dims=hidden_dims[::-1], num_groups=num_groups,
                                   out_channels=out_channels, input_shape=input_shape)

    def forward(self, x, *args: Any, **kwargs: Any) -> Any:
        x = self.encoder(x)
        x = self.positional_encoder(x)
        x = self.transformer(x)
        x = self.decoder(x)
        return x + 1 / 2

    def _step(self, batch, batch_idx, step: str = "train"):
        x, y = batch
        y_hat = self.forward(x)

        loss = torch.nn.functional.mse_loss(y_hat, y)
        loss_dict = {
            f'{step}/loss': loss
        }
        if not self.training:
            loss_dict[f'{step}/miou'] = self._metric(y_hat, y)
        self.log_dict(
            loss_dict,
            prog_bar=True,
            on_step=self.training,
            on_epoch=not self.training
        )
        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self._step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.99)
        return {
            'optimizer': optim,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}
        }


if __name__ == '__main__':
    model_params = {
        'hidden_dims': [64, 128, 256, 512],
        'input_shape': (3, 160, 320),
        'num_groups': 32,
        'in_channels': 3,
        'out_channels': 1,
        'learning_rate': 1e-5,
    }
    model = SegmentationUnet(**model_params)
    torchinfo.summary(model, (1, 3, 160, 320), col_names=["input_size", "output_size", "num_params", "mult_adds"],
                      depth=5)
