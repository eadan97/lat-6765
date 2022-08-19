from typing import Any

import pytorch_lightning as pl
import torch


class TotalVariationLoss(pl.LightningModule):
    def __init__(self, tv_weight: float = 1e-5):
        super(TotalVariationLoss, self).__init__()
        self.save_hyperparameters()

    def forward(self, img):
        w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
        h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
        loss = self.hparams.tv_weight * (h_variance + w_variance)
        return loss
