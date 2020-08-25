from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


class ConvNet(nn.Module):
    def __init__(self, in_size, out_features):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size

        modules = []
        modules += [nn.Conv1d(in_channels=in_size, out_channels=16, kernel_size=3, padding=1)]
        modules += [nn.LeakyReLU(0.2)]
        #modules += [nn.MaxPool(2)]
        modules += [nn.Dropout(0.1)]
        modules += [nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)]
        modules += [nn.LeakyReLU(0.2)]
        #modules += [nn.MaxPool(2)]
        modules += [nn.Dropout(0.1)]
        modules += [nn.Conv1d(in_channels=32, out_channels=256, kernel_size=3, padding=1)]
        modules += [nn.LeakyReLU(0.2)]
        #modules += [nn.MaxPool(2)]
        modules += [nn.Dropout(0.2)]
        modules += [nn.Conv1d(in_channels=256, out_channels=64, kernel_size=3, padding=1)]
        modules += [nn.Conv1d(in_channels=64, out_channels=out_features, kernel_size=3, padding=1)]
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        x = x.unsqueeze(-1)
        y = self.cnn(x)
        # ========================
        return y


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f'{checkpoint_file}.pt'

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    torch.save(gen_model, checkpoint_file)
    saved = True
    # ========================

    return saved
