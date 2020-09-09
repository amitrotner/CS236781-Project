import torch.nn as nn


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
        # modules += [nn.MaxPool(2)]
        modules += [nn.Dropout(0.1)]
        modules += [nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)]
        modules += [nn.LeakyReLU(0.2)]
        # modules += [nn.MaxPool(2)]
        modules += [nn.Dropout(0.1)]
        modules += [nn.Conv1d(in_channels=32, out_channels=256, kernel_size=3, padding=1)]
        modules += [nn.LeakyReLU(0.2)]
        # modules += [nn.MaxPool(2)]
        modules += [nn.Dropout(0.2)]
        modules += [nn.Conv1d(in_channels=256, out_channels=64, kernel_size=3, padding=1)]
        modules += [nn.Conv1d(in_channels=64, out_channels=out_features, kernel_size=3, padding=1)]
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        x = x.unsqueeze(-1)
        y = self.cnn(x)
        return y
