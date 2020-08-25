import torch
import itertools as it
import torch.nn as nn


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes: int):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ReLU)*P -> MaxPool]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs should exist at the end, without a MaxPool after them.
        # ====== YOUR CODE: ======
        temp_size = list(self.in_size)
        last_in_channel = in_channels
        iters = [iter(self.channels)] * self.pool_every
        for channels in it.zip_longest(*iters, fillvalue=None):
            for channel in channels:
                if channel is None:
                    last_in_channel = None
                    break
                else:
                    layers += [nn.Conv2d(in_channels=last_in_channel, out_channels=channel, kernel_size=3, padding=1)]
                    layers += [nn.ReLU()]
                    last_in_channel = channel
            if last_in_channel is None:
                break
            else:
                layers += [nn.MaxPool2d(kernel_size=2)]
                temp_size[1] /= 2
                temp_size[2] /= 2
        temp_size[0] = self.channels[-1]
        self.in_size = tuple(temp_size)
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        #  (Linear -> ReLU)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        last_in_channel = int(in_channels * in_h * in_w)
        for channel in self.hidden_dims:
            layers += [nn.Linear(last_in_channel, channel)]
            layers += [nn.ReLU()]
            last_in_channel = channel
        layers += [nn.Linear(last_in_channel, self.out_classes)]
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(self, in_channels: int, channels: list, kernel_sizes: list,
                 batchnorm=False, dropout=0.):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
        convolution in the block. The length determines the number of
        convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
        be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
        convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
        Zero means don't apply dropout.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order). Should end with a
        #    final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use. This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        main_path_layers = []
        last_in_channel = in_channels
        for channel, kernel_size in zip(channels[:-1], kernel_sizes[:-1]):
            main_path_layers += [
                nn.Conv2d(in_channels=last_in_channel, out_channels=channel, kernel_size=kernel_size,
                          padding=int((kernel_size - 1) / 2),
                          bias=True)]

            if dropout > 0:
                main_path_layers += [nn.Dropout2d(dropout)]
            if batchnorm:
                main_path_layers += [nn.BatchNorm2d(channel)]

            main_path_layers += [nn.ReLU()]
            last_in_channel = channel
        main_path_layers += [
            nn.Conv2d(in_channels=last_in_channel, out_channels=channels[-1], kernel_size=kernel_sizes[-1],
                      padding=int((kernel_sizes[-1] - 1) / 2),
                      bias=True)]
        self.main_path = nn.Sequential(*main_path_layers)
        if in_channels != channels[-1]:
            self.shortcut_path = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=channels[-1], kernel_size=1, bias=False))
        else:
            self.shortcut_path = nn.Sequential(nn.Identity())
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out

