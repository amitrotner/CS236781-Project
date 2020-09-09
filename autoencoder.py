import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []
        modules += [nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1)]
        modules += [nn.LeakyReLU(0.2)]
        #modules += [nn.Dropout(0.1)]
        modules += [nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)]
        modules += [nn.LeakyReLU(0.2)]
        #modules += [nn.Dropout(0.1)]
        modules += [nn.Conv1d(in_channels=64, out_channels=out_channels, kernel_size=3, padding=1)]
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        x = x.unsqueeze(-1)
        return self.cnn(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        modules += [nn.ConvTranspose1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)]
        modules += [nn.LeakyReLU(0.2)]
        #modules += [nn.Dropout(0.2)]
        modules += [nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)]
        modules += [nn.LeakyReLU(0.2)]
        #modules += [nn.Dropout(0.2)]
        modules += [nn.ConvTranspose1d(in_channels=128, out_channels=out_channels, kernel_size=3, padding=1)]

        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        x_rec = self.cnn(h)
        x_rec = x_rec.squeeze(-1)
        return x_rec


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, clustering_layer, num_clusters):
        """
        :param encoder: Instance of an encoder the extracts features
        from an input.
        :param decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.encoder = encoder
        self.clustering_layer = clustering_layer
        self.decoder = decoder
        self.num_clusters = num_clusters

    def encode(self, x):
        device = next(self.parameters()).device
        z = self.encoder(x).to(device)
        return z

    def decode(self, z):
        device = next(self.parameters()).device
        x_rec = self.decoder(z).to(device)
        return x_rec

    def forward(self, x):
        z = self.encode(x)
        clustering_out = self.clustering_layer(z.squeeze(-1))
        x_rec = self.decode(z)
        return x_rec, clustering_out, z.squeeze(-1)
