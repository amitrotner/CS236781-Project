import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans


# Clustering layer definition (see DCEC article for equations)
class ClusteringLayer(nn.Module):
    def __init__(self, in_features=10, out_features=10, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(1) - self.weight
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha + 1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, alpha={}'.format(
            self.in_features, self.out_features, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)


# Function forwarding data through network, collecting clustering weight output and returning prediciotns and labels
def calculate_predictions(model, dataloader, device):
    output_array = None
    label_array = None
    model.eval()
    for data in dataloader:
        dimension = data.shape[1] - 1
        inputs = data[:, list(range(dimension))]
        labels = data[:, dimension].unsqueeze(-1).long()
        inputs = inputs.to(device)
        labels = labels.to(device)
        _, outputs, _ = model(inputs)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
            label_array = np.concatenate((label_array, labels.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
            label_array = labels.cpu().detach().numpy()

    preds = np.argmax(output_array.data, axis=1)
    # print(output_array.shape)
    return output_array, label_array, preds


# Calculate target distribution
def target(out_distr):
    tar_dist = out_distr ** 2 / np.sum(out_distr, axis=0)
    tar_dist = np.transpose(np.transpose(tar_dist) / np.sum(tar_dist, axis=1))
    return tar_dist


# K-means clusters initialisation
def kmeans(model, dataloader, device):
    km = KMeans(n_clusters=model.num_clusters, n_init=20)
    output_array = None
    model.eval()
    # Iterate through the data and concatenate the latent space representations
    for data in dataloader:
        dimension = data.shape[1] - 1
        inputs = data[:, list(range(dimension))]
        inputs = inputs.to(device)
        _, _, outputs = model(inputs)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
        # print(output_array.shape)
        if output_array.shape[0] > 50000:
            break

    # Perform K-means
    km.fit_predict(output_array)
    # Update clustering layer weights
    weights = torch.from_numpy(km.cluster_centers_)
    model.clustering_layer.set_weight(weights.to(device))
    # torch.cuda.empty_cache()
