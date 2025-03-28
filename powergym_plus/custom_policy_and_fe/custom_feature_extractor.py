import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
#from typing import Dict
import gymnasium as gym
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, embed_dim=64):
        node_feature_shape = observation_space["node_features"].shape
        super(GNNFeatureExtractor, self).__init__(observation_space, features_dim=embed_dim)

        self.conv1 = GCNConv(node_feature_shape[1], 128)
        self.conv2 = GCNConv(128, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        x = observations["node_features"]  # Shape: [batch_size, num_nodes, num_node_features]
        edge_index = observations["edge_index"]  # Shape: [batch_size, 2, num_edges]

        batch_size, num_nodes, num_features = x.shape
        _, _, num_edges = edge_index.shape

        print("Node features shape:", x.shape)
        print("Original edge_index shape:", edge_index.shape)

        # Reshape the node features from [batch_size, num_nodes, num_features] to [total_nodes, num_features]
        x = x.view(-1, num_features)  # Shape: [batch_size * num_nodes, num_features]

        # Reshape the edge_index from [batch_size, 2, num_edges] to [2, total_edges]
        edge_index = edge_index.view(2, -1)

        # Ensure edge_index is of type int64
        edge_index = edge_index.long()

        # Handle empty edges
        if edge_index.size(1) == 0:
            print("No edges found, skipping GCN layers.")
            # Return a tensor filled with zeros or handle it appropriately
            return torch.zeros(x.size(0), self.features_dim, device=x.device)

        # Create the `batch` tensor: [0, 0, ..., 1, 1, ..., batch_size-1, batch_size-1]
        batch = torch.arange(batch_size, device=edge_index.device).repeat_interleave(num_nodes)

        print("Processed edge_index shape:", edge_index.shape)
        print("Batch shape:", batch.shape)

        # Apply GCN layers
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        # Global mean pooling per graph in the batch
        x = global_mean_pool(x, batch)

        # Final fully connected layer
        x = torch.relu(self.fc(x))

        return x




