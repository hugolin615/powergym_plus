import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import gymnasium as gym

class CombinedGCN(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, num_actions: int, embed_dim=64):
        super(CombinedGCN, self).__init__()
        node_feature_shape = observation_space["node_features"].shape
        num_node_features = node_feature_shape[1]
        
        # GCN Layers
        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, embed_dim) # 64 by default
        
        # Global Pooling and Final Linear Layers
        self.global_pool = global_mean_pool #single vector to represent whole graph
        self.fc = nn.Linear(embed_dim, embed_dim) #linear layer
        
        # Output Layer for Q-values
        self.output_layer = nn.Linear(embed_dim, num_actions) #Q-values for each action, with the number of outputs equal to the number of actions
        
        # Set the features_dim attribute
        self.features_dim = num_actions

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        x = observations["node_features"]  # Shape: [batch_size, num_nodes, num_node_features]
        edge_index = observations["edge_index"]  # Shape: [batch_size, 2, num_edges]

        batch_size, num_nodes, num_features = x.shape
        _, _, num_edges = edge_index.shape

        # Reshape node features and edge_index to make it compatible with PyG GCN
        x = x.view(-1, num_features)  # Shape: [batch_size * num_nodes, num_features]
        edge_index = edge_index.view(2, -1)  # Shape: [2, batch_size * num_edges]

        # Ensure edge_index is of type int64
        edge_index = edge_index.long()

        # Handle empty edges
        if edge_index.size(1) == 0:
            # Return a tensor filled with zeros
            return torch.zeros(x.size(0), self.output_layer.out_features, device=x.device)

        # Create the `batch` tensor
        batch = torch.arange(batch_size, device=edge_index.device).repeat_interleave(num_nodes)

        # Apply GCN layers
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))

        # Global mean pooling per graph in the batch
        x = self.global_pool(x, batch)

        # Fully connected layer
        x = torch.relu(self.fc(x))

        # Output Q-values
        q_values = self.output_layer(x)

        return q_values
