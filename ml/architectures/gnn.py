import torch
import torch.nn as nn

class MessagePassingLayer(nn.Module):
    """Custom Message Passing layer using raw PyTorch."""

    def __init__(self, in_node_dim: int, in_edge_dim: int, out_dim: int) -> None:
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_node_dim * 2 + in_edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_node_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # x: [num_nodes, in_node_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, in_edge_dim]
        src, dst = edge_index[0], edge_index[1]

        # Gather source and target node features for each edge
        x_src = x[src]  # [num_edges, in_node_dim]
        x_dst = x[dst]  # [num_edges, in_node_dim]

        # Compute edge message
        edge_inputs = torch.cat([x_src, x_dst, edge_attr], dim=-1)
        messages = self.msg_mlp(edge_inputs)  # [num_edges, out_dim]

        # Aggregate messages at destination nodes
        agg_messages = torch.zeros(x.size(0), messages.size(-1), device=x.device)
        agg_messages.index_add_(0, dst, messages)  # [num_nodes, out_dim]

        # Update node embeddings
        update_inputs = torch.cat([x, agg_messages], dim=-1)
        return self.update_mlp(update_inputs)  # [num_nodes, out_dim]


class GNNRouterModel(nn.Module):
    """GNN Model for scoring candidate paths based on network state and routing target."""

    def __init__(self, node_dim: int = 3, edge_dim: int = 4, hidden_dim: int = 32) -> None:
        super().__init__()
        self.conv1 = MessagePassingLayer(node_dim, edge_dim, hidden_dim)
        self.conv2 = MessagePassingLayer(hidden_dim, edge_dim, hidden_dim)
        
        # Path evaluator MLP
        self.path_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        paths: list[list[int]],
    ) -> torch.Tensor:
        # x: [num_nodes, node_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_dim]
        # paths: list of candidate paths (each path is a list of node indices)
        
        # 1. Message passing to get node embeddings
        h = self.conv1(x, edge_index, edge_attr)
        h = torch.relu(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = torch.relu(h)  # [num_nodes, hidden_dim]

        # 2. Score candidate paths
        scores = []
        for path_nodes in paths:
            if len(path_nodes) == 0:
                scores.append(torch.tensor([1e5], device=x.device))
                continue
            
            # Aggregate node embeddings along the path (mean)
            path_nodes_tensor = torch.tensor(path_nodes, dtype=torch.long, device=x.device)
            path_embedding = h[path_nodes_tensor].mean(dim=0)  # [hidden_dim]
            
            # Predict path cost
            path_cost = self.path_mlp(path_embedding)
            scores.append(path_cost)
            
        return torch.cat(scores, dim=0)  # [num_paths]
