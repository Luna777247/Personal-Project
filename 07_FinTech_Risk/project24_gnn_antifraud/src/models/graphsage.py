"""
GraphSAGE Model for Fraud Detection
Implementation of "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import HeteroData
from typing import Dict, Optional, Tuple


class GraphSAGE(nn.Module):
    """
    GraphSAGE model for node classification
    
    Paper: Hamilton et al. "Inductive Representation Learning on Large Graphs"
    https://arxiv.org/abs/1706.02216
    
    Key Features:
    - Inductive learning: Can generalize to unseen nodes
    - Neighborhood sampling: Scalable to large graphs
    - Multiple aggregators: mean, LSTM, pooling
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.3,
        aggregator: str = 'mean',
        normalize: bool = True
    ):
        """
        Initialize GraphSAGE model
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output dimension (num classes)
            num_layers: Number of GraphSAGE layers
            dropout: Dropout probability
            aggregator: Aggregation function ('mean', 'max', 'lstm')
            normalize: Whether to normalize embeddings
        """
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalize = normalize
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(
            SAGEConv(
                in_channels, 
                hidden_channels,
                aggr=aggregator,
                normalize=normalize
            )
        )
        
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(
                    hidden_channels,
                    hidden_channels,
                    aggr=aggregator,
                    normalize=normalize
                )
            )
        
        self.convs.append(
            SAGEConv(
                hidden_channels,
                hidden_channels,
                aggr=aggregator,
                normalize=normalize
            )
        )
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize model parameters"""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
        for layer in self.classifier:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            return_embeddings: If True, return node embeddings instead of logits
        
        Returns:
            Node predictions or embeddings
        """
        # GraphSAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Return embeddings if requested
        if return_embeddings:
            return x
        
        # Classification
        out = self.classifier(x)
        
        return out
    
    def get_embeddings(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Get node embeddings"""
        return self.forward(x, edge_index, return_embeddings=True)


class HeteroGraphSAGE(nn.Module):
    """
    Heterogeneous GraphSAGE for multi-type nodes
    
    Handles different node types with separate encoders
    and message passing across heterogeneous edges
    """
    
    def __init__(
        self,
        node_types: Dict[str, int],
        edge_types: list,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.3,
        aggregator: str = 'mean'
    ):
        """
        Initialize Heterogeneous GraphSAGE
        
        Args:
            node_types: Dict mapping node type to input dimension
            edge_types: List of edge types (src, relation, dst)
            hidden_channels: Hidden dimension
            out_channels: Output dimension
            num_layers: Number of layers
            dropout: Dropout rate
            aggregator: Aggregation function
        """
        super(HeteroGraphSAGE, self).__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Node type encoders (project to hidden_channels)
        self.node_encoders = nn.ModuleDict()
        for node_type, in_channels in node_types.items():
            self.node_encoders[node_type] = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # GraphSAGE layers for each edge type
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = nn.ModuleDict()
            for edge_type in edge_types:
                src, rel, dst = edge_type
                conv_dict[f"{src}__{rel}__{dst}"] = SAGEConv(
                    hidden_channels,
                    hidden_channels,
                    aggr=aggregator,
                    normalize=True
                )
            self.convs.append(conv_dict)
        
        # Batch normalization for each node type
        self.batch_norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.BatchNorm1d(hidden_channels)
                for node_type in node_types.keys()
            })
            for _ in range(num_layers)
        ])
        
        # Classifier for each node type
        self.classifiers = nn.ModuleDict()
        for node_type in node_types.keys():
            self.classifiers[node_type] = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, out_channels)
            )
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        target_node_type: str,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for heterogeneous graph
        
        Args:
            x_dict: Dict of node features {node_type: features}
            edge_index_dict: Dict of edge indices {edge_type: edge_index}
            target_node_type: Node type to predict
            return_embeddings: Return embeddings instead of logits
        
        Returns:
            Predictions for target node type
        """
        # Encode node features
        x_encoded = {}
        for node_type, x in x_dict.items():
            x_encoded[node_type] = self.node_encoders[node_type](x)
        
        # Message passing
        for layer_idx in range(self.num_layers):
            x_new = {node_type: [] for node_type in self.node_types.keys()}
            
            # Aggregate messages from each edge type
            for edge_type, edge_index in edge_index_dict.items():
                src, rel, dst = edge_type
                conv_key = f"{src}__{rel}__{dst}"
                
                if conv_key in self.convs[layer_idx]:
                    # Message passing: src -> dst
                    msg = self.convs[layer_idx][conv_key](
                        x_encoded[src],
                        edge_index
                    )
                    x_new[dst].append(msg)
            
            # Aggregate messages and apply activation
            for node_type in self.node_types.keys():
                if len(x_new[node_type]) > 0:
                    # Sum aggregation across edge types
                    x_agg = torch.stack(x_new[node_type]).sum(dim=0)
                    
                    # Add residual connection
                    x_agg = x_agg + x_encoded[node_type]
                    
                    # Batch norm and activation
                    x_agg = self.batch_norms[layer_idx][node_type](x_agg)
                    x_agg = F.relu(x_agg)
                    x_agg = F.dropout(x_agg, p=self.dropout, training=self.training)
                    
                    x_encoded[node_type] = x_agg
        
        # Get embeddings for target node type
        embeddings = x_encoded[target_node_type]
        
        if return_embeddings:
            return embeddings
        
        # Classification
        out = self.classifiers[target_node_type](embeddings)
        
        return out
    
    def get_all_embeddings(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Get embeddings for all node types"""
        embeddings = {}
        for node_type in self.node_types.keys():
            embeddings[node_type] = self.forward(
                x_dict,
                edge_index_dict,
                target_node_type=node_type,
                return_embeddings=True
            )
        return embeddings


class MiniBatchGraphSAGE(nn.Module):
    """
    GraphSAGE with mini-batch training support
    Uses neighbor sampling for scalability
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        """Initialize mini-batch GraphSAGE"""
        super(MiniBatchGraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
    
    def forward(self, x, edge_index):
        """Forward with mini-batch"""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x
    
    def inference(self, x_all, subgraph_loader):
        """
        Full-batch inference with subgraph sampling
        
        Args:
            x_all: All node features
            subgraph_loader: PyG NeighborLoader for sampling
        
        Returns:
            Full predictions
        """
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id].to(batch.x.device)
                edge_index = batch.edge_index
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x[:batch.batch_size].cpu())
            
            x_all = torch.cat(xs, dim=0)
        
        return x_all


def create_graphsage_model(
    model_type: str,
    **kwargs
) -> nn.Module:
    """
    Factory function to create GraphSAGE models
    
    Args:
        model_type: Type of model ('standard', 'hetero', 'minibatch')
        **kwargs: Model-specific arguments
    
    Returns:
        GraphSAGE model
    """
    if model_type == 'standard':
        return GraphSAGE(**kwargs)
    elif model_type == 'hetero':
        return HeteroGraphSAGE(**kwargs)
    elif model_type == 'minibatch':
        return MiniBatchGraphSAGE(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
