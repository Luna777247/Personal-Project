"""
Graph Attention Networks (GAT) for Fraud Detection
Implementation of "Graph Attention Networks" (ICLR 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.data import HeteroData
from typing import Dict, Optional, Tuple, List


class GAT(nn.Module):
    """
    Graph Attention Network for node classification
    
    Paper: Veličković et al. "Graph Attention Networks"
    https://arxiv.org/abs/1710.10903
    
    Key Features:
    - Attention mechanism: Learn importance of neighbors
    - Multi-head attention: Capture multiple relationship patterns
    - Better interpretability: Can visualize attention weights
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        attention_dropout: float = 0.1,
        use_gatv2: bool = False,
        concat_heads: bool = True
    ):
        """
        Initialize GAT model
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output dimension (num classes)
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            attention_dropout: Attention coefficient dropout
            use_gatv2: Use GATv2 (improved) instead of GAT
            concat_heads: Concatenate heads (True) or average (False)
        """
        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        
        # Select GAT version
        GATLayer = GATv2Conv if use_gatv2 else GATConv
        
        # GAT layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATLayer(
                in_channels,
                hidden_channels,
                heads=num_heads,
                dropout=attention_dropout,
                concat=concat_heads
            )
        )
        
        # Hidden layers
        input_dim = hidden_channels * num_heads if concat_heads else hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(
                GATLayer(
                    input_dim,
                    hidden_channels,
                    heads=num_heads,
                    dropout=attention_dropout,
                    concat=concat_heads
                )
            )
        
        # Last layer (average heads for final representation)
        self.convs.append(
            GATLayer(
                input_dim,
                hidden_channels,
                heads=num_heads,
                dropout=attention_dropout,
                concat=False  # Average heads
            )
        )
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(input_dim if i > 0 else hidden_channels * num_heads)
            for i in range(num_layers - 1)
        ] + [nn.BatchNorm1d(hidden_channels)])
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # Store attention weights for visualization
        self.attention_weights = []
        
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
        return_embeddings: bool = False,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            return_embeddings: Return node embeddings instead of logits
            return_attention: Return attention weights
        
        Returns:
            Node predictions and optionally attention weights
        """
        attention_weights = []
        
        # GAT layers
        for i, conv in enumerate(self.convs):
            if return_attention:
                # Get attention weights
                x, (edge_index_att, alpha) = conv(
                    x, edge_index, return_attention_weights=True
                )
                attention_weights.append((edge_index_att, alpha))
            else:
                x = conv(x, edge_index)
            
            # Apply batch norm and activation (except last layer)
            if i < len(self.convs) - 1:
                x = self.batch_norms[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final batch norm
        x = self.batch_norms[-1](x)
        
        # Return embeddings if requested
        if return_embeddings:
            if return_attention:
                return x, attention_weights
            return x
        
        # Classification
        out = self.classifier(x)
        
        if return_attention:
            return out, attention_weights
        return out
    
    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Get node embeddings"""
        return self.forward(x, edge_index, return_embeddings=True)
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get attention weights for visualization
        
        Args:
            x: Node features
            edge_index: Edge indices
            layer_idx: Specific layer index (None for all layers)
        
        Returns:
            List of (edge_index, attention_weights) tuples
        """
        _, attention_weights = self.forward(
            x, edge_index, return_attention=True
        )
        
        if layer_idx is not None:
            return [attention_weights[layer_idx]]
        
        return attention_weights


class HeteroGAT(nn.Module):
    """
    Heterogeneous GAT for multi-type nodes
    
    Applies attention mechanism across different node and edge types
    """
    
    def __init__(
        self,
        node_types: Dict[str, int],
        edge_types: list,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        attention_dropout: float = 0.1
    ):
        """
        Initialize Heterogeneous GAT
        
        Args:
            node_types: Dict mapping node type to input dimension
            edge_types: List of edge types (src, relation, dst)
            hidden_channels: Hidden dimension
            out_channels: Output dimension
            num_layers: Number of layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
        """
        super(HeteroGAT, self).__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Node type encoders
        self.node_encoders = nn.ModuleDict()
        for node_type, in_channels in node_types.items():
            self.node_encoders[node_type] = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ELU(),
                nn.Dropout(dropout)
            )
        
        # GAT layers for each edge type
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            conv_dict = nn.ModuleDict()
            
            # Determine input dimension
            if layer_idx == 0:
                in_dim = hidden_channels
            else:
                in_dim = hidden_channels * num_heads
            
            # Last layer averages heads
            concat = (layer_idx < num_layers - 1)
            
            for edge_type in edge_types:
                src, rel, dst = edge_type
                conv_dict[f"{src}__{rel}__{dst}"] = GATConv(
                    in_dim,
                    hidden_channels,
                    heads=num_heads,
                    dropout=attention_dropout,
                    concat=concat
                )
            
            self.convs.append(conv_dict)
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.BatchNorm1d(
                    hidden_channels * num_heads if i < num_layers - 1 
                    else hidden_channels
                )
                for node_type in node_types.keys()
            })
            for i in range(num_layers)
        ])
        
        # Type-specific attention aggregation
        self.type_attention = nn.ModuleDict()
        for node_type in node_types.keys():
            self.type_attention[node_type] = nn.Linear(
                hidden_channels, 1
            )
        
        # Classifiers
        self.classifiers = nn.ModuleDict()
        for node_type in node_types.keys():
            self.classifiers[node_type] = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ELU(),
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
            x_dict: Dict of node features
            edge_index_dict: Dict of edge indices
            target_node_type: Node type to predict
            return_embeddings: Return embeddings instead of logits
        
        Returns:
            Predictions for target node type
        """
        # Encode node features
        x_encoded = {}
        for node_type, x in x_dict.items():
            x_encoded[node_type] = self.node_encoders[node_type](x)
        
        # Message passing with attention
        for layer_idx in range(self.num_layers):
            x_new = {node_type: [] for node_type in self.node_types.keys()}
            
            # Aggregate messages from each edge type
            for edge_type, edge_index in edge_index_dict.items():
                src, rel, dst = edge_type
                conv_key = f"{src}__{rel}__{dst}"
                
                if conv_key in self.convs[layer_idx]:
                    # Attention-based message passing
                    msg = self.convs[layer_idx][conv_key](
                        x_encoded[src],
                        edge_index
                    )
                    x_new[dst].append(msg)
            
            # Aggregate messages with learned type-specific attention
            for node_type in self.node_types.keys():
                if len(x_new[node_type]) > 0:
                    # Stack messages from different edge types
                    messages = torch.stack(x_new[node_type])  # [num_edge_types, num_nodes, hidden]
                    
                    # Type-specific attention weights
                    attn_scores = self.type_attention[node_type](messages)
                    attn_weights = F.softmax(attn_scores, dim=0)
                    
                    # Weighted aggregation
                    x_agg = (messages * attn_weights).sum(dim=0)
                    
                    # Residual connection
                    if layer_idx > 0:
                        x_agg = x_agg + x_encoded[node_type]
                    
                    # Batch norm and activation
                    x_agg = self.batch_norms[layer_idx][node_type](x_agg)
                    x_agg = F.elu(x_agg)
                    x_agg = F.dropout(x_agg, p=self.dropout, training=self.training)
                    
                    x_encoded[node_type] = x_agg
        
        # Get embeddings for target node type
        embeddings = x_encoded[target_node_type]
        
        if return_embeddings:
            return embeddings
        
        # Classification
        out = self.classifiers[target_node_type](embeddings)
        
        return out


class MultiHeadGAT(nn.Module):
    """
    Multi-head GAT with configurable head attention
    Allows different heads to focus on different patterns
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads_per_layer: List[int] = None,
        dropout: float = 0.3
    ):
        """
        Initialize Multi-head GAT
        
        Args:
            in_channels: Input dimension
            hidden_channels: Hidden dimension
            out_channels: Output dimension
            num_layers: Number of layers
            heads_per_layer: Number of heads for each layer
            dropout: Dropout rate
        """
        super(MultiHeadGAT, self).__init__()
        
        if heads_per_layer is None:
            heads_per_layer = [8, 8, 1]  # Default: 8 heads for hidden, 1 for output
        
        assert len(heads_per_layer) == num_layers
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers with varying heads
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=heads_per_layer[0],
                concat=True
            )
        )
        
        # Hidden layers
        for i in range(1, num_layers - 1):
            self.convs.append(
                GATConv(
                    hidden_channels * heads_per_layer[i-1],
                    hidden_channels,
                    heads=heads_per_layer[i],
                    concat=True
                )
            )
        
        # Output layer
        self.convs.append(
            GATConv(
                hidden_channels * heads_per_layer[-2],
                out_channels,
                heads=heads_per_layer[-1],
                concat=False
            )
        )
    
    def forward(self, x, edge_index):
        """Forward pass"""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


def create_gat_model(
    model_type: str,
    **kwargs
) -> nn.Module:
    """
    Factory function to create GAT models
    
    Args:
        model_type: Type of model ('standard', 'hetero', 'multihead')
        **kwargs: Model-specific arguments
    
    Returns:
        GAT model
    """
    if model_type == 'standard':
        return GAT(**kwargs)
    elif model_type == 'hetero':
        return HeteroGAT(**kwargs)
    elif model_type == 'multihead':
        return MultiHeadGAT(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
