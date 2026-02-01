"""
Visualization utilities for GNN fraud detection
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_training_curves(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training and validation curves
    
    Args:
        history: Dictionary with train/val metrics per epoch
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # AUC-ROC
    axes[0, 1].plot(history['train_auc'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_auc'], label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC-ROC')
    axes[0, 1].set_title('AUC-ROC Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(history['train_f1'], label='Train', linewidth=2)
    axes[1, 0].plot(history['val_f1'], label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1, 1].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[str] = None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix array
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal', 'Fraud'],
        yticklabels=['Normal', 'Fraud'],
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, 
                   save_path: Optional[str] = None):
    """
    Plot ROC curve
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: AUC score
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")
    
    plt.show()


def plot_precision_recall_curve(precision: np.ndarray, recall: np.ndarray, 
                                 ap: float, save_path: Optional[str] = None):
    """
    Plot Precision-Recall curve
    
    Args:
        precision: Precision values
        recall: Recall values
        ap: Average precision score
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {ap:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PR curve to {save_path}")
    
    plt.show()


def plot_embeddings_tsne(embeddings: torch.Tensor, labels: torch.Tensor,
                          save_path: Optional[str] = None):
    """
    Plot t-SNE visualization of node embeddings
    
    Args:
        embeddings: Node embeddings (N, D)
        labels: Node labels (N,)
        save_path: Path to save figure
    """
    # Convert to numpy
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # Apply t-SNE
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    for label, color, name in [(0, 'blue', 'Normal'), (1, 'red', 'Fraud')]:
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=color, label=name, alpha=0.6, s=20
        )
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('Node Embeddings Visualization (t-SNE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")
    
    plt.show()


def plot_embeddings_pca(embeddings: torch.Tensor, labels: torch.Tensor,
                         save_path: Optional[str] = None):
    """
    Plot PCA visualization of node embeddings
    
    Args:
        embeddings: Node embeddings (N, D)
        labels: Node labels (N,)
        save_path: Path to save figure
    """
    # Convert to numpy
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    for label, color, name in [(0, 'blue', 'Normal'), (1, 'red', 'Fraud')]:
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=color, label=name, alpha=0.6, s=20
        )
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title('Node Embeddings Visualization (PCA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PCA plot to {save_path}")
    
    plt.show()


def visualize_fraud_subgraph(graph_builder, fraud_node_ids: List[int],
                              save_path: Optional[str] = None):
    """
    Visualize subgraph around fraud nodes
    
    Args:
        graph_builder: BankingGraphBuilder instance
        fraud_node_ids: List of fraud node IDs
        save_path: Path to save figure
    """
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add fraud nodes and their neighbors
    for node_id in fraud_node_ids[:10]:  # Limit to 10 for visibility
        # Add center node
        G.add_node(node_id, node_type='fraud', color='red')
        
        # Add neighbors (placeholder - implement based on graph structure)
        # for neighbor in graph_builder.get_neighbors(node_id):
        #     G.add_node(neighbor, node_type='normal', color='blue')
        #     G.add_edge(node_id, neighbor)
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    # Draw nodes
    node_colors = [G.nodes[n].get('color', 'blue') for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                           node_size=300, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title('Fraud Node Subgraph')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved fraud subgraph to {save_path}")
    
    plt.show()


def plot_attention_weights(attention_weights: torch.Tensor, 
                           node_names: List[str] = None,
                           save_path: Optional[str] = None):
    """
    Plot attention weights heatmap (for GAT)
    
    Args:
        attention_weights: Attention weight matrix (N, N)
        node_names: Optional node names
        save_path: Path to save figure
    """
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        attention_weights[:50, :50],  # Show 50x50 for visibility
        cmap='YlOrRd', cbar_kws={'label': 'Attention Weight'},
        xticklabels=node_names[:50] if node_names else False,
        yticklabels=node_names[:50] if node_names else False
    )
    
    plt.xlabel('Target Node')
    plt.ylabel('Source Node')
    plt.title('Attention Weight Heatmap (GAT)')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention heatmap to {save_path}")
    
    plt.show()


def plot_feature_importance(feature_names: List[str], importance: np.ndarray,
                             save_path: Optional[str] = None):
    """
    Plot feature importance
    
    Args:
        feature_names: List of feature names
        importance: Feature importance scores
        save_path: Path to save figure
    """
    # Sort by importance
    indices = np.argsort(importance)[::-1][:20]  # Top 20
    
    plt.figure(figsize=(10, 8))
    
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance to {save_path}")
    
    plt.show()
