"""
Training Pipeline for GraphSAGE Model
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.graph_builder import FraudGraphBuilder
from src.models.graphsage import GraphSAGE, HeteroGraphSAGE
from src.utils.metrics import compute_metrics, print_metrics


def train_epoch(model, data, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x.to(device), data.edge_index.to(device))
    
    # Compute loss on training nodes
    loss = criterion(out[data.train_mask], data.y[data.train_mask].to(device))
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask, device):
    """Evaluate model"""
    model.eval()
    
    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out[mask].argmax(dim=-1)
    
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred.cpu().numpy()
    y_score = F.softmax(out[mask], dim=-1)[:, 1].cpu().numpy()
    
    metrics = compute_metrics(y_true, y_pred, y_score)
    
    return metrics


def train_graphsage(args):
    """Main training function"""
    
    print("="*80)
    print("GraphSAGE Training for Fraud Detection")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load graph
    print(f"\nLoading graph from {args.data}...")
    builder = FraudGraphBuilder.load(args.data)
    graph = builder.graph
    
    # For simplicity, train on account nodes
    # In practice, you'd train on all node types with HeteroGraphSAGE
    node_type = 'account'
    
    if node_type not in graph.node_types:
        raise ValueError(f"Node type '{node_type}' not found in graph")
    
    # Get data
    data = graph[node_type]
    x = data.x
    y = data.y
    
    # Get edge index (use 'user', 'owns', 'account' edges)
    edge_type = ('user', 'owns', 'account')
    if edge_type in graph.edge_types:
        edge_index = graph[edge_type].edge_index
    else:
        print(f"Warning: Edge type {edge_type} not found. Using empty edges.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    print(f"\nData statistics:")
    print(f"  Nodes: {x.size(0)}")
    print(f"  Features: {x.size(1)}")
    print(f"  Edges: {edge_index.size(1)}")
    print(f"  Classes: {y.max().item() + 1}")
    print(f"  Fraud ratio: {y.float().mean():.2%}")
    
    # Create train/val/test split
    num_nodes = x.size(0)
    indices = torch.randperm(num_nodes)
    
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    # Add masks to data
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.edge_index = edge_index
    
    print(f"\nSplit:")
    print(f"  Train: {train_mask.sum()} ({train_mask.sum()/num_nodes:.1%})")
    print(f"  Val:   {val_mask.sum()} ({val_mask.sum()/num_nodes:.1%})")
    print(f"  Test:  {test_mask.sum()} ({test_mask.sum()/num_nodes:.1%})")
    
    # Create model
    print(f"\nInitializing GraphSAGE model...")
    model = GraphSAGE(
        in_channels=x.size(1),
        hidden_channels=args.hidden_dim,
        out_channels=2,  # Binary classification
        num_layers=args.num_layers,
        dropout=args.dropout,
        aggregator=args.aggregator
    ).to(device)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    # Loss function (with class weights for imbalanced data)
    class_counts = torch.bincount(y)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * 2.0
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    print(f"  Class weights: {class_weights.tolist()}")
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_f1': [],
        'val_auc': [],
        'learning_rate': []
    }
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, data, optimizer, criterion, device)
        
        # Evaluate
        val_metrics = evaluate(model, data, data.val_mask, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc_roc'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Scheduler step
        scheduler.step(val_metrics['f1'])
        
        # Print progress
        if epoch % args.log_interval == 0:
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc_roc']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['f1'],
                'val_auc': val_metrics['auc_roc']
            }, os.path.join(args.output_dir, 'graphsage_best.pth'))
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print(f"\nBest epoch: {best_epoch} (Val F1: {best_val_f1:.4f})")
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.output_dir, 'graphsage_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    print("\nFinal Evaluation:")
    print("-" * 80)
    
    train_metrics = evaluate(model, data, data.train_mask, device)
    val_metrics = evaluate(model, data, data.val_mask, device)
    test_metrics = evaluate(model, data, data.test_mask, device)
    
    print("\nTrain Set:")
    print_metrics(train_metrics)
    
    print("\nValidation Set:")
    print_metrics(val_metrics)
    
    print("\nTest Set:")
    print_metrics(test_metrics)
    
    # Save results
    results = {
        'model': 'GraphSAGE',
        'config': vars(args),
        'best_epoch': best_epoch,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'history': history
    }
    
    with open(os.path.join(args.output_dir, 'graphsage_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"  Best model: {os.path.join(args.output_dir, 'graphsage_best.pth')}")
    print(f"  Results: {os.path.join(args.output_dir, 'graphsage_results.json')}")


def main():
    parser = argparse.ArgumentParser(description="Train GraphSAGE for fraud detection")
    
    # Data
    parser.add_argument('--data', type=str, required=True, help='Path to graph data')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory')
    
    # Model
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--aggregator', type=str, default='mean', 
                       choices=['mean', 'max', 'lstm'], help='Aggregator type')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train
    train_graphsage(args)


if __name__ == "__main__":
    main()
