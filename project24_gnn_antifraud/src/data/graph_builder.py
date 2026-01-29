"""
Graph Builder for Anti-Fraud System
Constructs heterogeneous graph from banking entities
"""

import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import HeteroData
import pickle
from tqdm import tqdm


class FraudGraphBuilder:
    """
    Builds heterogeneous graph for fraud detection
    
    Graph Schema:
        Nodes: User, Account, Device, IP, Merchant
        Edges: 
            - User -> Account (owns)
            - Account -> Device (uses)
            - Account -> IP (connects_from)
            - Account -> Merchant (transacts_with)
            - Device -> IP (connects_from)
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize graph builder
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Graph storage
        self.graph = HeteroData()
        self.nx_graph = nx.MultiDiGraph()
        
        # Entity mappings
        self.user_map = {}
        self.account_map = {}
        self.device_map = {}
        self.ip_map = {}
        self.merchant_map = {}
        
        # Node features
        self.node_features = {
            'user': [],
            'account': [],
            'device': [],
            'ip': [],
            'merchant': []
        }
        
        # Edge lists
        self.edges = {
            ('user', 'owns', 'account'): [],
            ('account', 'uses', 'device'): [],
            ('account', 'connects_from', 'ip'): [],
            ('account', 'transacts_with', 'merchant'): [],
            ('device', 'connects_from', 'ip'): []
        }
        
        # Labels
        self.labels = {
            'user': [],
            'account': [],
            'device': [],
            'ip': [],
            'merchant': []
        }
        
    def add_user(
        self, 
        user_id: int, 
        age: int,
        income_level: float,
        account_age_days: int,
        num_accounts: int,
        is_fraud: bool = False
    ) -> int:
        """
        Add user node to graph
        
        Args:
            user_id: Unique user identifier
            age: User age
            income_level: Income level (normalized)
            account_age_days: Days since account creation
            num_accounts: Number of accounts owned
            is_fraud: Whether user is fraudulent
        
        Returns:
            Node index in graph
        """
        if user_id in self.user_map:
            return self.user_map[user_id]
        
        node_idx = len(self.user_map)
        self.user_map[user_id] = node_idx
        
        # User features
        features = [
            age / 100.0,  # Normalized age
            income_level,
            account_age_days / 3650.0,  # Normalized to 10 years
            num_accounts / 10.0,  # Normalized
            1.0 if is_fraud else 0.0
        ]
        
        self.node_features['user'].append(features)
        self.labels['user'].append(1 if is_fraud else 0)
        
        # Add to NetworkX graph
        self.nx_graph.add_node(f"user_{user_id}", type='user', fraud=is_fraud)
        
        return node_idx
    
    def add_account(
        self,
        account_id: int,
        user_id: int,
        account_type: str,
        balance: float,
        transaction_count: int,
        avg_transaction_amount: float,
        is_fraud: bool = False
    ) -> int:
        """
        Add account node and link to user
        
        Args:
            account_id: Unique account identifier
            user_id: Owner user ID
            account_type: Type of account (checking, savings, credit)
            balance: Current balance
            transaction_count: Total transactions
            avg_transaction_amount: Average transaction amount
            is_fraud: Whether account is fraudulent
        
        Returns:
            Node index in graph
        """
        if account_id in self.account_map:
            return self.account_map[account_id]
        
        node_idx = len(self.account_map)
        self.account_map[account_id] = node_idx
        
        # Account type encoding
        type_encoding = {
            'checking': 0.0,
            'savings': 0.33,
            'credit': 0.67,
            'loan': 1.0
        }
        
        # Account features
        features = [
            type_encoding.get(account_type, 0.5),
            np.log1p(balance) / 20.0,  # Log-normalized balance
            transaction_count / 1000.0,
            np.log1p(avg_transaction_amount) / 15.0,
            1.0 if is_fraud else 0.0
        ]
        
        self.node_features['account'].append(features)
        self.labels['account'].append(1 if is_fraud else 0)
        
        # Add to NetworkX graph
        self.nx_graph.add_node(
            f"account_{account_id}", 
            type='account', 
            fraud=is_fraud
        )
        
        # Create edge: User -> Account
        if user_id in self.user_map:
            user_idx = self.user_map[user_id]
            self.edges[('user', 'owns', 'account')].append([user_idx, node_idx])
            self.nx_graph.add_edge(
                f"user_{user_id}", 
                f"account_{account_id}",
                relation='owns'
            )
        
        return node_idx
    
    def add_device(
        self,
        device_id: str,
        account_id: int,
        device_type: str,
        os_type: str,
        browser: str,
        first_seen_days: int,
        num_accounts: int,
        is_suspicious: bool = False
    ) -> int:
        """
        Add device node and link to account
        
        Args:
            device_id: Unique device fingerprint
            account_id: Associated account ID
            device_type: Type of device (mobile, desktop, tablet)
            os_type: Operating system
            browser: Browser type
            first_seen_days: Days since first seen
            num_accounts: Number of accounts using this device
            is_suspicious: Whether device is suspicious
        
        Returns:
            Node index in graph
        """
        if device_id in self.device_map:
            # Update existing device connection
            device_idx = self.device_map[device_id]
            if account_id in self.account_map:
                account_idx = self.account_map[account_id]
                self.edges[('account', 'uses', 'device')].append([account_idx, device_idx])
                self.nx_graph.add_edge(
                    f"account_{account_id}",
                    f"device_{device_id}",
                    relation='uses'
                )
            return device_idx
        
        node_idx = len(self.device_map)
        self.device_map[device_id] = node_idx
        
        # Device type encoding
        device_encoding = {'mobile': 0.0, 'desktop': 0.5, 'tablet': 1.0}
        os_encoding = {'ios': 0.0, 'android': 0.33, 'windows': 0.67, 'mac': 1.0}
        browser_encoding = {'chrome': 0.0, 'safari': 0.33, 'firefox': 0.67, 'other': 1.0}
        
        # Device features
        features = [
            device_encoding.get(device_type, 0.5),
            os_encoding.get(os_type, 0.5),
            browser_encoding.get(browser, 0.5),
            first_seen_days / 365.0,
            num_accounts / 20.0,
            1.0 if is_suspicious else 0.0
        ]
        
        self.node_features['device'].append(features)
        self.labels['device'].append(1 if is_suspicious else 0)
        
        # Add to NetworkX graph
        self.nx_graph.add_node(
            f"device_{device_id}",
            type='device',
            suspicious=is_suspicious
        )
        
        # Create edge: Account -> Device
        if account_id in self.account_map:
            account_idx = self.account_map[account_id]
            self.edges[('account', 'uses', 'device')].append([account_idx, node_idx])
            self.nx_graph.add_edge(
                f"account_{account_id}",
                f"device_{device_id}",
                relation='uses'
            )
        
        return node_idx
    
    def add_ip(
        self,
        ip_address: str,
        account_id: int,
        device_id: str,
        country: str,
        city: str,
        is_vpn: bool,
        is_tor: bool,
        risk_score: float,
        num_accounts: int,
        is_suspicious: bool = False
    ) -> int:
        """
        Add IP address node and link to account/device
        
        Args:
            ip_address: IP address
            account_id: Associated account ID
            device_id: Associated device ID
            country: Country code
            city: City name
            is_vpn: Whether IP is from VPN
            is_tor: Whether IP is from Tor
            risk_score: IP risk score (0-1)
            num_accounts: Number of accounts from this IP
            is_suspicious: Whether IP is suspicious
        
        Returns:
            Node index in graph
        """
        if ip_address in self.ip_map:
            # Update existing IP connections
            ip_idx = self.ip_map[ip_address]
            if account_id in self.account_map:
                account_idx = self.account_map[account_id]
                self.edges[('account', 'connects_from', 'ip')].append([account_idx, ip_idx])
                self.nx_graph.add_edge(
                    f"account_{account_id}",
                    f"ip_{ip_address}",
                    relation='connects_from'
                )
            if device_id in self.device_map:
                device_idx = self.device_map[device_id]
                self.edges[('device', 'connects_from', 'ip')].append([device_idx, ip_idx])
                self.nx_graph.add_edge(
                    f"device_{device_id}",
                    f"ip_{ip_address}",
                    relation='connects_from'
                )
            return ip_idx
        
        node_idx = len(self.ip_map)
        self.ip_map[ip_address] = node_idx
        
        # Country encoding (simplified)
        country_encoding = hash(country) % 100 / 100.0
        
        # IP features
        features = [
            country_encoding,
            1.0 if is_vpn else 0.0,
            1.0 if is_tor else 0.0,
            risk_score,
            num_accounts / 50.0,
            1.0 if is_suspicious else 0.0
        ]
        
        self.node_features['ip'].append(features)
        self.labels['ip'].append(1 if is_suspicious else 0)
        
        # Add to NetworkX graph
        self.nx_graph.add_node(
            f"ip_{ip_address}",
            type='ip',
            suspicious=is_suspicious
        )
        
        # Create edges
        if account_id in self.account_map:
            account_idx = self.account_map[account_id]
            self.edges[('account', 'connects_from', 'ip')].append([account_idx, node_idx])
            self.nx_graph.add_edge(
                f"account_{account_id}",
                f"ip_{ip_address}",
                relation='connects_from'
            )
        
        if device_id in self.device_map:
            device_idx = self.device_map[device_id]
            self.edges[('device', 'connects_from', 'ip')].append([device_idx, node_idx])
            self.nx_graph.add_edge(
                f"device_{device_id}",
                f"ip_{ip_address}",
                relation='connects_from'
            )
        
        return node_idx
    
    def add_merchant(
        self,
        merchant_id: int,
        account_id: int,
        merchant_category: str,
        transaction_amount: float,
        transaction_count: int,
        avg_amount: float,
        fraud_history_rate: float,
        is_suspicious: bool = False
    ) -> int:
        """
        Add merchant node and transaction edge
        
        Args:
            merchant_id: Unique merchant identifier
            account_id: Account making transaction
            merchant_category: Merchant category (retail, food, etc.)
            transaction_amount: Current transaction amount
            transaction_count: Total transactions with this merchant
            avg_amount: Average transaction amount
            fraud_history_rate: Historical fraud rate
            is_suspicious: Whether merchant is suspicious
        
        Returns:
            Node index in graph
        """
        if merchant_id in self.merchant_map:
            # Update transaction edge
            merchant_idx = self.merchant_map[merchant_id]
            if account_id in self.account_map:
                account_idx = self.account_map[account_id]
                self.edges[('account', 'transacts_with', 'merchant')].append([account_idx, merchant_idx])
                self.nx_graph.add_edge(
                    f"account_{account_id}",
                    f"merchant_{merchant_id}",
                    relation='transacts_with',
                    amount=transaction_amount
                )
            return merchant_idx
        
        node_idx = len(self.merchant_map)
        self.merchant_map[merchant_id] = node_idx
        
        # Category encoding
        category_encoding = {
            'retail': 0.0,
            'food': 0.2,
            'entertainment': 0.4,
            'travel': 0.6,
            'utilities': 0.8,
            'other': 1.0
        }
        
        # Merchant features
        features = [
            category_encoding.get(merchant_category, 0.5),
            np.log1p(transaction_amount) / 15.0,
            transaction_count / 1000.0,
            np.log1p(avg_amount) / 15.0,
            fraud_history_rate,
            1.0 if is_suspicious else 0.0
        ]
        
        self.node_features['merchant'].append(features)
        self.labels['merchant'].append(1 if is_suspicious else 0)
        
        # Add to NetworkX graph
        self.nx_graph.add_node(
            f"merchant_{merchant_id}",
            type='merchant',
            suspicious=is_suspicious
        )
        
        # Create transaction edge
        if account_id in self.account_map:
            account_idx = self.account_map[account_id]
            self.edges[('account', 'transacts_with', 'merchant')].append([account_idx, merchant_idx])
            self.nx_graph.add_edge(
                f"account_{account_id}",
                f"merchant_{merchant_id}",
                relation='transacts_with',
                amount=transaction_amount
            )
        
        return node_idx
    
    def build_pyg_graph(self) -> HeteroData:
        """
        Build PyTorch Geometric HeteroData object
        
        Returns:
            HeteroData object with nodes, edges, and features
        """
        print("Building PyTorch Geometric graph...")
        
        # Convert node features to tensors
        for node_type in ['user', 'account', 'device', 'ip', 'merchant']:
            if len(self.node_features[node_type]) > 0:
                self.graph[node_type].x = torch.tensor(
                    self.node_features[node_type],
                    dtype=torch.float
                )
                self.graph[node_type].y = torch.tensor(
                    self.labels[node_type],
                    dtype=torch.long
                )
                print(f"  {node_type}: {len(self.node_features[node_type])} nodes")
        
        # Convert edges to tensors
        for edge_type, edge_list in self.edges.items():
            if len(edge_list) > 0:
                edge_array = np.array(edge_list).T
                self.graph[edge_type].edge_index = torch.tensor(
                    edge_array,
                    dtype=torch.long
                )
                print(f"  {edge_type}: {len(edge_list)} edges")
        
        return self.graph
    
    def save(self, filepath: str):
        """Save graph to disk"""
        print(f"Saving graph to {filepath}...")
        
        data = {
            'pyg_graph': self.graph,
            'nx_graph': self.nx_graph,
            'mappings': {
                'user': self.user_map,
                'account': self.account_map,
                'device': self.device_map,
                'ip': self.ip_map,
                'merchant': self.merchant_map
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Graph saved successfully")
    
    @staticmethod
    def load(filepath: str) -> 'FraudGraphBuilder':
        """Load graph from disk"""
        print(f"Loading graph from {filepath}...")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        builder = FraudGraphBuilder()
        builder.graph = data['pyg_graph']
        builder.nx_graph = data['nx_graph']
        builder.user_map = data['mappings']['user']
        builder.account_map = data['mappings']['account']
        builder.device_map = data['mappings']['device']
        builder.ip_map = data['mappings']['ip']
        builder.merchant_map = data['mappings']['merchant']
        
        print(f"✓ Graph loaded successfully")
        return builder
    
    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        stats = {
            'num_nodes': {
                'user': len(self.user_map),
                'account': len(self.account_map),
                'device': len(self.device_map),
                'ip': len(self.ip_map),
                'merchant': len(self.merchant_map),
                'total': (len(self.user_map) + len(self.account_map) + 
                         len(self.device_map) + len(self.ip_map) + 
                         len(self.merchant_map))
            },
            'num_edges': {
                edge_type: len(edge_list) 
                for edge_type, edge_list in self.edges.items()
            },
            'fraud_ratio': {
                node_type: np.mean(labels) if len(labels) > 0 else 0.0
                for node_type, labels in self.labels.items()
            }
        }
        
        # NetworkX stats
        if len(self.nx_graph) > 0:
            stats['networkx'] = {
                'nodes': self.nx_graph.number_of_nodes(),
                'edges': self.nx_graph.number_of_edges(),
                'density': nx.density(self.nx_graph),
            }
        
        return stats
