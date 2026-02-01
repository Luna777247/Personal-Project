"""
Generate Synthetic Graph Data for Fraud Detection
Creates realistic banking transaction graph with fraud patterns
"""

import os
import sys
import argparse
import random
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.graph_builder import FraudGraphBuilder


class SyntheticDataGenerator:
    """Generate synthetic banking transaction data with fraud patterns"""
    
    def __init__(self, seed: int = 42):
        """Initialize generator"""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Fraud patterns
        self.fraud_rings = []  # List of fraud ring user IDs
        self.suspicious_devices = []
        self.suspicious_ips = []
        self.suspicious_merchants = []
    
    def generate_fraud_ring(
        self,
        ring_size: int,
        base_user_id: int
    ) -> List[int]:
        """
        Generate fraud ring (connected fraudulent users)
        
        Args:
            ring_size: Number of users in ring
            base_user_id: Starting user ID
        
        Returns:
            List of fraud ring user IDs
        """
        ring = list(range(base_user_id, base_user_id + ring_size))
        self.fraud_rings.append(ring)
        return ring
    
    def generate_users(
        self,
        num_users: int,
        fraud_ratio: float = 0.05
    ) -> List[Dict]:
        """
        Generate user data
        
        Args:
            num_users: Number of users
            fraud_ratio: Ratio of fraudulent users
        
        Returns:
            List of user dicts
        """
        users = []
        num_fraud = int(num_users * fraud_ratio)
        
        print(f"Generating {num_users} users ({num_fraud} fraudulent)...")
        
        # Normal users
        for user_id in tqdm(range(num_users - num_fraud)):
            users.append({
                'user_id': user_id,
                'age': random.randint(18, 80),
                'income_level': random.uniform(0.2, 1.0),
                'account_age_days': random.randint(30, 3650),
                'num_accounts': random.randint(1, 5),
                'is_fraud': False
            })
        
        # Fraud users (organized in rings)
        fraud_user_id = num_users - num_fraud
        ring_sizes = [random.randint(3, 15) for _ in range(num_fraud // 10)]
        
        for ring_size in ring_sizes:
            ring = self.generate_fraud_ring(ring_size, fraud_user_id)
            
            for user_id in ring:
                users.append({
                    'user_id': user_id,
                    'age': random.randint(20, 40),  # Fraudsters typically younger
                    'income_level': random.uniform(0.1, 0.4),  # Lower income
                    'account_age_days': random.randint(1, 180),  # New accounts
                    'num_accounts': random.randint(2, 10),  # Multiple accounts
                    'is_fraud': True
                })
            
            fraud_user_id += ring_size
        
        # Fill remaining fraud users (individual fraudsters)
        while fraud_user_id < num_users:
            users.append({
                'user_id': fraud_user_id,
                'age': random.randint(20, 45),
                'income_level': random.uniform(0.1, 0.5),
                'account_age_days': random.randint(1, 365),
                'num_accounts': random.randint(1, 5),
                'is_fraud': True
            })
            fraud_user_id += 1
        
        return users
    
    def generate_accounts(
        self,
        users: List[Dict],
        accounts_per_user: Tuple[int, int] = (1, 3)
    ) -> List[Dict]:
        """
        Generate account data
        
        Args:
            users: List of user dicts
            accounts_per_user: Min and max accounts per user
        
        Returns:
            List of account dicts
        """
        accounts = []
        account_id = 0
        
        print(f"Generating accounts...")
        
        for user in tqdm(users):
            num_accounts = random.randint(*accounts_per_user)
            
            # Fraudsters tend to have more accounts
            if user['is_fraud']:
                num_accounts = min(num_accounts + 2, 10)
            
            for _ in range(num_accounts):
                account_type = random.choice(['checking', 'savings', 'credit', 'loan'])
                
                # Fraud accounts have suspicious patterns
                if user['is_fraud'] and random.random() < 0.7:
                    balance = random.uniform(0, 1000)  # Low balance
                    transaction_count = random.randint(50, 500)  # High activity
                    avg_amount = random.uniform(100, 5000)  # Large amounts
                else:
                    balance = random.uniform(1000, 100000)
                    transaction_count = random.randint(10, 200)
                    avg_amount = random.uniform(20, 1000)
                
                accounts.append({
                    'account_id': account_id,
                    'user_id': user['user_id'],
                    'account_type': account_type,
                    'balance': balance,
                    'transaction_count': transaction_count,
                    'avg_transaction_amount': avg_amount,
                    'is_fraud': user['is_fraud']
                })
                
                account_id += 1
        
        return accounts
    
    def generate_devices(
        self,
        accounts: List[Dict],
        devices_per_account: Tuple[int, int] = (1, 3)
    ) -> List[Dict]:
        """Generate device data"""
        devices = []
        device_fingerprints = set()
        
        print(f"Generating devices...")
        
        for account in tqdm(accounts):
            num_devices = random.randint(*devices_per_account)
            
            # Fraudsters share devices
            if account['is_fraud'] and random.random() < 0.3:
                # Use device from fraud ring
                if len(self.suspicious_devices) > 0:
                    device_id = random.choice(self.suspicious_devices)
                    devices.append({
                        'device_id': device_id,
                        'account_id': account['account_id'],
                        'device_type': random.choice(['mobile', 'desktop', 'tablet']),
                        'os_type': random.choice(['android', 'ios', 'windows', 'mac']),
                        'browser': random.choice(['chrome', 'safari', 'firefox', 'other']),
                        'first_seen_days': random.randint(1, 90),
                        'num_accounts': len([d for d in devices if d['device_id'] == device_id]) + 1,
                        'is_suspicious': True
                    })
                    continue
            
            for _ in range(num_devices):
                device_id = f"device_{len(device_fingerprints)}"
                device_fingerprints.add(device_id)
                
                is_suspicious = account['is_fraud'] and random.random() < 0.5
                
                if is_suspicious:
                    self.suspicious_devices.append(device_id)
                
                devices.append({
                    'device_id': device_id,
                    'account_id': account['account_id'],
                    'device_type': random.choice(['mobile', 'desktop', 'tablet']),
                    'os_type': random.choice(['android', 'ios', 'windows', 'mac']),
                    'browser': random.choice(['chrome', 'safari', 'firefox', 'other']),
                    'first_seen_days': random.randint(1, 365),
                    'num_accounts': 1,
                    'is_suspicious': is_suspicious
                })
        
        return devices
    
    def generate_ips(
        self,
        accounts: List[Dict],
        devices: List[Dict],
        ips_per_account: Tuple[int, int] = (1, 5)
    ) -> List[Dict]:
        """Generate IP address data"""
        ips = []
        ip_addresses = set()
        
        print(f"Generating IP addresses...")
        
        # Group devices by account
        account_devices = {}
        for device in devices:
            account_id = device['account_id']
            if account_id not in account_devices:
                account_devices[account_id] = []
            account_devices[account_id].append(device['device_id'])
        
        for account in tqdm(accounts):
            account_id = account['account_id']
            num_ips = random.randint(*ips_per_account)
            
            # Fraudsters share IPs
            if account['is_fraud'] and random.random() < 0.4:
                if len(self.suspicious_ips) > 0:
                    ip_address = random.choice(self.suspicious_ips)
                    device_id = random.choice(account_devices.get(account_id, ['device_0']))
                    
                    ips.append({
                        'ip_address': ip_address,
                        'account_id': account_id,
                        'device_id': device_id,
                        'country': 'XX',
                        'city': 'Unknown',
                        'is_vpn': True,
                        'is_tor': random.random() < 0.3,
                        'risk_score': random.uniform(0.7, 1.0),
                        'num_accounts': len([ip for ip in ips if ip['ip_address'] == ip_address]) + 1,
                        'is_suspicious': True
                    })
                    continue
            
            for _ in range(num_ips):
                ip_address = f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"
                ip_addresses.add(ip_address)
                
                is_suspicious = account['is_fraud'] and random.random() < 0.4
                
                if is_suspicious:
                    self.suspicious_ips.append(ip_address)
                
                device_id = random.choice(account_devices.get(account_id, ['device_0']))
                
                ips.append({
                    'ip_address': ip_address,
                    'account_id': account_id,
                    'device_id': device_id,
                    'country': random.choice(['US', 'CN', 'RU', 'BR', 'IN', 'UK', 'DE']),
                    'city': 'City',
                    'is_vpn': is_suspicious and random.random() < 0.5,
                    'is_tor': is_suspicious and random.random() < 0.2,
                    'risk_score': random.uniform(0.7, 1.0) if is_suspicious else random.uniform(0.0, 0.3),
                    'num_accounts': 1,
                    'is_suspicious': is_suspicious
                })
        
        return ips
    
    def generate_merchants(
        self,
        num_merchants: int,
        fraud_ratio: float = 0.1
    ) -> List[Dict]:
        """Generate merchant data"""
        merchants = []
        num_suspicious = int(num_merchants * fraud_ratio)
        
        print(f"Generating {num_merchants} merchants ({num_suspicious} suspicious)...")
        
        categories = ['retail', 'food', 'entertainment', 'travel', 'utilities', 'other']
        
        for merchant_id in tqdm(range(num_merchants)):
            is_suspicious = merchant_id < num_suspicious
            
            if is_suspicious:
                self.suspicious_merchants.append(merchant_id)
            
            merchants.append({
                'merchant_id': merchant_id,
                'merchant_category': random.choice(categories),
                'fraud_history_rate': random.uniform(0.5, 1.0) if is_suspicious else random.uniform(0.0, 0.1),
                'is_suspicious': is_suspicious
            })
        
        return merchants
    
    def generate_transactions(
        self,
        accounts: List[Dict],
        merchants: List[Dict],
        num_transactions: int
    ) -> List[Dict]:
        """Generate transaction data"""
        transactions = []
        
        print(f"Generating {num_transactions} transactions...")
        
        for _ in tqdm(range(num_transactions)):
            account = random.choice(accounts)
            merchant = random.choice(merchants)
            
            # Fraud transactions link fraud accounts to suspicious merchants
            if account['is_fraud'] and random.random() < 0.5:
                if len(self.suspicious_merchants) > 0:
                    merchant_id = random.choice(self.suspicious_merchants)
                    merchant = next(m for m in merchants if m['merchant_id'] == merchant_id)
            
            # Transaction amount
            if account['is_fraud'] and merchant['is_suspicious']:
                amount = random.uniform(1000, 10000)  # Large fraud transactions
            else:
                amount = random.uniform(10, 1000)
            
            transactions.append({
                'account_id': account['account_id'],
                'merchant_id': merchant['merchant_id'],
                'merchant_category': merchant['merchant_category'],
                'transaction_amount': amount,
                'transaction_count': 1,
                'avg_amount': amount,
                'fraud_history_rate': merchant['fraud_history_rate'],
                'is_suspicious': merchant['is_suspicious']
            })
        
        return transactions


def main():
    """Main generation script"""
    parser = argparse.ArgumentParser(description="Generate synthetic fraud detection graph")
    parser.add_argument('--num-users', type=int, default=10000, help='Number of users')
    parser.add_argument('--num-transactions', type=int, default=50000, help='Number of transactions')
    parser.add_argument('--fraud-ratio', type=float, default=0.05, help='Ratio of fraudulent users')
    parser.add_argument('--output', type=str, default='data/graph_data.pkl', help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("=" * 80)
    print("Synthetic Fraud Detection Graph Generation")
    print("=" * 80)
    print(f"Users: {args.num_users}")
    print(f"Transactions: {args.num_transactions}")
    print(f"Fraud Ratio: {args.fraud_ratio:.1%}")
    print(f"Seed: {args.seed}")
    print()
    
    # Initialize generator
    generator = SyntheticDataGenerator(seed=args.seed)
    builder = FraudGraphBuilder(seed=args.seed)
    
    # Generate entities
    users = generator.generate_users(args.num_users, args.fraud_ratio)
    accounts = generator.generate_accounts(users)
    devices = generator.generate_devices(accounts)
    ips = generator.generate_ips(accounts, devices)
    merchants = generator.generate_merchants(num_merchants=1000, fraud_ratio=0.1)
    transactions = generator.generate_transactions(accounts, merchants, args.num_transactions)
    
    # Build graph
    print("\nBuilding graph...")
    
    # Add users
    for user in tqdm(users, desc="Adding users"):
        builder.add_user(**user)
    
    # Add accounts
    for account in tqdm(accounts, desc="Adding accounts"):
        builder.add_account(**account)
    
    # Add devices
    for device in tqdm(devices, desc="Adding devices"):
        builder.add_device(**device)
    
    # Add IPs
    for ip in tqdm(ips, desc="Adding IPs"):
        builder.add_ip(**ip)
    
    # Add merchants and transactions
    for merchant in tqdm(merchants, desc="Adding merchants"):
        merchant_id = merchant['merchant_id']
        merchant_txns = [t for t in transactions if t['merchant_id'] == merchant_id]
        
        if len(merchant_txns) > 0:
            txn = merchant_txns[0]
            builder.add_merchant(
                merchant_id=merchant_id,
                account_id=txn['account_id'],
                **{k: v for k, v in merchant.items() if k != 'merchant_id'}
            )
    
    # Build PyG graph
    graph = builder.build_pyg_graph()
    
    # Save graph
    builder.save(args.output)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Graph Statistics")
    print("=" * 80)
    
    stats = builder.get_statistics()
    
    print("\nNodes:")
    for node_type, count in stats['num_nodes'].items():
        if node_type != 'total':
            fraud_ratio = stats['fraud_ratio'].get(node_type, 0.0)
            print(f"  {node_type:12s}: {count:8d} ({fraud_ratio:.2%} fraud)")
    print(f"  {'Total':12s}: {stats['num_nodes']['total']:8d}")
    
    print("\nEdges:")
    total_edges = 0
    for edge_type, count in stats['num_edges'].items():
        src, rel, dst = edge_type
        print(f"  {src} --{rel}--> {dst}: {count:8d}")
        total_edges += count
    print(f"  {'Total':35s}: {total_edges:8d}")
    
    if 'networkx' in stats:
        print(f"\nNetworkX Stats:")
        print(f"  Density: {stats['networkx']['density']:.6f}")
    
    print("\n✓ Graph generation complete!")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
