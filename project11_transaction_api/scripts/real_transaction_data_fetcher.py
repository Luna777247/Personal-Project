#!/usr/bin/env python3
"""
Real Transaction Data Fetcher for Project 11: Transaction Management API

This module generates realistic transaction data for the Spring Boot API.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTransactionDataFetcher:
    """Generates realistic transaction data"""

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)

    def generate_mock_transaction_data(self, num_transactions: int = 500) -> List[Dict[str, Any]]:
        """Generate realistic mock transaction data for demonstration"""
        logger.info(f"Generating {num_transactions} mock transactions...")

        # Sample data
        descriptions = [
            'Online Purchase', 'Grocery Shopping', 'Gas Station', 'Restaurant Payment',
            'Utility Bill', 'Rent Payment', 'Salary Deposit', 'ATM Withdrawal',
            'Online Transfer', 'Mobile Payment', 'Subscription Service', 'Insurance Premium'
        ]

        currencies = ['USD', 'EUR', 'GBP', 'CAD', 'AUD']
        statuses = ['COMPLETED', 'PENDING', 'FAILED', 'CANCELLED']
        payment_methods = ['credit_card', 'debit_card', 'bank_transfer', 'paypal', 'cash']

        transactions = []
        base_date = datetime.now() - timedelta(days=365)

        for i in range(num_transactions):
            # Random date within the last year
            days_offset = random.randint(0, 365)
            transaction_date = base_date + timedelta(days=days_offset)

            # Random amount between $1 and $5000
            amount = round(random.uniform(1, 5000), 2)

            transaction = {
                'id': f"mock_txn_{i+1:06d}",
                'amount': amount,
                'currency': random.choice(currencies),
                'description': random.choice(descriptions),
                'status': random.choices(statuses, weights=[0.8, 0.1, 0.05, 0.05])[0],
                'transaction_date': transaction_date.isoformat(),
                'payment_method': random.choice(payment_methods),
                'source': 'MOCK_DATA'
            }
            transactions.append(transaction)

        logger.info(f"✅ Generated {len(transactions)} mock transactions")
        return transactions

    def save_transactions_to_csv(self, transactions: List[Dict[str, Any]], filename: str = 'real_transactions.csv'):
        """Save transactions to CSV file"""
        filepath = os.path.join(self.data_dir, filename)
        df = pd.DataFrame(transactions)
        df.to_csv(filepath, index=False)
        logger.info(f"💾 Saved {len(transactions)} transactions to {filepath}")

    def save_transactions_to_json(self, transactions: List[Dict[str, Any]], filename: str = 'real_transactions.json'):
        """Save transactions to JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(transactions, f, indent=2)
        logger.info(f"💾 Saved {len(transactions)} transactions to {filepath}")

    def create_api_import_script(self, transactions: List[Dict[str, Any]], filename: str = 'import_transactions.py'):
        """Create a Python script to import transactions into the Spring Boot API"""
        # Create the script content as a regular string
        transactions_json = json.dumps(transactions[:50], indent=2)

        script_content = '''#!/usr/bin/env python3
"""
Transaction Import Script for Spring Boot Transaction API

This script imports transaction data into the running Spring Boot application.
Make sure the API is running on http://localhost:8080 before running this script.
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = 'http://localhost:8080'
API_URL = f'{BASE_URL}/api'

def create_user(username: str, email: str, password: str, first_name: str, last_name: str):
    """Create a user via API"""
    user_data = {
        "username": username,
        "email": email,
        "password": password,
        "firstName": first_name,
        "lastName": last_name
    }

    response = requests.post(f'{API_URL}/auth/signup', json=user_data)
    if response.status_code == 200:
        print(f"✅ Created user: {username}")
        return response.json()
    else:
        print(f"❌ Failed to create user {username}: {response.text}")
        return None

def login_user(username: str, password: str):
    """Login and get JWT token"""
    login_data = {
        "username": username,
        "password": password
    }

    response = requests.post(f'{API_URL}/auth/signin', json=login_data)
    if response.status_code == 200:
        token = response.json().get('token')
        print(f"✅ Logged in as: {username}")
        return token
    else:
        print(f"❌ Failed to login {username}: {response.text}")
        return None

def create_transaction(token: str, transaction_data: dict):
    """Create a transaction via API"""
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    # Convert transaction data to API format
    api_data = {
        "amount": transaction_data["amount"],
        "currency": transaction_data["currency"],
        "description": transaction_data["description"],
        "transactionType": "DEBIT" if transaction_data["amount"] < 0 else "CREDIT",
        "status": transaction_data["status"]
    }

    response = requests.post(f'{API_URL}/transactions', json=api_data, headers=headers)
    if response.status_code == 201:
        print(f"✅ Created transaction: {transaction_data['id']}")
        return True
    else:
        print(f"❌ Failed to create transaction {transaction_data['id']}: {response.text}")
        return False

def main():
    """Main import function"""
    print("🚀 Starting transaction import to Spring Boot API...")
    print("=" * 60)

    # Load transaction data
    transactions = ''' + transactions_json + '''

    # Create demo user
    user = create_user("demo_user", "demo@example.com", "password123", "Demo", "User")
    if not user:
        print("❌ Cannot proceed without user")
        return

    # Login to get token
    token = login_user("demo_user", "password123")
    if not token:
        print("❌ Cannot proceed without authentication")
        return

    # Import transactions
    success_count = 0
    for i, txn in enumerate(transactions):
        print(f"📝 Importing transaction {i+1}/{len(transactions)}: {txn['id']}")
        if create_transaction(token, txn):
            success_count += 1
        time.sleep(0.1)  # Small delay to avoid overwhelming the API

    print("\\n" + "=" * 60)
    print(f"✅ Import completed! Successfully imported {success_count}/{len(transactions)} transactions")
    print("🌐 API available at: http://localhost:8080")

if __name__ == "__main__":
    main()
'''

        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w') as f:
            f.write(script_content)
        logger.info(f"📝 Created API import script: {filepath}")

def main():
    """Main function to generate and save transaction data"""
    fetcher = RealTransactionDataFetcher()

    print("🚀 Real Transaction Data Fetcher for Project 11")
    print("=" * 50)

    # Generate mock transaction data
    transactions = fetcher.generate_mock_transaction_data(200)

    if transactions:
        # Save to files
        fetcher.save_transactions_to_csv(transactions)
        fetcher.save_transactions_to_json(transactions)

        # Create API import script
        fetcher.create_api_import_script(transactions)

        print(f"\n✅ Successfully generated {len(transactions)} transactions!")
        print("📁 Data saved to data/ directory")
        print("🔧 Run 'python data/import_transactions.py' to import into Spring Boot API")
    else:
        print("❌ No transaction data could be generated")

if __name__ == "__main__":
    main()