#!/usr/bin/env python3
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
    transactions = [
  {
    "id": "mock_txn_000001",
    "amount": 1966.45,
    "currency": "AUD",
    "description": "Utility Bill",
    "status": "COMPLETED",
    "transaction_date": "2025-06-29T11:07:13.236061",
    "payment_method": "cash",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000002",
    "amount": 3402.56,
    "currency": "GBP",
    "description": "Grocery Shopping",
    "status": "COMPLETED",
    "transaction_date": "2025-07-02T11:07:13.236061",
    "payment_method": "debit_card",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000003",
    "amount": 4055.97,
    "currency": "EUR",
    "description": "Rent Payment",
    "status": "COMPLETED",
    "transaction_date": "2025-05-31T11:07:13.236061",
    "payment_method": "bank_transfer",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000004",
    "amount": 123.81,
    "currency": "GBP",
    "description": "Gas Station",
    "status": "COMPLETED",
    "transaction_date": "2025-10-30T11:07:13.236061",
    "payment_method": "cash",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000005",
    "amount": 1692.92,
    "currency": "USD",
    "description": "Grocery Shopping",
    "status": "FAILED",
    "transaction_date": "2025-12-09T11:07:13.236061",
    "payment_method": "debit_card",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000006",
    "amount": 241.79,
    "currency": "USD",
    "description": "Grocery Shopping",
    "status": "COMPLETED",
    "transaction_date": "2025-01-26T11:07:13.236061",
    "payment_method": "paypal",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000007",
    "amount": 969.35,
    "currency": "GBP",
    "description": "Utility Bill",
    "status": "COMPLETED",
    "transaction_date": "2025-09-21T11:07:13.236061",
    "payment_method": "credit_card",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000008",
    "amount": 1276.51,
    "currency": "AUD",
    "description": "Restaurant Payment",
    "status": "COMPLETED",
    "transaction_date": "2024-12-11T11:07:13.236061",
    "payment_method": "paypal",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000009",
    "amount": 4147.97,
    "currency": "GBP",
    "description": "Gas Station",
    "status": "COMPLETED",
    "transaction_date": "2025-02-11T11:07:13.236061",
    "payment_method": "bank_transfer",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000010",
    "amount": 3940.75,
    "currency": "USD",
    "description": "Insurance Premium",
    "status": "COMPLETED",
    "transaction_date": "2025-02-02T11:07:13.236061",
    "payment_method": "credit_card",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000011",
    "amount": 2804.2,
    "currency": "CAD",
    "description": "Mobile Payment",
    "status": "COMPLETED",
    "transaction_date": "2025-11-18T11:07:13.236061",
    "payment_method": "paypal",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000012",
    "amount": 3594.36,
    "currency": "AUD",
    "description": "Grocery Shopping",
    "status": "COMPLETED",
    "transaction_date": "2024-12-31T11:07:13.236061",
    "payment_method": "cash",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000013",
    "amount": 1257.26,
    "currency": "USD",
    "description": "Online Purchase",
    "status": "COMPLETED",
    "transaction_date": "2025-11-24T11:07:13.236061",
    "payment_method": "cash",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000014",
    "amount": 238.32,
    "currency": "USD",
    "description": "Mobile Payment",
    "status": "COMPLETED",
    "transaction_date": "2025-10-20T11:07:13.236061",
    "payment_method": "bank_transfer",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000015",
    "amount": 3801.0,
    "currency": "EUR",
    "description": "ATM Withdrawal",
    "status": "COMPLETED",
    "transaction_date": "2025-11-23T11:07:13.236061",
    "payment_method": "paypal",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000016",
    "amount": 2195.61,
    "currency": "AUD",
    "description": "Mobile Payment",
    "status": "COMPLETED",
    "transaction_date": "2025-11-19T11:07:13.236061",
    "payment_method": "credit_card",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000017",
    "amount": 3451.32,
    "currency": "USD",
    "description": "Restaurant Payment",
    "status": "COMPLETED",
    "transaction_date": "2025-02-25T11:07:13.236061",
    "payment_method": "bank_transfer",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000018",
    "amount": 178.78,
    "currency": "GBP",
    "description": "Online Purchase",
    "status": "COMPLETED",
    "transaction_date": "2025-01-19T11:07:13.236061",
    "payment_method": "paypal",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000019",
    "amount": 3475.72,
    "currency": "AUD",
    "description": "Gas Station",
    "status": "PENDING",
    "transaction_date": "2025-06-22T11:07:13.236061",
    "payment_method": "credit_card",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000020",
    "amount": 4494.6,
    "currency": "AUD",
    "description": "ATM Withdrawal",
    "status": "PENDING",
    "transaction_date": "2025-03-12T11:07:13.236061",
    "payment_method": "bank_transfer",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000021",
    "amount": 2156.18,
    "currency": "GBP",
    "description": "ATM Withdrawal",
    "status": "COMPLETED",
    "transaction_date": "2025-05-21T11:07:13.236061",
    "payment_method": "cash",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000022",
    "amount": 373.97,
    "currency": "CAD",
    "description": "Grocery Shopping",
    "status": "PENDING",
    "transaction_date": "2025-10-25T11:07:13.236061",
    "payment_method": "debit_card",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000023",
    "amount": 219.37,
    "currency": "EUR",
    "description": "Restaurant Payment",
    "status": "PENDING",
    "transaction_date": "2025-02-07T11:07:13.236061",
    "payment_method": "credit_card",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000024",
    "amount": 2751.81,
    "currency": "CAD",
    "description": "Gas Station",
    "status": "COMPLETED",
    "transaction_date": "2025-06-12T11:07:13.236061",
    "payment_method": "cash",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000025",
    "amount": 2101.73,
    "currency": "AUD",
    "description": "Salary Deposit",
    "status": "COMPLETED",
    "transaction_date": "2025-07-05T11:07:13.236061",
    "payment_method": "credit_card",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000026",
    "amount": 2413.4,
    "currency": "EUR",
    "description": "Online Purchase",
    "status": "COMPLETED",
    "transaction_date": "2025-01-26T11:07:13.236061",
    "payment_method": "credit_card",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000027",
    "amount": 4079.84,
    "currency": "CAD",
    "description": "Mobile Payment",
    "status": "COMPLETED",
    "transaction_date": "2025-01-20T11:07:13.236061",
    "payment_method": "credit_card",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000028",
    "amount": 3956.65,
    "currency": "CAD",
    "description": "Grocery Shopping",
    "status": "COMPLETED",
    "transaction_date": "2025-04-10T11:07:13.236061",
    "payment_method": "cash",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000029",
    "amount": 630.87,
    "currency": "AUD",
    "description": "Insurance Premium",
    "status": "COMPLETED",
    "transaction_date": "2025-03-03T11:07:13.236061",
    "payment_method": "paypal",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000030",
    "amount": 4086.44,
    "currency": "CAD",
    "description": "Restaurant Payment",
    "status": "COMPLETED",
    "transaction_date": "2025-01-27T11:07:13.236061",
    "payment_method": "credit_card",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000031",
    "amount": 1960.41,
    "currency": "AUD",
    "description": "Rent Payment",
    "status": "COMPLETED",
    "transaction_date": "2025-04-07T11:07:13.236061",
    "payment_method": "debit_card",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000032",
    "amount": 2770.16,
    "currency": "CAD",
    "description": "Online Purchase",
    "status": "COMPLETED",
    "transaction_date": "2025-10-05T11:07:13.236061",
    "payment_method": "paypal",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000033",
    "amount": 3609.19,
    "currency": "AUD",
    "description": "Salary Deposit",
    "status": "COMPLETED",
    "transaction_date": "2025-06-28T11:07:13.236061",
    "payment_method": "paypal",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000034",
    "amount": 317.96,
    "currency": "AUD",
    "description": "Insurance Premium",
    "status": "COMPLETED",
    "transaction_date": "2025-05-01T11:07:13.236061",
    "payment_method": "paypal",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000035",
    "amount": 3086.96,
    "currency": "GBP",
    "description": "Rent Payment",
    "status": "COMPLETED",
    "transaction_date": "2025-02-14T11:07:13.236061",
    "payment_method": "paypal",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000036",
    "amount": 1656.98,
    "currency": "AUD",
    "description": "Mobile Payment",
    "status": "COMPLETED",
    "transaction_date": "2025-11-29T11:07:13.236061",
    "payment_method": "cash",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000037",
    "amount": 682.98,
    "currency": "AUD",
    "description": "Gas Station",
    "status": "COMPLETED",
    "transaction_date": "2025-08-19T11:07:13.236061",
    "payment_method": "credit_card",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000038",
    "amount": 775.54,
    "currency": "USD",
    "description": "Mobile Payment",
    "status": "COMPLETED",
    "transaction_date": "2025-11-27T11:07:13.236061",
    "payment_method": "cash",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000039",
    "amount": 1384.84,
    "currency": "AUD",
    "description": "ATM Withdrawal",
    "status": "COMPLETED",
    "transaction_date": "2025-12-03T11:07:13.236061",
    "payment_method": "paypal",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000040",
    "amount": 3830.91,
    "currency": "AUD",
    "description": "ATM Withdrawal",
    "status": "COMPLETED",
    "transaction_date": "2025-07-05T11:07:13.236061",
    "payment_method": "paypal",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000041",
    "amount": 2946.27,
    "currency": "EUR",
    "description": "ATM Withdrawal",
    "status": "CANCELLED",
    "transaction_date": "2025-04-28T11:07:13.236061",
    "payment_method": "credit_card",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000042",
    "amount": 4018.8,
    "currency": "CAD",
    "description": "Rent Payment",
    "status": "COMPLETED",
    "transaction_date": "2025-04-10T11:07:13.236061",
    "payment_method": "bank_transfer",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000043",
    "amount": 2693.72,
    "currency": "AUD",
    "description": "Insurance Premium",
    "status": "COMPLETED",
    "transaction_date": "2025-05-02T11:07:13.236061",
    "payment_method": "paypal",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000044",
    "amount": 3224.09,
    "currency": "CAD",
    "description": "Online Purchase",
    "status": "COMPLETED",
    "transaction_date": "2025-11-19T11:07:13.236061",
    "payment_method": "credit_card",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000045",
    "amount": 4310.34,
    "currency": "GBP",
    "description": "Online Purchase",
    "status": "COMPLETED",
    "transaction_date": "2025-06-22T11:07:13.236061",
    "payment_method": "bank_transfer",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000046",
    "amount": 3885.5,
    "currency": "CAD",
    "description": "Restaurant Payment",
    "status": "CANCELLED",
    "transaction_date": "2025-02-24T11:07:13.236061",
    "payment_method": "credit_card",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000047",
    "amount": 4683.39,
    "currency": "USD",
    "description": "Mobile Payment",
    "status": "COMPLETED",
    "transaction_date": "2025-10-23T11:07:13.236061",
    "payment_method": "bank_transfer",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000048",
    "amount": 321.46,
    "currency": "USD",
    "description": "Online Purchase",
    "status": "COMPLETED",
    "transaction_date": "2025-10-25T11:07:13.236061",
    "payment_method": "cash",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000049",
    "amount": 3881.09,
    "currency": "CAD",
    "description": "Online Transfer",
    "status": "COMPLETED",
    "transaction_date": "2025-03-08T11:07:13.236061",
    "payment_method": "credit_card",
    "source": "MOCK_DATA"
  },
  {
    "id": "mock_txn_000050",
    "amount": 2323.34,
    "currency": "CAD",
    "description": "ATM Withdrawal",
    "status": "COMPLETED",
    "transaction_date": "2025-06-11T11:07:13.236061",
    "payment_method": "credit_card",
    "source": "MOCK_DATA"
  }
]

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

    print("\n" + "=" * 60)
    print(f"✅ Import completed! Successfully imported {success_count}/{len(transactions)} transactions")
    print("🌐 API available at: http://localhost:8080")

if __name__ == "__main__":
    main()
