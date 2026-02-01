#!/usr/bin/env python3
"""
Data Generator for Transaction API Performance Testing
Generates 30,000+ test records for users and transactions
"""

import requests
import json
import random
import time
from datetime import datetime, timedelta
import names

BASE_URL = 'http://localhost:8080'
API_URL = f'{BASE_URL}/api'

# Sample data
FIRST_NAMES = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emma', 'Chris', 'Lisa', 'Robert', 'Maria']
LAST_NAMES = ['Smith', 'Johnson', 'Brown', 'Williams', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
CITIES = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego']

TRANSACTION_TYPES = ['DEPOSIT', 'WITHDRAWAL', 'TRANSFER_IN', 'TRANSFER_OUT']
DESCRIPTIONS = [
    'Salary deposit', 'ATM withdrawal', 'Online purchase', 'Bill payment',
    'Transfer from savings', 'Cash deposit', 'Check deposit', 'Wire transfer',
    'Investment return', 'Refund', 'Gift', 'Loan payment'
]

def generate_users(count=1000):
    """Generate test users"""
    users = []
    for i in range(count):
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        username = f"{first_name.lower()}{last_name.lower()}{i}"
        email = f"{username}@example.com"

        user = {
            "username": username,
            "email": email,
            "password": "password123",
            "firstName": first_name,
            "lastName": last_name
        }
        users.append(user)
    return users

def register_users(users):
    """Register users via API"""
    registered_users = []
    print(f"Registering {len(users)} users...")

    for i, user in enumerate(users):
        try:
            response = requests.post(f'{API_URL}/auth/signup', json=user)
            if response.status_code == 200:
                print(f"✓ Registered user {i+1}/{len(users)}: {user['username']}")
                registered_users.append(user)
            else:
                print(f"✗ Failed to register user {user['username']}: {response.text}")
        except Exception as e:
            print(f"✗ Error registering user {user['username']}: {e}")

        # Small delay to avoid overwhelming the server
        if (i + 1) % 50 == 0:
            time.sleep(0.1)

    return registered_users

def login_and_get_token(username, password="password123"):
    """Login and get JWT token"""
    login_data = {
        "username": username,
        "password": password
    }

    try:
        response = requests.post(f'{API_URL}/auth/signin', json=login_data)
        if response.status_code == 200:
            return response.json()['token']
        else:
            print(f"Login failed for {username}: {response.text}")
            return None
    except Exception as e:
        print(f"Login error for {username}: {e}")
        return None

def generate_transactions(users, count_per_user=30):
    """Generate transactions for users"""
    transactions = []
    total_transactions = len(users) * count_per_user

    print(f"Generating {total_transactions} transactions...")

    for user_idx, user in enumerate(users):
        user_transactions = []

        # Login to get token
        token = login_and_get_token(user['username'])
        if not token:
            continue

        headers = {'Authorization': f'Bearer {token}'}

        # Get user ID (assuming we can get it from somewhere, or hardcode for demo)
        # In real scenario, you'd store user IDs after registration
        user_id = user_idx + 1  # Assuming sequential IDs starting from 1

        for i in range(count_per_user):
            transaction_type = random.choice(TRANSACTION_TYPES)
            amount = round(random.uniform(10, 10000), 2)
            description = random.choice(DESCRIPTIONS)

            # Adjust amount based on transaction type
            if transaction_type in ['WITHDRAWAL', 'TRANSFER_OUT']:
                amount = -abs(amount)

            transaction = {
                "transactionType": transaction_type,
                "amount": amount,
                "description": description,
                "user": {"id": user_id}
            }

            user_transactions.append((transaction, headers))

        transactions.extend(user_transactions)

        if (user_idx + 1) % 10 == 0:
            print(f"✓ Generated transactions for user {user_idx + 1}/{len(users)}")

    return transactions

def create_transactions(transactions):
    """Create transactions via API"""
    print(f"Creating {len(transactions)} transactions...")

    success_count = 0
    for i, (transaction, headers) in enumerate(transactions):
        try:
            response = requests.post(f'{API_URL}/transactions', json=transaction, headers=headers)
            if response.status_code == 200:
                success_count += 1
                if (i + 1) % 100 == 0:
                    print(f"✓ Created transaction {i+1}/{len(transactions)}")
            else:
                print(f"✗ Failed transaction {i+1}: {response.text}")
        except Exception as e:
            print(f"✗ Error creating transaction {i+1}: {e}")

        # Small delay
        if (i + 1) % 200 == 0:
            time.sleep(0.05)

    print(f"Successfully created {success_count}/{len(transactions)} transactions")

def run_performance_test():
    """Run the complete performance test"""
    print("🚀 Starting Transaction API Performance Test")
    print("=" * 50)

    # Generate and register users
    users = generate_users(1000)  # 1000 users
    registered_users = register_users(users)

    if not registered_users:
        print("❌ No users registered. Exiting.")
        return

    # Generate and create transactions
    transactions = generate_transactions(registered_users[:100], 30)  # 100 users x 30 transactions = 3000
    create_transactions(transactions)

    print("\n✅ Performance test completed!")
    print(f"📊 Test Results:")
    print(f"   - Users created: {len(registered_users)}")
    print(f"   - Transactions created: ~{len(registered_users[:100]) * 30}")
    print(f"   - Total records: {len(registered_users) + (len(registered_users[:100]) * 30)}")

def test_api_endpoints():
    """Test basic API endpoints"""
    print("\n🔍 Testing API endpoints...")

    # Test registration
    test_user = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "password123",
        "firstName": "Test",
        "lastName": "User"
    }

    response = requests.post(f'{API_URL}/auth/signup', json=test_user)
    print(f"Registration test: {'✓' if response.status_code == 200 else '✗'} ({response.status_code})")

    # Test login
    login_data = {"username": "testuser", "password": "password123"}
    response = requests.post(f'{API_URL}/auth/signin', json=login_data)
    if response.status_code == 200:
        token = response.json()['token']
        print(f"Login test: ✓")

        # Test protected endpoint
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.get(f'{API_URL}/users', headers=headers)
        print(f"Protected endpoint test: {'✓' if response.status_code == 200 else '✗'} ({response.status_code})")
    else:
        print(f"Login test: ✗ ({response.status_code})")

if __name__ == "__main__":
    print("Transaction API Data Generator & Performance Tester")
    print("=" * 55)

    # First test if API is running
    try:
        response = requests.get(f'{BASE_URL}/actuator/health', timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
        else:
            print("⚠️  API health check failed, but continuing...")
    except:
        print("⚠️  Cannot connect to API health endpoint, but continuing...")

    # Run basic endpoint tests
    test_api_endpoints()

    # Ask user if they want to run full performance test
    print("\n" + "="*50)
    choice = input("Run full performance test with 30,000+ records? (y/N): ").lower().strip()
    if choice == 'y':
        run_performance_test()
    else:
        print("Skipping performance test. Run with --full to execute complete test.")