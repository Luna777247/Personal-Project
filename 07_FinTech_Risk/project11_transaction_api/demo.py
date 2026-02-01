#!/usr/bin/env python3
"""
Transaction API Demo Script
Demonstrates the main features of the Transaction Management API
"""

import requests
import json
import time

BASE_URL = 'http://localhost:8080'
API_URL = f'{BASE_URL}/api'

def print_separator(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def demo_authentication():
    """Demo user registration and login"""
    print_separator("1. USER AUTHENTICATION")

    # Register a new user
    user_data = {
        "username": "demo_user",
        "email": "demo@example.com",
        "password": "password123",
        "firstName": "Demo",
        "lastName": "User"
    }

    print("📝 Registering new user...")
    response = requests.post(f'{API_URL}/auth/signup', json=user_data)
    print(f"Registration response: {response.status_code}")
    if response.status_code == 200:
        print("✅ User registered successfully!")
    else:
        print(f"❌ Registration failed: {response.text}")

    # Login
    print("\n🔐 Logging in...")
    login_data = {
        "username": "demo_user",
        "password": "password123"
    }

    response = requests.post(f'{API_URL}/auth/signin', json=login_data)
    print(f"Login response: {response.status_code}")

    if response.status_code == 200:
        auth_data = response.json()
        token = auth_data['token']
        user_id = auth_data['id']
        print("✅ Login successful!")
        print(f"   Token: {token[:50]}...")
        print(f"   User ID: {user_id}")
        return token, user_id
    else:
        print(f"❌ Login failed: {response.text}")
        return None, None

def demo_user_management(token):
    """Demo user CRUD operations"""
    print_separator("2. USER MANAGEMENT")

    headers = {'Authorization': f'Bearer {token}'}

    # Get all users
    print("👥 Getting all users...")
    response = requests.get(f'{API_URL}/users', headers=headers)
    print(f"Get users response: {response.status_code}")
    if response.status_code == 200:
        users = response.json()
        print(f"✅ Found {len(users)} users")
        for user in users[:3]:  # Show first 3 users
            print(f"   - {user['username']} ({user['email']})")

    # Create another user
    new_user = {
        "username": "test_user",
        "email": "test@example.com",
        "password": "password123",
        "firstName": "Test",
        "lastName": "User"
    }

    print("\n➕ Creating another user...")
    response = requests.post(f'{API_URL}/users', json=new_user, headers=headers)
    print(f"Create user response: {response.status_code}")
    if response.status_code == 200:
        created_user = response.json()
        print("✅ User created successfully!")
        print(f"   Username: {created_user['username']}")
        print(f"   Balance: ${created_user['accountBalance']}")
        return created_user['id']
    else:
        print(f"❌ User creation failed: {response.text}")
        return None

def demo_transactions(token, user_id, recipient_id=None):
    """Demo transaction operations"""
    print_separator("3. TRANSACTION MANAGEMENT")

    headers = {'Authorization': f'Bearer {token}'}

    # Create a deposit transaction
    deposit = {
        "transactionType": "DEPOSIT",
        "amount": 1000.00,
        "description": "Initial deposit",
        "user": {"id": user_id}
    }

    print("💰 Creating deposit transaction...")
    response = requests.post(f'{API_URL}/transactions', json=deposit, headers=headers)
    print(f"Deposit response: {response.status_code}")
    if response.status_code == 200:
        transaction = response.json()
        print("✅ Deposit successful!")
        print(f"   Amount: ${transaction['amount']}")
        print(f"   Type: {transaction['transactionType']}")
        print(f"   Status: {transaction['status']}")

    # Create a withdrawal
    withdrawal = {
        "transactionType": "WITHDRAWAL",
        "amount": 200.00,
        "description": "ATM withdrawal",
        "user": {"id": user_id}
    }

    print("\n💸 Creating withdrawal transaction...")
    response = requests.post(f'{API_URL}/transactions', json=withdrawal, headers=headers)
    print(f"Withdrawal response: {response.status_code}")
    if response.status_code == 200:
        transaction = response.json()
        print("✅ Withdrawal successful!")
        print(f"   Amount: ${transaction['amount']}")
        print(f"   Balance after: ${transaction['user']['accountBalance']}")

    # Get user's transactions
    print(f"\n📊 Getting transactions for user {user_id}...")
    response = requests.get(f'{API_URL}/transactions/user/{user_id}', headers=headers)
    print(f"Get transactions response: {response.status_code}")
    if response.status_code == 200:
        transactions = response.json()
        print(f"✅ Found {len(transactions)} transactions")
        for tx in transactions[-3:]:  # Show last 3 transactions
            print(f"   - {tx['transactionType']}: ${tx['amount']} - {tx['description']}")

    # Check balance
    print(f"\n💵 Checking account balance for user {user_id}...")
    response = requests.get(f'{API_URL}/users/{user_id}/balance', headers=headers)
    print(f"Balance response: {response.status_code}")
    if response.status_code == 200:
        balance = response.json()
        print(f"✅ Current balance: ${balance}")

    # Transfer money if we have a recipient
    if recipient_id:
        transfer_data = {
            "senderId": user_id,
            "recipientId": recipient_id,
            "amount": 100.00,
            "description": "Demo transfer"
        }

        print(f"\n🔄 Transferring $100 from user {user_id} to user {recipient_id}...")
        response = requests.post(f'{API_URL}/transactions/transfer', json=transfer_data, headers=headers)
        print(f"Transfer response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ Transfer successful!")
            print(f"   Transaction ID: {result['transactionId']}")
        else:
            print(f"❌ Transfer failed: {response.text}")

def demo_error_handling(token):
    """Demo error handling"""
    print_separator("4. ERROR HANDLING")

    headers = {'Authorization': f'Bearer {token}'}

    # Try to access without token
    print("🚫 Trying to access protected endpoint without token...")
    response = requests.get(f'{API_URL}/users')
    print(f"No token response: {response.status_code} - {response.reason}")

    # Try invalid login
    print("\n🔐 Trying invalid login...")
    invalid_login = {"username": "nonexistent", "password": "wrong"}
    response = requests.post(f'{API_URL}/auth/signin', json=invalid_login)
    print(f"Invalid login response: {response.status_code}")

    # Try to create user with existing username
    print("\n👤 Trying to create user with existing username...")
    duplicate_user = {
        "username": "demo_user",  # Already exists
        "email": "duplicate@example.com",
        "password": "password123",
        "firstName": "Duplicate",
        "lastName": "User"
    }
    response = requests.post(f'{API_URL}/users', json=duplicate_user, headers=headers)
    print(f"Duplicate user response: {response.status_code}")

def run_demo():
    """Run the complete demo"""
    print("🚀 TRANSACTION MANAGEMENT API DEMO")
    print("=" * 60)
    print("This demo will showcase the main features of the Transaction API")
    print("Make sure the API is running on http://localhost:8080")
    print("=" * 60)

    # Check if API is running
    try:
        response = requests.get(BASE_URL, timeout=5)
        print("✅ API server is running")
    except:
        print("❌ Cannot connect to API server. Please start the application first.")
        print("   Run: mvn spring-boot:run")
        return

    # Demo authentication
    token, user_id = demo_authentication()
    if not token or not user_id:
        print("❌ Authentication failed. Cannot continue demo.")
        return

    # Demo user management
    recipient_id = demo_user_management(token)

    # Demo transactions
    demo_transactions(token, user_id, recipient_id)

    # Demo error handling
    demo_error_handling(token)

    print_separator("DEMO COMPLETED")
    print("🎉 All major API features have been demonstrated!")
    print("\n📚 Next steps:")
    print("   - Check the README.md for API documentation")
    print("   - Run the performance test: python scripts/generate_test_data.py")
    print("   - Explore the code structure and extend functionality")
    print("\n🔗 API Documentation: http://localhost:8080/swagger-ui.html (if enabled)")

if __name__ == "__main__":
    run_demo()