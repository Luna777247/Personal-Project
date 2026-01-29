#!/usr/bin/env python3
"""
Test script for the Web Security Lab
"""

import subprocess
import sys
import time
import requests
import os

def test_web_app():
    """Test the vulnerable web application"""
    print("Testing vulnerable web application...")

    # Start the web app in background
    try:
        app_process = subprocess.Popen(
            [sys.executable, 'app/app.py'],
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(3)  # Wait for app to start

        # Test basic connectivity
        response = requests.get('http://localhost:5000', timeout=5)
        if response.status_code == 200:
            print("✅ Web application is running")
        else:
            print("❌ Web application not responding")
            return False

        # Test login page
        response = requests.get('http://localhost:5000/login', timeout=5)
        if 'Login' in response.text:
            print("✅ Login page accessible")
        else:
            print("❌ Login page not working")
            return False

    except Exception as e:
        print(f"❌ Error testing web app: {e}")
        return False
    finally:
        # Clean up
        if 'app_process' in locals():
            app_process.terminate()
            app_process.wait()

    return True

def test_pentest_script():
    """Test the penetration testing script"""
    print("\nTesting penetration testing script...")

    # This would require the web app to be running
    # For demo purposes, just check if script exists and is syntactically correct
    try:
        with open('pentest/pentest_script.py', 'r') as f:
            content = f.read()

        # Basic syntax check
        compile(content, 'pentest_script.py', 'exec')
        print("✅ Penetration testing script syntax OK")

        return True
    except Exception as e:
        print(f"❌ Error in pentest script: {e}")
        return False

def test_azure_scripts():
    """Test Azure lab scripts"""
    print("\nTesting Azure lab scripts...")

    scripts = [
        'cloud_lab/setup_azure_lab.py',
        'cloud_lab/cleanup_azure_lab.py'
    ]

    for script in scripts:
        try:
            with open(script, 'r') as f:
                content = f.read()
            compile(content, script, 'exec')
            print(f"✅ {script} syntax OK")
        except Exception as e:
            print(f"❌ Error in {script}: {e}")
            return False

    return True

def main():
    """Run all tests"""
    print("🧪 WEB SECURITY LAB - TEST SUITE")
    print("=" * 50)

    results = []

    # Test web application
    results.append(test_web_app())

    # Test pentest script
    results.append(test_pentest_script())

    # Test Azure scripts
    results.append(test_azure_scripts())

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")

    passed = sum(results)
    total = len(results)

    print(f"Tests Passed: {passed}/{total}")

    if passed == total:
        print("✅ All tests passed! Security lab is ready.")
        print("\n🚀 Next steps:")
        print("1. Run: cd app && python app.py")
        print("2. Run: cd pentest && python pentest_script.py")
        print("3. Review reports in docs/ and code_review/")
        print("4. For cloud lab: cd cloud_lab && python setup_azure_lab.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main()