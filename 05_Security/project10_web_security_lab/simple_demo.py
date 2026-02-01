#!/usr/bin/env python3
"""
Simple Vulnerability Demo - Shows code examples without running web app
"""

def show_vulnerabilities():
    """Show vulnerability examples with code snippets"""

    print("🔐 WEB SECURITY VULNERABILITIES DEMO")
    print("=" * 50)
    print("This demo shows the vulnerable code patterns found in the application\n")

    # SQL Injection Demo
    print("1️⃣ SQL INJECTION VULNERABILITY")
    print("-" * 40)
    print("❌ VULNERABLE CODE:")
    print("""
def login():
    username = request.form['username']
    password = request.form['password']

    # DANGEROUS: Direct string concatenation
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    cursor.execute(query)
    """)

    print("✅ EXPLOIT PAYLOAD:")
    print("Username: ' OR '1'='1' --")
    print("Password: anything")
    print("💡 This bypasses authentication by making the query always true\n")

    # XSS Demo
    print("2️⃣ CROSS-SITE SCRIPTING (XSS) VULNERABILITY")
    print("-" * 40)
    print("❌ VULNERABLE CODE:")
    print("""
@app.route('/dashboard')
def dashboard():
    posts = get_user_posts(session['user_id'])
    return render_template('dashboard.html', posts=posts)

<!-- dashboard.html -->
{% for post in posts %}
<div>{{ post.content }}</div>  <!-- NO ESCAPING! -->
{% endfor %}
    """)

    print("✅ EXPLOIT PAYLOAD:")
    print('<script>alert("XSS Attack!")</script><h1 style="color:red">HACKED!</h1>')
    print("💡 This executes JavaScript in victim's browser\n")

    # IDOR Demo
    print("3️⃣ INSECURE DIRECT OBJECT REFERENCE (IDOR)")
    print("-" * 40)
    print("❌ VULNERABLE CODE:")
    print("""
@app.route('/dashboard')
def dashboard():
    user_id = request.args.get('user_id', session['user_id'])
    posts = get_user_posts(user_id)  # TRUSTS USER INPUT!
    return render_template('dashboard.html', posts=posts)
    """)

    print("✅ EXPLOIT:")
    print("URL: /dashboard?user_id=999")
    print("💡 Access other users' data by changing the user_id parameter\n")

    # CSRF Demo
    print("4️⃣ CROSS-SITE REQUEST FORGERY (CSRF)")
    print("-" * 40)
    print("❌ VULNERABLE CODE:")
    print("""
@app.route('/transfer', methods=['POST'])
def transfer():
    amount = request.form['amount']
    to_user = request.form['to_user']
    # NO CSRF PROTECTION!
    transfer_money(session['user_id'], to_user, amount)
    return "Transfer successful!"
    """)

    print("✅ EXPLOIT:")
    print("JavaScript fetch attack:")
    print("""
fetch('/transfer', {
  method: 'POST',
  headers: {'Content-Type': 'application/x-www-form-urlencoded'},
  body: 'amount=1000&to_user=999'
})
    """)
    print("💡 Transfers money without user consent\n")

    # Hard-coded Secret
    print("5️⃣ HARD-CODED SECRET")
    print("-" * 40)
    print("❌ VULNERABLE CODE:")
    print("""
SECRET_KEY = 'super-secret-key-change-me-in-production'
app.secret_key = SECRET_KEY
    """)
    print("💡 Secret is visible in source code - easy to find and exploit\n")

    print("🎯 SUMMARY")
    print("-" * 40)
    print("✅ All vulnerabilities demonstrated through code analysis")
    print("✅ OWASP Top 10 patterns identified")
    print("✅ Remediation strategies provided in reports")
    print("\n📁 Check these files for full details:")
    print("- docs/pentest_report.md (penetration testing)")
    print("- code_review/security_review.md (code analysis)")
    print("- DEMO_GUIDE.md (manual testing steps)")

if __name__ == "__main__":
    show_vulnerabilities()