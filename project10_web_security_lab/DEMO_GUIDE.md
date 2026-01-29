# Web Security Lab - Demo Guide

## 🚀 Quick Start Demo

### Step 1: Start the Vulnerable Web Application
```bash
cd app
python app.py
```
App will run on: http://localhost:5000

### Step 2: Manual Vulnerability Testing

#### 1. SQL Injection Demo
1. Open browser to http://localhost:5000/login
2. Enter these credentials:
   - Username: `' OR '1'='1' --`
   - Password: `anything`
3. Click Login - you'll be logged in as admin without valid password!

#### 2. XSS Demo
1. Login normally with: user1 / pass1
2. Go to dashboard and create a post with:
   ```
   <script>alert("XSS Attack!")</script><h1 style="color:red">HACKED!</h1>
   ```
3. The script will execute and show alert + red "HACKED!" text

#### 3. IDOR Demo
1. Login as user1
2. In browser URL, change: `/dashboard?user_id=1`
3. You'll see admin's posts instead of your own!

#### 4. CSRF Demo
1. Login as user1
2. Open browser developer tools (F12)
3. In console, run:
```javascript
fetch('/transfer', {
  method: 'POST',
  headers: {'Content-Type': 'application/x-www-form-urlencoded'},
  body: 'amount=1000&to_user=999'
}).then(r => r.text()).then(console.log)
```
4. Check response - money transfer succeeds without user consent!

### Step 3: Automated Testing
```bash
cd pentest
python pentest_script.py
```

### Step 4: Review Reports
- `docs/pentest_report.md` - Penetration testing results
- `code_review/security_review.md` - Code analysis findings

### Step 5: Cloud Security Lab (Optional)
```bash
cd cloud_lab
# Set your Azure subscription
$env:AZURE_SUBSCRIPTION_ID = "your-subscription-id"
python setup_azure_lab.py
```

## 🎯 Expected Demo Results

### SQL Injection
```
✅ SQL Injection vulnerability found - login bypass successful
```

### XSS
```
✅ XSS vulnerability found - payload reflected in dashboard
```

### IDOR
```
✅ IDOR vulnerability found - can access other users' data
```

### CSRF
```
✅ CSRF vulnerability found - transfer successful without token
```

## 🔧 Remediation Examples

### Fix SQL Injection
```python
# BEFORE (vulnerable)
query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"

# AFTER (secure)
cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
```

### Fix XSS
```html
<!-- BEFORE (vulnerable) -->
<p>{{ post.content }}</p>

<!-- AFTER (secure) -->
<p>{{ post.content | e }}</p>
```

### Fix IDOR
```python
# BEFORE (vulnerable)
user_id = request.args.get('user_id', session['user_id'])

# AFTER (secure)
user_id = session['user_id']  # Only access own data
```

### Fix CSRF
```python
# Add Flask-WTF
from flask_wtf.csrf import CSRFProtect
csrf = CSRFProtect(app)

# Add to templates
<form method="POST">
    {{ csrf_token() }}
    <!-- form fields -->
</form>
```

## 📊 Security Assessment Summary

| Vulnerability | Severity | Status | Impact |
|---------------|----------|--------|---------|
| SQL Injection | Critical | ✅ Exploitable | Authentication bypass |
| XSS | High | ✅ Exploitable | Script injection |
| IDOR | High | ✅ Exploitable | Data exposure |
| CSRF | Medium | ✅ Exploitable | Unauthorized actions |
| Hard-coded Secrets | Medium | ✅ Identified | Session compromise |

## 🎓 Learning Outcomes

This lab demonstrates:
- **OWASP Top 10** vulnerability exploitation
- **Secure coding practices** implementation
- **Penetration testing methodology**
- **Cloud security misconfigurations**
- **Professional security reporting**

## ⚠️ Ethical Notice

This is an educational tool for learning security concepts. Never use these techniques on systems you don't own or have explicit permission to test.

## 🛡️ Next Steps

1. Implement the remediation fixes shown above
2. Run the tests again to verify fixes work
3. Study the detailed reports for deeper understanding
4. Explore the cloud security lab for infrastructure security learning