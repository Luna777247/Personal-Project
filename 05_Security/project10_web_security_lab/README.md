# Web Penetration Testing & Security Lab Project

## Overview

This comprehensive security project demonstrates web application penetration testing, secure code review, and cloud security practices. The project includes a vulnerable web application, automated penetration testing tools, security code analysis, and Azure cloud security lab.

## Project Structure

```
project10_web_security_lab/
├── app/                          # Vulnerable web application
│   ├── app.py                   # Flask backend with vulnerabilities
│   └── templates/               # HTML templates with XSS
├── pentest/                     # Penetration testing tools
│   └── pentest_script.py        # Automated security testing
├── code_review/                 # Security code analysis
│   └── security_review.md       # Detailed vulnerability report
├── cloud_lab/                   # Azure cloud security
│   ├── setup_azure_lab.py      # Lab environment creation
│   ├── cleanup_azure_lab.py    # Resource cleanup
│   └── azure_security_lab.md   # Cloud security guide
├── docs/                        # Documentation and reports
│   └── pentest_report.md        # Penetration testing results
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Components

### 1. Vulnerable Web Application

A Flask-based web application intentionally designed with OWASP Top 10 vulnerabilities:

- **SQL Injection**: Authentication bypass in login
- **XSS**: Post content rendering without sanitization
- **IDOR**: Direct object reference vulnerabilities
- **CSRF**: Missing token validation in forms
- **Hard-coded Secrets**: Application secrets in source code

### 2. Penetration Testing Suite

Automated testing scripts that demonstrate:

- **XSS Exploitation**: JavaScript injection attacks
- **SQL Injection**: Database query manipulation
- **IDOR Attacks**: Unauthorized data access
- **CSRF Attacks**: Cross-site request forgery

### 3. Secure Code Review

Comprehensive analysis of:

- **Backend Security**: Python/Flask vulnerabilities
- **Frontend Security**: HTML/JavaScript issues
- **Database Security**: SQL injection and data exposure
- **Configuration Security**: Hard-coded secrets and misconfigurations

### 4. Azure Cloud Security Lab

Hands-on cloud security demonstration:

- **Resource Deployment**: Automated Azure environment setup
- **Misconfiguration Detection**: Security assessment tools
- **Remediation Scripts**: Automated security fixes
- **Best Practices**: Identity, access control, and monitoring

## Installation & Setup

### Prerequisites

- Python 3.8+
- Azure CLI (for cloud lab)
- Azure subscription (for cloud lab)

### Installation

1. **Clone/download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **For Azure lab, configure Azure CLI**:
   ```bash
   az login
   az account set --subscription "your-subscription-id"
   export AZURE_SUBSCRIPTION_ID="your-subscription-id"
   ```

## Usage

### Web Application Testing

1. **Start the vulnerable application**:
   ```bash
   cd app
   python app.py
   ```
   Application runs on http://localhost:5000

2. **Run penetration tests**:
   ```bash
   cd pentest
   python pentest_script.py
   ```

3. **Review security findings**:
   - Check `docs/pentest_report.md` for detailed analysis
   - Review `code_review/security_review.md` for code issues

### Cloud Security Lab

1. **Setup Azure lab environment**:
   ```bash
   cd cloud_lab
   python setup_azure_lab.py
   ```

2. **Run security assessment**:
   ```bash
   python setup_azure_lab.py assess
   ```

3. **Cleanup resources**:
   ```bash
   python cleanup_azure_lab.py
   ```

## Security Vulnerabilities Demonstrated

### OWASP Top 10 Coverage

| Vulnerability | Location | Severity | Status |
|---------------|----------|----------|--------|
| A03:2021-Injection (SQLi) | Login authentication | Critical | Exploitable |
| A03:2021-Injection (XSS) | Post display | High | Exploitable |
| A01:2021-Broken Access Control (IDOR) | User data access | High | Exploitable |
| A01:2021-Broken Access Control (CSRF) | Form submissions | Medium | Exploitable |
| A05:2021-Security Misconfiguration | App configuration | Medium | Identified |

### Cloud Security Issues

- Public storage account access
- Open management ports (RDP/SSH)
- Missing encryption
- Insufficient monitoring
- Over-privileged access

## Learning Objectives

### Penetration Testing Skills
- Understanding OWASP Top 10 vulnerabilities
- Developing exploitation techniques
- Creating automated testing tools
- Writing security assessment reports

### Secure Coding Practices
- Input validation and sanitization
- Proper authentication and authorization
- Secure session management
- Safe database operations

### Cloud Security Knowledge
- Azure security best practices
- Misconfiguration detection
- Identity and access management
- Security monitoring and alerting

## Remediation Examples

### SQL Injection Fix
```python
# Vulnerable
query = f"SELECT * FROM users WHERE username='{username}'"

# Secure
cursor.execute("SELECT * FROM users WHERE username=?", (username,))
```

### XSS Prevention
```html
<!-- Vulnerable -->
<p>{{ user_input }}</p>

<!-- Secure -->
<p>{{ user_input | e }}</p>
```

### CSRF Protection
```python
from flask_wtf.csrf import CSRFProtect
csrf = CSRFProtect(app)
```

## Security Best Practices Implemented

### Web Application Security
- Parameterized queries
- Input validation
- CSRF tokens
- Content Security Policy
- Secure headers

### Cloud Security
- Least privilege access
- Network security groups
- Encryption at rest
- Security monitoring
- Regular audits

## Testing & Validation

### Automated Testing
- Vulnerability exploitation scripts
- Security assessment tools
- Cloud configuration checks

### Manual Testing
- Code review walkthrough
- Penetration testing methodology
- Remediation validation

## Educational Value

This project serves as a comprehensive learning resource for:

- **Security Students**: Understanding common vulnerabilities
- **Developers**: Learning secure coding practices
- **Security Professionals**: Penetration testing techniques
- **DevOps Engineers**: Cloud security implementation

## Ethical Considerations

⚠️ **Important**: This project is for educational purposes only. Do not use these techniques on systems you do not own or have explicit permission to test. Always follow responsible disclosure practices and obtain written authorization before conducting security assessments.

## Future Enhancements

- Additional vulnerability types (XXE, SSRF, etc.)
- Advanced exploitation techniques
- CI/CD security integration
- Automated remediation tools
- Multi-cloud security labs

## Contributing

This is an educational project. Contributions for additional vulnerabilities, better remediation examples, or improved testing methodologies are welcome.

## License

Educational use only. Not for production deployment or malicious activities.

## Contact

Security education project demonstrating responsible disclosure and ethical hacking practices.