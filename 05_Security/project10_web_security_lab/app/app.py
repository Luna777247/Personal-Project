from flask import Flask, request, render_template, redirect, url_for, session, make_response
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'hardcoded_secret_key'  # Hard-coded secret - vulnerability

# Initialize database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, username TEXT, password TEXT, email TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS posts
                 (id INTEGER PRIMARY KEY, user_id INTEGER, content TEXT, timestamp TEXT)''')
    # Insert sample data
    c.execute("INSERT OR IGNORE INTO users VALUES (1, 'admin', 'password123', 'admin@example.com')")
    c.execute("INSERT OR IGNORE INTO users VALUES (2, 'user1', 'pass1', 'user1@example.com')")
    c.execute("INSERT OR IGNORE INTO users VALUES (3, 'user2', 'pass2', 'user2@example.com')")
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # SQL Injection vulnerability
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
        c.execute(query)
        user = c.fetchone()
        conn.close()

        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # IDOR vulnerability - no proper authorization check
    user_id = request.args.get('user_id', session['user_id'])

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM posts WHERE user_id=?", (user_id,))
    posts = c.fetchall()
    conn.close()

    return render_template('dashboard.html', posts=posts, user_id=user_id)

@app.route('/post', methods=['POST'])
def post():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    content = request.form['content']
    user_id = session['user_id']

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO posts (user_id, content, timestamp) VALUES (?, ?, ?)",
              (user_id, content, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

    return redirect(url_for('dashboard'))

@app.route('/transfer', methods=['GET', 'POST'])
def transfer():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        # CSRF vulnerability - no token validation
        amount = request.form['amount']
        to_user = request.form['to_user']

        # Simulate transfer (no actual money transfer logic)
        return f"Transferred ${amount} to user {to_user}"

    return render_template('transfer.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)