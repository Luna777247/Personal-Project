"""
Streamlit Dashboard for Fraud Detection Monitoring
Real-time monitoring and alerting dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time
from streamlit_autorefresh import st_autorefresh

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")

# Auto-refresh configuration
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)
st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []


def check_api_health():
    """Check API health status"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None


def predict_transaction(transaction_data):
    """Send transaction for prediction"""
    try:
        response = requests.post(f"{API_URL}/predict", json=transaction_data, timeout=10)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except Exception as e:
        return False, str(e)


def generate_sample_transaction():
    """Generate sample transaction for testing"""
    return {
        "transaction_id": f"TXN_{int(time.time())}",
        "customer_id": np.random.randint(1000, 9999),
        "merchant_id": np.random.randint(100, 999),
        "amount": float(np.random.lognormal(4, 1.5)),
        "timestamp": datetime.now().isoformat(),
        "merchant_category": np.random.choice(['retail', 'food', 'gas', 'entertainment', 'travel']),
        "card_type": np.random.choice(['credit', 'debit']),
        "transaction_type": np.random.choice(['online', 'in-store', 'atm']),
        "country_code": np.random.choice(['US', 'GB', 'DE', 'FR'])
    }


# Title and header
st.title("🔒 Fraud Detection Dashboard")
st.markdown("Real-time monitoring and alerting system for fraudulent transactions")

# Sidebar - API Status
st.sidebar.header("🔌 API Status")
api_healthy, health_data = check_api_health()

if api_healthy:
    st.sidebar.success("✅ API Online")
    if health_data:
        st.sidebar.metric("Model Version", health_data.get('model_version', 'N/A'))
        st.sidebar.caption(f"Last updated: {health_data.get('timestamp', 'N/A')}")
else:
    st.sidebar.error("❌ API Offline")
    st.sidebar.warning("Please ensure the API is running")

# Sidebar - Controls
st.sidebar.header("⚙️ Controls")

# Test transaction generator
if st.sidebar.button("🎲 Generate Test Transaction"):
    if api_healthy:
        test_transaction = generate_sample_transaction()
        success, result = predict_transaction(test_transaction)
        
        if success:
            st.session_state.predictions.append(result)
            
            # Add to alerts if fraud detected
            if result['is_fraud'] == 1:
                alert = {
                    'timestamp': result['timestamp'],
                    'transaction_id': result['transaction_id'],
                    'risk_level': result['risk_level'],
                    'fraud_probability': result['fraud_probability']
                }
                st.session_state.alerts.append(alert)
                
            st.sidebar.success(f"Transaction {result['transaction_id'][:15]}... processed")
        else:
            st.sidebar.error("Failed to process transaction")
    else:
        st.sidebar.error("API is offline")

# Clear data button
if st.sidebar.button("🗑️ Clear Dashboard Data"):
    st.session_state.predictions = []
    st.session_state.alerts = []
    st.sidebar.success("Data cleared")

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

# Calculate metrics
if st.session_state.predictions:
    df_predictions = pd.DataFrame(st.session_state.predictions)
    total_transactions = len(df_predictions)
    fraud_count = df_predictions['is_fraud'].sum()
    fraud_rate = (fraud_count / total_transactions) * 100
    avg_fraud_prob = df_predictions['fraud_probability'].mean() * 100
    
    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Fraud Detected", f"{fraud_count:,}", delta=f"{fraud_rate:.1f}%")
    col3.metric("Avg Fraud Probability", f"{avg_fraud_prob:.1f}%")
    col4.metric("Active Alerts", len(st.session_state.alerts))
else:
    col1.metric("Total Transactions", "0")
    col2.metric("Fraud Detected", "0")
    col3.metric("Avg Fraud Probability", "0%")
    col4.metric("Active Alerts", "0")

st.divider()

# Charts section
if st.session_state.predictions:
    df_predictions = pd.DataFrame(st.session_state.predictions)
    df_predictions['timestamp'] = pd.to_datetime(df_predictions['timestamp'])
    
    # Row 1: Time series and distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Fraud Probability Over Time")
        fig_timeseries = px.line(
            df_predictions.sort_values('timestamp'),
            x='timestamp',
            y='fraud_probability',
            color='risk_level',
            color_discrete_map={'low': 'green', 'medium': 'orange', 'high': 'red'},
            markers=True
        )
        fig_timeseries.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                                annotation_text="Threshold")
        fig_timeseries.update_layout(height=300)
        st.plotly_chart(fig_timeseries, use_container_width=True)
    
    with col2:
        st.subheader("📊 Risk Level Distribution")
        risk_counts = df_predictions['risk_level'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={'low': 'green', 'medium': 'orange', 'high': 'red'}
        )
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Row 2: Histogram and gauge
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 Fraud Probability Distribution")
        fig_hist = px.histogram(
            df_predictions,
            x='fraud_probability',
            nbins=30,
            color='is_fraud',
            color_discrete_map={0: 'blue', 1: 'red'},
            labels={'is_fraud': 'Fraud Status', 'fraud_probability': 'Fraud Probability'}
        )
        fig_hist.update_layout(height=300)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Current Fraud Rate")
        fraud_rate_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fraud_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fraud Rate (%)"},
            delta={'reference': 5.0},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 2], 'color': "lightgreen"},
                    {'range': [2, 5], 'color': "yellow"},
                    {'range': [5, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 10
                }
            }
        ))
        fraud_rate_gauge.update_layout(height=300)
        st.plotly_chart(fraud_rate_gauge, use_container_width=True)

else:
    st.info("📊 No transaction data available. Generate test transactions to see visualizations.")

st.divider()

# Alerts section
st.subheader("🚨 Recent Fraud Alerts")

if st.session_state.alerts:
    # Keep only last 50 alerts
    st.session_state.alerts = st.session_state.alerts[-50:]
    
    df_alerts = pd.DataFrame(st.session_state.alerts)
    df_alerts['timestamp'] = pd.to_datetime(df_alerts['timestamp'])
    df_alerts = df_alerts.sort_values('timestamp', ascending=False)
    
    # Color-code by risk level
    def highlight_risk(row):
        if row['risk_level'] == 'high':
            return ['background-color: #ffcccc'] * len(row)
        elif row['risk_level'] == 'medium':
            return ['background-color: #fff4cc'] * len(row)
        return [''] * len(row)
    
    styled_df = df_alerts.style.apply(highlight_risk, axis=1).format({
        'fraud_probability': '{:.2%}',
        'timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M:%S')
    })
    
    st.dataframe(styled_df, use_container_width=True, height=300)
else:
    st.info("No fraud alerts detected")

st.divider()

# Recent transactions table
st.subheader("📋 Recent Transactions")

if st.session_state.predictions:
    df_recent = pd.DataFrame(st.session_state.predictions).tail(20).iloc[::-1]
    df_recent['timestamp'] = pd.to_datetime(df_recent['timestamp'])
    
    # Format display
    display_df = df_recent[['transaction_id', 'timestamp', 'is_fraud', 
                            'fraud_probability', 'risk_level', 'model_version']].copy()
    
    def highlight_fraud(row):
        if row['is_fraud'] == 1:
            return ['background-color: #ffcccc'] * len(row)
        return ['background-color: #ccffcc'] * len(row)
    
    styled_recent = display_df.style.apply(highlight_fraud, axis=1).format({
        'fraud_probability': '{:.2%}',
        'timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M:%S')
    })
    
    st.dataframe(styled_recent, use_container_width=True, height=400)
else:
    st.info("No recent transactions")

# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh every {refresh_interval}s")
