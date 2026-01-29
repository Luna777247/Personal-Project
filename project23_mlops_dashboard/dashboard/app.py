"""
Streamlit Dashboard for AI Ops
Main dashboard application
"""

import os
import sys
from pathlib import Path
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import mlflow
from mlflow.tracking import MlflowClient
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.triton_client import TritonGRPCClient


# Page config
st.set_page_config(
    page_title="AI Ops Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-live {
        color: #28a745;
        font-weight: bold;
    }
    .status-down {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Initialize connections
@st.cache_resource
def init_mlflow_client():
    """Initialize MLflow client"""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


@st.cache_resource
def init_triton_client():
    """Initialize Triton client"""
    try:
        triton_url = os.getenv("TRITON_GRPC_URL", "triton:8001")
        return TritonGRPCClient(url=triton_url, verbose=False)
    except Exception as e:
        st.error(f"Failed to connect to Triton: {e}")
        return None


def get_experiments(client: MlflowClient) -> pd.DataFrame:
    """Get all experiments from MLflow"""
    experiments = client.search_experiments()
    
    data = []
    for exp in experiments:
        data.append({
            "Name": exp.name,
            "ID": exp.experiment_id,
            "Artifact Location": exp.artifact_location,
            "Lifecycle Stage": exp.lifecycle_stage
        })
    
    return pd.DataFrame(data)


def get_recent_runs(client: MlflowClient, experiment_id: str, limit: int = 10) -> pd.DataFrame:
    """Get recent runs from an experiment"""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=limit
    )
    
    data = []
    for run in runs:
        metrics = {k: v for k, v in run.data.metrics.items()}
        params = {k: v for k, v in run.data.params.items()}
        
        data.append({
            "Run ID": run.info.run_id[:8],
            "Run Name": run.info.run_name or "N/A",
            "Status": run.info.status,
            "Start Time": datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M"),
            "Duration (s)": round((run.info.end_time - run.info.start_time) / 1000) if run.info.end_time else None,
            **{f"metric_{k}": v for k, v in metrics.items()},
            **{f"param_{k}": v for k, v in params.items()}
        })
    
    return pd.DataFrame(data)


def get_registered_models(client: MlflowClient) -> pd.DataFrame:
    """Get all registered models"""
    models = client.search_registered_models()
    
    data = []
    for model in models:
        latest_versions = client.get_latest_versions(model.name)
        
        stages = {}
        for mv in latest_versions:
            stages[mv.current_stage] = mv.version
        
        data.append({
            "Name": model.name,
            "Description": model.description or "N/A",
            "Production": stages.get("Production", "-"),
            "Staging": stages.get("Staging", "-"),
            "None": stages.get("None", "-"),
            "Total Versions": len(client.search_model_versions(f"name='{model.name}'"))
        })
    
    return pd.DataFrame(data)


def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 AI Ops Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize clients
    mlflow_client = init_mlflow_client()
    triton_client = init_triton_client()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Overview", "📊 Experiments", "📦 Models", "🚀 Deployment", "📈 Monitoring"]
    )
    
    # Overview Page
    if page == "🏠 Overview":
        st.header("System Overview")
        
        # System status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MLflow Status", "🟢 Live" if mlflow_client else "🔴 Down")
        
        with col2:
            st.metric("Triton Status", "🟢 Live" if triton_client else "🔴 Down")
        
        with col3:
            experiments = get_experiments(mlflow_client)
            st.metric("Experiments", len(experiments))
        
        with col4:
            models = get_registered_models(mlflow_client)
            st.metric("Registered Models", len(models))
        
        st.divider()
        
        # Recent activity
        st.subheader("Recent Activity")
        
        # Get all experiments
        experiments = get_experiments(mlflow_client)
        
        if not experiments.empty:
            # Get runs from first experiment
            exp_id = experiments.iloc[0]["ID"]
            recent_runs = get_recent_runs(mlflow_client, exp_id, limit=5)
            
            if not recent_runs.empty:
                st.dataframe(recent_runs, use_container_width=True)
            else:
                st.info("No runs found")
        else:
            st.info("No experiments found")
        
        st.divider()
        
        # Quick actions
        st.subheader("Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔬 Start Training", use_container_width=True):
                st.info("Training feature coming soon!")
        
        with col2:
            if st.button("📤 Deploy Model", use_container_width=True):
                st.info("Deployment feature coming soon!")
        
        with col3:
            if st.button("📊 View Metrics", use_container_width=True):
                st.info("Metrics feature coming soon!")
    
    # Experiments Page
    elif page == "📊 Experiments":
        st.header("Experiment Tracking")
        
        # List experiments
        experiments = get_experiments(mlflow_client)
        
        if experiments.empty:
            st.info("No experiments found. Start training to create experiments!")
            return
        
        # Select experiment
        selected_exp = st.selectbox(
            "Select Experiment",
            experiments["Name"].tolist()
        )
        
        exp_id = experiments[experiments["Name"] == selected_exp]["ID"].iloc[0]
        
        st.divider()
        
        # Experiment details
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Recent Runs")
            runs = get_recent_runs(mlflow_client, exp_id, limit=20)
            
            if not runs.empty:
                st.dataframe(runs, use_container_width=True)
                
                # Metrics visualization
                st.subheader("Metrics Comparison")
                
                # Get metric columns
                metric_cols = [col for col in runs.columns if col.startswith("metric_")]
                
                if metric_cols:
                    selected_metric = st.selectbox("Select Metric", metric_cols)
                    
                    # Plot metric over runs
                    fig = px.line(
                        runs,
                        x="Run Name",
                        y=selected_metric,
                        title=f"{selected_metric} across runs",
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No runs found in this experiment")
        
        with col2:
            st.subheader("Statistics")
            
            if not runs.empty:
                st.metric("Total Runs", len(runs))
                st.metric("Completed", len(runs[runs["Status"] == "FINISHED"]))
                st.metric("Failed", len(runs[runs["Status"] == "FAILED"]))
                
                # Best run
                metric_cols = [col for col in runs.columns if col.startswith("metric_")]
                if metric_cols:
                    best_col = metric_cols[0]
                    best_value = runs[best_col].max()
                    st.metric(f"Best {best_col}", f"{best_value:.4f}")
    
    # Models Page
    elif page == "📦 Models":
        st.header("Model Registry")
        
        # List models
        models = get_registered_models(mlflow_client)
        
        if models.empty:
            st.info("No registered models found. Train and register models first!")
            return
        
        # Display models
        st.dataframe(models, use_container_width=True)
        
        st.divider()
        
        # Model details
        selected_model = st.selectbox("Select Model", models["Name"].tolist())
        
        if selected_model:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"Model: {selected_model}")
                
                # Get versions
                versions = mlflow_client.search_model_versions(f"name='{selected_model}'")
                versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
                
                version_data = []
                for v in versions:
                    version_data.append({
                        "Version": v.version,
                        "Stage": v.current_stage,
                        "Created": datetime.fromtimestamp(v.creation_timestamp / 1000).strftime("%Y-%m-%d %H:%M"),
                        "Status": v.status,
                        "Description": v.description or "N/A"
                    })
                
                st.dataframe(pd.DataFrame(version_data), use_container_width=True)
            
            with col2:
                st.subheader("Actions")
                
                version_to_promote = st.selectbox(
                    "Select Version",
                    [v["Version"] for v in version_data]
                )
                
                stage = st.selectbox(
                    "Target Stage",
                    ["Staging", "Production", "Archived"]
                )
                
                if st.button("Promote Model", use_container_width=True):
                    try:
                        mlflow_client.transition_model_version_stage(
                            name=selected_model,
                            version=version_to_promote,
                            stage=stage
                        )
                        st.success(f"✓ Model v{version_to_promote} promoted to {stage}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to promote model: {e}")
    
    # Deployment Page
    elif page == "🚀 Deployment":
        st.header("Model Deployment")
        
        if not triton_client:
            st.error("Triton server is not available")
            return
        
        # Server info
        st.subheader("Triton Server Status")
        
        try:
            server_info = triton_client.get_server_metadata()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Server", server_info["name"])
            with col2:
                st.metric("Version", server_info["version"])
            with col3:
                st.metric("Status", "🟢 Ready")
        except Exception as e:
            st.error(f"Failed to get server info: {e}")
        
        st.divider()
        
        # Test inference
        st.subheader("Test Inference")
        
        model_name = st.text_input("Model Name", value="fraud_detector")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Check model status
            if st.button("Check Model Status"):
                try:
                    is_ready = triton_client.is_model_ready(model_name)
                    if is_ready:
                        st.success(f"✓ Model '{model_name}' is ready")
                        
                        # Get metadata
                        metadata = triton_client.get_model_metadata(model_name)
                        st.json(metadata)
                    else:
                        st.error(f"✗ Model '{model_name}' is not ready")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col2:
            # Test inference
            if st.button("Run Test Inference"):
                try:
                    import numpy as np
                    
                    # Generate test data
                    test_data = np.random.rand(1, 10).astype(np.float32)
                    
                    # Infer
                    results = triton_client.infer(
                        model_name=model_name,
                        inputs={"float_input": test_data}
                    )
                    
                    st.success("✓ Inference successful")
                    st.json({k: v.tolist() for k, v in results.items()})
                except Exception as e:
                    st.error(f"Inference failed: {e}")
    
    # Monitoring Page
    elif page == "📈 Monitoring":
        st.header("System Monitoring")
        
        # Prometheus metrics
        st.subheader("API Metrics")
        
        prometheus_url = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
        
        try:
            # Query Prometheus (example)
            st.info("Connect to Grafana for detailed metrics visualization")
            
            grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3000")
            st.markdown(f"[Open Grafana Dashboard]({grafana_url})")
            
        except Exception as e:
            st.error(f"Failed to fetch metrics: {e}")
        
        st.divider()
        
        # Model statistics
        st.subheader("Model Statistics")
        
        if triton_client:
            try:
                stats = triton_client.get_inference_statistics()
                
                if stats["model_stats"]:
                    stats_df = pd.DataFrame(stats["model_stats"])
                    st.dataframe(stats_df, use_container_width=True)
                else:
                    st.info("No model statistics available")
            except Exception as e:
                st.error(f"Failed to get statistics: {e}")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>AI Ops Dashboard v1.0.0 | Built with MLflow + Triton + Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
