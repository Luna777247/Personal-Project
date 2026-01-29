# dashboard.py
import streamlit as st
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="News Bias Detector", layout="wide")

def load_analysis_data():
    """Load bias analysis results"""
    analysis_file = Path("data/analysis/bias_analysis.json")
    if analysis_file.exists():
        with open(analysis_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def create_bias_chart(data):
    """Create bias score visualization"""
    events = []
    avg_scores = []
    article_counts = []
    
    for event, event_data in data.items():
        summary = event_data["event_summary"]
        events.append(event)
        avg_scores.append(summary["average_bias_score"])
        article_counts.append(summary["total_articles"])
    
    df = pd.DataFrame({
        "Event": events,
        "Average Bias Score": avg_scores,
        "Article Count": article_counts
    })
    
    fig = px.bar(df, x="Event", y="Average Bias Score", 
                 title="Average Headline Bias Score by Event",
                 color="Article Count")
    return fig

def create_source_comparison(data):
    """Compare bias by source"""
    source_bias = {}
    
    for event_data in data.values():
        for article in event_data["articles"]:
            source = article["source"]
            score = article["headline_bias_score"]
            
            if source not in source_bias:
                source_bias[source] = []
            source_bias[source].append(score)
    
    sources = []
    avg_bias = []
    
    for source, scores in source_bias.items():
        sources.append(source)
        avg_bias.append(sum(scores) / len(scores))
    
    df = pd.DataFrame({
        "Source": sources,
        "Average Bias Score": avg_bias
    })
    
    fig = px.bar(df, x="Source", y="Average Bias Score",
                 title="Average Bias Score by News Source")
    return fig

def main():
    st.title("📰 News Bias Detector Dashboard")
    st.markdown("Phát hiện thiên lệch & giật tít trong tin tức thiên tai Việt Nam")
    
    data = load_analysis_data()
    
    if not data:
        st.error("No analysis data found. Please run bias analysis first.")
        return
    
    # Summary metrics
    total_events = len(data)
    total_articles = sum(event_data["event_summary"]["total_articles"] for event_data in data.values())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Events", total_events)
    with col2:
        st.metric("Total Articles", total_articles)
    with col3:
        avg_bias = sum(event_data["event_summary"]["average_bias_score"] for event_data in data.values()) / total_events
        st.metric("Average Bias Score", f"{avg_bias:.3f}")
    
    # Charts
    st.subheader("📊 Bias Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        bias_chart = create_bias_chart(data)
        st.plotly_chart(bias_chart, use_container_width=True)
    
    with col2:
        source_chart = create_source_comparison(data)
        st.plotly_chart(source_chart, use_container_width=True)
    
    # Detailed view
    st.subheader("📋 Detailed Event Analysis")
    
    selected_event = st.selectbox("Select Event", list(data.keys()))
    
    if selected_event:
        event_data = data[selected_event]
        summary = event_data["event_summary"]
        
        st.write(f"**Event: {selected_event}**")
        st.write(f"Articles: {summary['total_articles']}")
        st.write(f"Average Bias: {summary['average_bias_score']:.3f}")
        
        # Article table
        articles_df = pd.DataFrame(event_data["articles"])
        if not articles_df.empty:
            st.dataframe(articles_df[["source", "title", "headline_bias_score", "exaggeration_level"]])

if __name__ == "__main__":
    main()