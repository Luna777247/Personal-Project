import pandas as pd
import numpy as np

def calculate_revenue_kpis(df):
    """Calculate revenue-related KPIs"""
    kpis = {}
    
    # Total revenue
    kpis['total_revenue'] = df['revenue'].sum()
    
    # Revenue by category
    if 'category' in df.columns:
        kpis['revenue_by_category'] = df.groupby('category')['revenue'].sum().to_dict()
    
    # Monthly revenue growth
    if 'date' in df.columns:
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly_revenue = df.groupby('month')['revenue'].sum()
        kpis['monthly_revenue_growth'] = monthly_revenue.pct_change().to_dict()
    
    # Average order value
    if 'order_id' in df.columns:
        kpis['avg_order_value'] = df.groupby('order_id')['revenue'].sum().mean()
    
    return kpis

def calculate_user_kpis(df):
    """Calculate user behavior KPIs"""
    kpis = {}
    
    # Total users
    kpis['total_users'] = df['user_id'].nunique()
    
    # User acquisition (monthly)
    if 'date' in df.columns:
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly_users = df.groupby('month')['user_id'].nunique()
        kpis['monthly_user_acquisition'] = monthly_users.to_dict()
    
    # User retention (simplified)
    if 'user_type' in df.columns:
        retention_rate = df[df['user_type'] == 'returning']['user_id'].nunique() / df['user_id'].nunique()
        kpis['user_retention_rate'] = retention_rate
    
    # Average session duration (if available)
    if 'session_duration' in df.columns:
        kpis['avg_session_duration'] = df['session_duration'].mean()
    
    return kpis

def calculate_growth_kpis(df):
    """Calculate growth-related KPIs"""
    kpis = {}
    
    # Revenue growth rate
    if 'date' in df.columns:
        df_sorted = df.sort_values('date')
        total_periods = len(df_sorted['date'].unique())
        if total_periods > 1:
            initial_revenue = df_sorted.groupby('date')['revenue'].sum().iloc[0]
            final_revenue = df_sorted.groupby('date')['revenue'].sum().iloc[-1]
            kpis['revenue_growth_rate'] = (final_revenue - initial_revenue) / initial_revenue
    
    # User growth rate
    if 'date' in df.columns:
        monthly_users = df.groupby(pd.to_datetime(df['date']).dt.to_period('M'))['user_id'].nunique()
        if len(monthly_users) > 1:
            kpis['user_growth_rate'] = monthly_users.pct_change().mean()
    
    return kpis

def calculate_all_kpis(df):
    """Calculate all KPIs"""
    revenue_kpis = calculate_revenue_kpis(df)
    user_kpis = calculate_user_kpis(df)
    growth_kpis = calculate_growth_kpis(df)
    
    all_kpis = {
        'revenue_kpis': revenue_kpis,
        'user_kpis': user_kpis,
        'growth_kpis': growth_kpis
    }
    
    return all_kpis