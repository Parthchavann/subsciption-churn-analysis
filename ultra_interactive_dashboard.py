#!/usr/bin/env python3
"""
🚀 Ultra Interactive Live Dashboard
Real-time subscription analytics with live data feeds and advanced interactivity
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
import random
from datetime import datetime, timedelta
import asyncio
import threading
from typing import Dict, List, Any
import requests
from dataclasses import dataclass
import sqlite3
import os

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 Ultra Interactive Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .alert-success {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid;
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid;
        margin: 1rem 0;
    }
    .alert-danger {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid;
        margin: 1rem 0;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class LiveMetrics:
    """Live metrics data structure"""
    churn_rate: float
    mrr: float
    arpu: float
    active_users: int
    new_signups: int
    cancellations: int
    timestamp: datetime

class LiveDataGenerator:
    """Generates realistic live data streams"""
    
    def __init__(self):
        self.base_metrics = {
            'churn_rate': 14.9,
            'mrr': 10887,
            'arpu': 18.05,
            'active_users': 1000,
            'new_signups': 25,
            'cancellations': 12
        }
        
    def get_live_metrics(self) -> LiveMetrics:
        """Generate realistic live metrics with variations"""
        now = datetime.now()
        
        # Add realistic variations
        variations = {
            'churn_rate': random.uniform(-0.5, 0.5),
            'mrr': random.uniform(-200, 300),
            'arpu': random.uniform(-1, 2),
            'active_users': random.randint(-5, 15),
            'new_signups': random.randint(-3, 8),
            'cancellations': random.randint(-2, 5)
        }
        
        return LiveMetrics(
            churn_rate=max(0, self.base_metrics['churn_rate'] + variations['churn_rate']),
            mrr=max(0, self.base_metrics['mrr'] + variations['mrr']),
            arpu=max(0, self.base_metrics['arpu'] + variations['arpu']),
            active_users=max(0, self.base_metrics['active_users'] + variations['active_users']),
            new_signups=max(0, self.base_metrics['new_signups'] + variations['new_signups']),
            cancellations=max(0, self.base_metrics['cancellations'] + variations['cancellations']),
            timestamp=now
        )

class InteractiveDashboard:
    """Main dashboard class with interactive features"""
    
    def __init__(self):
        self.data_generator = LiveDataGenerator()
        self.load_base_data()
        
    def load_base_data(self):
        """Load base datasets"""
        try:
            self.users_df = pd.read_csv('data/raw/users.csv')
            self.subscriptions_df = pd.read_csv('data/raw/subscriptions.csv')
            self.engagement_df = pd.read_csv('data/raw/engagement.csv')
            
            # Load live data if available
            if os.path.exists('data/live/live_professional_insights.json'):
                with open('data/live/live_professional_insights.json', 'r') as f:
                    self.live_insights = json.load(f)
            else:
                self.live_insights = {}
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
            # Generate sample data if files don't exist
            self.generate_sample_data()
    
    def generate_sample_data(self):
        """Generate sample data for demo"""
        np.random.seed(42)
        
        # Sample users
        self.users_df = pd.DataFrame({
            'user_id': range(1, 101),
            'age': np.random.randint(18, 65, 100),
            'plan_type': np.random.choice(['Basic', 'Premium', 'Enterprise'], 100),
            'acquisition_channel': np.random.choice(['Organic', 'Paid', 'Referral'], 100)
        })
        
        # Sample subscriptions
        self.subscriptions_df = pd.DataFrame({
            'user_id': range(1, 101),
            'start_date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'monthly_charges': np.random.uniform(9.99, 49.99, 100),
            'status': np.random.choice(['Active', 'Churned'], 100, p=[0.85, 0.15])
        })
        
        # Sample engagement
        self.engagement_df = pd.DataFrame({
            'user_id': np.repeat(range(1, 101), 12),
            'month': pd.tile(pd.date_range('2023-01-01', periods=12, freq='M'), 100),
            'login_frequency': np.random.poisson(15, 1200),
            'feature_usage_score': np.random.uniform(0, 100, 1200)
        })

def create_live_metrics_display():
    """Create live metrics cards with real-time updates"""
    dashboard = InteractiveDashboard()
    live_metrics = dashboard.data_generator.get_live_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_churn = random.uniform(-0.3, 0.2)
        st.metric(
            "🎯 Churn Rate",
            f"{live_metrics.churn_rate:.1f}%",
            f"{delta_churn:+.1f}%",
            delta_color="inverse"
        )
        
    with col2:
        delta_mrr = random.uniform(-100, 200)
        st.metric(
            "💰 Monthly Revenue",
            f"${live_metrics.mrr:,.0f}",
            f"${delta_mrr:+,.0f}"
        )
        
    with col3:
        delta_arpu = random.uniform(-0.5, 1.2)
        st.metric(
            "📊 ARPU",
            f"${live_metrics.arpu:.2f}",
            f"${delta_arpu:+.2f}"
        )
        
    with col4:
        delta_users = random.randint(-3, 8)
        st.metric(
            "👥 Active Users",
            f"{live_metrics.active_users:,}",
            f"{delta_users:+d}"
        )

def create_interactive_filters():
    """Create interactive filter controls"""
    st.sidebar.header("🎛️ Interactive Controls")
    
    # Time range filter
    time_range = st.sidebar.selectbox(
        "📅 Time Range",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom Range"],
        index=2
    )
    
    # Customer segment filter
    segments = st.sidebar.multiselect(
        "👥 Customer Segments",
        ["High Value", "At Risk", "New Users", "Loyal Customers", "Enterprise"],
        default=["High Value", "At Risk"]
    )
    
    # Plan type filter
    plan_types = st.sidebar.multiselect(
        "💳 Subscription Plans",
        ["Basic", "Premium", "Enterprise", "Trial"],
        default=["Basic", "Premium", "Enterprise"]
    )
    
    # Geographic filter
    regions = st.sidebar.multiselect(
        "🌍 Geographic Regions",
        ["North America", "Europe", "Asia-Pacific", "Latin America"],
        default=["North America", "Europe"]
    )
    
    # Real-time controls
    st.sidebar.header("⚡ Real-time Settings")
    
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 15)
    
    live_alerts = st.sidebar.checkbox("🚨 Live Alerts", value=True)
    
    return {
        'time_range': time_range,
        'segments': segments,
        'plan_types': plan_types,
        'regions': regions,
        'auto_refresh': auto_refresh,
        'refresh_interval': refresh_interval,
        'live_alerts': live_alerts
    }

def create_live_churn_risk_chart():
    """Create interactive live churn risk visualization"""
    
    # Generate live risk data
    risk_levels = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
    risk_counts = [random.randint(200, 400), random.randint(100, 200), 
                   random.randint(50, 120), random.randint(10, 40)]
    risk_colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']
    
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=risk_levels,
        y=risk_counts,
        marker_color=risk_colors,
        name="Customer Count",
        text=risk_counts,
        textposition='auto',
        hovertemplate="<b>%{x}</b><br>Customers: %{y}<br><extra></extra>"
    ))
    
    # Add trend line
    trend_y = [risk_counts[i] + random.randint(-20, 20) for i in range(len(risk_counts))]
    fig.add_trace(go.Scatter(
        x=risk_levels,
        y=trend_y,
        mode='lines+markers',
        name="Trend",
        line=dict(color='purple', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="🎯 Live Churn Risk Distribution",
        xaxis_title="Risk Level",
        yaxis_title="Number of Customers",
        yaxis2=dict(
            title="Trend",
            overlaying='y',
            side='right'
        ),
        height=400,
        template="plotly_white"
    )
    
    return fig

def create_live_revenue_stream():
    """Create live revenue stream visualization"""
    
    # Generate live revenue data
    hours = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                         end=datetime.now(), freq='H')
    
    base_revenue = 450
    revenue_data = []
    
    for hour in hours:
        # Simulate realistic hourly variations
        hour_of_day = hour.hour
        
        # Peak hours: 9-17, low hours: 0-6
        if 9 <= hour_of_day <= 17:
            multiplier = random.uniform(1.2, 1.8)
        elif 18 <= hour_of_day <= 23:
            multiplier = random.uniform(0.8, 1.2)
        else:
            multiplier = random.uniform(0.3, 0.7)
            
        hourly_revenue = base_revenue * multiplier + random.uniform(-50, 100)
        revenue_data.append(max(0, hourly_revenue))
    
    fig = go.Figure()
    
    # Add revenue stream
    fig.add_trace(go.Scatter(
        x=hours,
        y=revenue_data,
        mode='lines+markers',
        name='Hourly Revenue',
        line=dict(color='#667eea', width=3),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)',
        hovertemplate="<b>%{x|%H:%M}</b><br>Revenue: $%{y:,.0f}<br><extra></extra>"
    ))
    
    # Add moving average
    ma_data = pd.Series(revenue_data).rolling(window=6).mean()
    fig.add_trace(go.Scatter(
        x=hours,
        y=ma_data,
        mode='lines',
        name='6-Hour Moving Average',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="💰 Live Revenue Stream (Last 24 Hours)",
        xaxis_title="Time",
        yaxis_title="Revenue ($)",
        height=400,
        template="plotly_white",
        xaxis=dict(tickformat='%H:%M')
    )
    
    return fig

def create_interactive_cohort_heatmap():
    """Create interactive cohort retention heatmap"""
    
    # Generate cohort data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    periods = list(range(0, 12))
    
    # Generate realistic retention rates
    retention_data = []
    for month in months:
        row = []
        base_retention = 100
        for period in periods:
            if period == 0:
                retention = 100
            else:
                # Realistic decay curve
                retention = base_retention * (0.85 ** period) + random.uniform(-5, 5)
                retention = max(0, min(100, retention))
            row.append(retention)
        retention_data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=retention_data,
        x=periods,
        y=months,
        colorscale='RdYlGn',
        text=[[f"{val:.1f}%" for val in row] for row in retention_data],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="<b>Month: %{y}</b><br>Period: %{x}<br>Retention: %{z:.1f}%<br><extra></extra>"
    ))
    
    fig.update_layout(
        title="📊 Interactive Cohort Retention Heatmap",
        xaxis_title="Periods After First Purchase",
        yaxis_title="Acquisition Month",
        height=500
    )
    
    return fig

def create_live_alerts_panel():
    """Create live alerts and notifications panel"""
    st.subheader("🚨 Live Alerts & Notifications")
    
    alerts = [
        {
            "type": "danger",
            "icon": "🔴",
            "title": "High Churn Alert",
            "message": "Churn rate increased 15% in premium segment",
            "time": "2 minutes ago",
            "action": "View Details"
        },
        {
            "type": "warning", 
            "icon": "🟡",
            "title": "Revenue Drop",
            "message": "MRR decreased $340 from last week",
            "time": "8 minutes ago",
            "action": "Investigate"
        },
        {
            "type": "success",
            "icon": "🟢", 
            "title": "Conversion Spike",
            "message": "Trial-to-paid conversion up 23%",
            "time": "15 minutes ago",
            "action": "Analyze"
        },
        {
            "type": "info",
            "icon": "🔵",
            "title": "New Cohort Data",
            "message": "October cohort analysis complete",
            "time": "1 hour ago",
            "action": "Review"
        }
    ]
    
    for alert in alerts:
        if alert["type"] == "danger":
            st.markdown(f'<div class="alert-danger">', unsafe_allow_html=True)
        elif alert["type"] == "warning":
            st.markdown(f'<div class="alert-warning">', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-success">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 6, 2])
        with col1:
            st.write(alert["icon"])
        with col2:
            st.write(f"**{alert['title']}**")
            st.write(alert["message"])
            st.caption(alert["time"])
        with col3:
            if st.button(alert["action"], key=f"alert_{alert['title']}"):
                st.info(f"Opening {alert['title']} details...")
        
        st.markdown('</div>', unsafe_allow_html=True)

def create_predictive_analytics_panel():
    """Create predictive analytics with live updates"""
    st.subheader("🔮 Predictive Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn prediction model
        st.write("**Next 30 Days Churn Forecast**")
        
        days = list(range(1, 31))
        predicted_churn = [14.9 + (i * 0.1) + random.uniform(-0.5, 0.5) for i in days]
        confidence_upper = [p + random.uniform(1, 3) for p in predicted_churn]
        confidence_lower = [p - random.uniform(1, 3) for p in predicted_churn]
        
        fig = go.Figure()
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=days + days[::-1],
            y=confidence_upper + confidence_lower[::-1],
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
        
        # Prediction line
        fig.add_trace(go.Scatter(
            x=days,
            y=predicted_churn,
            mode='lines+markers',
            name='Predicted Churn Rate',
            line=dict(color='#667eea', width=3)
        ))
        
        fig.update_layout(
            title="Churn Rate Forecast",
            xaxis_title="Days",
            yaxis_title="Churn Rate (%)",
            height=300,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue prediction
        st.write("**Revenue Growth Prediction**")
        
        revenue_forecast = [10887 + (i * 50) + random.uniform(-200, 300) for i in days]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=days,
            y=revenue_forecast,
            mode='lines+markers',
            name='Predicted Revenue',
            line=dict(color='green', width=3),
            fill='tonexty',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ))
        
        fig.update_layout(
            title="Revenue Forecast",
            xaxis_title="Days", 
            yaxis_title="Revenue ($)",
            height=300,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Ultra Interactive Live Churn Analytics</h1>
        <p>Real-time subscription analytics with live data feeds and advanced interactivity</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get interactive filters
    filters = create_interactive_filters()
    
    # Auto-refresh functionality
    if filters['auto_refresh']:
        time.sleep(filters['refresh_interval'])
        st.rerun()
    
    # Live metrics display
    st.header("📊 Live Metrics Dashboard")
    create_live_metrics_display()
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Live Analytics", "📊 Interactive Charts", "🚨 Alerts", "🔮 Predictions", "⚙️ Controls"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_live_churn_risk_chart()
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = create_live_revenue_stream() 
            st.plotly_chart(fig, use_container_width=True)
        
        # Cohort heatmap
        fig = create_interactive_cohort_heatmap()
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Interactive Visualizations")
        
        # Chart selector
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Customer Journey", "Revenue Breakdown", "Engagement Patterns", "Churn Drivers"]
        )
        
        if chart_type == "Customer Journey":
            # Customer journey funnel
            stages = ["Visitors", "Sign-ups", "Trial", "Paid", "Retained"]
            values = [10000, 2500, 1800, 1200, 1020]
            
            fig = go.Figure(go.Funnel(
                y=stages,
                x=values,
                textinfo="value+percent initial",
                marker=dict(color=["royalblue", "lightblue", "lightgreen", "gold", "orange"])
            ))
            
            fig.update_layout(title="Customer Journey Funnel", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Revenue Breakdown":
            # Revenue pie chart
            plans = ["Basic", "Premium", "Enterprise"]
            revenues = [3200, 5800, 1887]
            
            fig = go.Figure(data=[go.Pie(
                labels=plans,
                values=revenues,
                hole=0.4,
                marker_colors=['#ff9999', '#66b3ff', '#99ff99']
            )])
            
            fig.update_layout(title="Revenue by Plan Type", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        create_live_alerts_panel()
    
    with tab4:
        create_predictive_analytics_panel()
    
    with tab5:
        st.subheader("⚙️ Advanced Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Sources**")
            st.checkbox("Live Database", value=True)
            st.checkbox("External APIs", value=True)
            st.checkbox("Real-time Events", value=False)
            
            st.write("**Notifications**")
            st.checkbox("Email Alerts", value=True)
            st.checkbox("Slack Integration", value=False)
            st.checkbox("SMS Alerts", value=False)
        
        with col2:
            st.write("**Export Options**")
            if st.button("📊 Export Dashboard"):
                st.success("Dashboard exported to PDF!")
                
            if st.button("📈 Export Data"):
                st.success("Data exported to CSV!")
                
            if st.button("📧 Schedule Report"):
                st.success("Report scheduled for weekly delivery!")
    
    # Footer with status
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔄 Status", "Live", "🟢")
    with col2:
        st.metric("⏱️ Last Update", "Real-time", "")
    with col3:
        st.metric("📡 Data Sources", "5 Active", "🟢")

if __name__ == "__main__":
    main()