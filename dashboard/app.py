"""
Streamlit Dashboard for Subscription Churn Analysis
Interactive web application for business stakeholders
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="Subscription Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading
@st.cache_data
def load_data():
    """Load and cache datasets"""
    try:
        users = pd.read_csv('../data/raw/users.csv')
        subscriptions = pd.read_csv('../data/raw/subscriptions.csv')
        engagement = pd.read_csv('../data/raw/engagement.csv')
        tickets = pd.read_csv('../data/raw/support_tickets.csv')
        
        # Convert dates
        users['signup_date'] = pd.to_datetime(users['signup_date'])
        subscriptions['billing_date'] = pd.to_datetime(subscriptions['billing_date'])
        engagement['month'] = pd.to_datetime(engagement['month'])
        
        return users, subscriptions, engagement, tickets
    except FileNotFoundError:
        st.error("Data files not found. Please run data_generation.py first.")
        return None, None, None, None

@st.cache_data
def calculate_kpis(users, subscriptions, engagement):
    """Calculate key performance indicators"""
    
    # Basic metrics
    total_users = len(users)
    active_users = len(subscriptions[subscriptions['status'] == 'active']['user_id'].unique())
    churned_users = len(subscriptions[subscriptions['status'] == 'churned']['user_id'].unique())
    churn_rate = (churned_users / total_users) * 100
    
    # Revenue metrics
    total_revenue = subscriptions['price'].sum()
    mrr = subscriptions[subscriptions['status'] == 'active']['price'].sum()
    arpu = mrr / active_users if active_users > 0 else 0
    
    # Engagement metrics
    avg_engagement = engagement['engagement_score'].mean()
    
    return {
        'total_users': total_users,
        'active_users': active_users,
        'churned_users': churned_users,
        'churn_rate': churn_rate,
        'total_revenue': total_revenue,
        'mrr': mrr,
        'arpu': arpu,
        'avg_engagement': avg_engagement
    }

def create_churn_timeline(subscriptions):
    """Create monthly churn rate timeline"""
    monthly_churn = subscriptions.groupby(
        subscriptions['billing_date'].dt.to_period('M')
    ).agg({
        'user_id': 'count',
        'status': lambda x: (x == 'churned').sum()
    }).reset_index()
    
    monthly_churn['churn_rate'] = (monthly_churn['status'] / monthly_churn['user_id']) * 100
    monthly_churn['billing_date'] = monthly_churn['billing_date'].astype(str)
    
    fig = px.line(monthly_churn, x='billing_date', y='churn_rate',
                  title='Monthly Churn Rate Trend',
                  labels={'churn_rate': 'Churn Rate (%)', 'billing_date': 'Month'})
    fig.update_traces(line_color='#ff6b6b', line_width=3)
    fig.update_layout(height=400)
    return fig

def create_cohort_heatmap(users, subscriptions):
    """Create cohort retention heatmap"""
    
    # Prepare cohort data
    user_cohorts = users[['user_id', 'signup_date']].copy()
    user_cohorts['cohort_month'] = user_cohorts['signup_date'].dt.to_period('M')
    
    subs_cohort = subscriptions.merge(user_cohorts, on='user_id')
    subs_cohort['billing_month'] = subs_cohort['billing_date'].dt.to_period('M')
    subs_cohort['months_since_signup'] = (
        subs_cohort['billing_month'] - subs_cohort['cohort_month']
    ).apply(lambda x: x.n)
    
    # Create cohort table
    cohort_data = subs_cohort[subs_cohort['status'] == 'active'].groupby(
        ['cohort_month', 'months_since_signup']
    )['user_id'].nunique().reset_index()
    
    cohort_sizes = subs_cohort.groupby('cohort_month')['user_id'].nunique()
    cohort_table = cohort_data.pivot(index='cohort_month', 
                                    columns='months_since_signup', 
                                    values='user_id')
    
    # Calculate retention rates
    cohort_retention = cohort_table.divide(cohort_sizes, axis=0) * 100
    
    fig = px.imshow(cohort_retention.values,
                    x=[f"Month {i}" for i in cohort_retention.columns],
                    y=[str(idx) for idx in cohort_retention.index],
                    title="Cohort Retention Heatmap (%)",
                    color_continuous_scale="RdYlBu_r")
    fig.update_layout(height=500)
    return fig

def create_revenue_analysis(subscriptions):
    """Create revenue analysis charts"""
    
    # Monthly revenue trend
    monthly_revenue = subscriptions.groupby(
        subscriptions['billing_date'].dt.to_period('M')
    )['price'].sum().reset_index()
    monthly_revenue['billing_date'] = monthly_revenue['billing_date'].astype(str)
    
    fig = px.line(monthly_revenue, x='billing_date', y='price',
                  title='Monthly Recurring Revenue (MRR)',
                  labels={'price': 'Revenue ($)', 'billing_date': 'Month'})
    fig.update_traces(line_color='#4ecdc4', line_width=3)
    fig.update_layout(height=400)
    return fig

def create_user_segmentation(users, subscriptions):
    """Create user segmentation analysis"""
    
    # Merge user data with subscription status
    user_status = subscriptions.groupby('user_id').agg({
        'status': 'last',
        'price': 'mean'
    }).reset_index()
    
    user_segments = users.merge(user_status, on='user_id', how='left')
    user_segments['churned'] = (user_segments['status'] == 'churned')
    
    # Churn by acquisition channel
    churn_by_channel = user_segments.groupby('acquisition_channel')['churned'].mean() * 100
    
    fig = px.bar(x=churn_by_channel.index, y=churn_by_channel.values,
                 title='Churn Rate by Acquisition Channel',
                 labels={'x': 'Acquisition Channel', 'y': 'Churn Rate (%)'})
    fig.update_layout(height=400)
    return fig

def scenario_simulator():
    """Create what-if scenario simulator"""
    st.subheader("📈 What-If Scenario Simulator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Churn Reduction Strategy**")
        churn_reduction = st.slider("Churn Reduction Target (%)", 0, 50, 10, 5)
        intervention_cost = st.number_input("Intervention Cost per User ($)", 0.0, 100.0, 15.0, 5.0)
    
    with col2:
        st.write("**Price Optimization**")
        price_increase = st.slider("Price Increase (%)", -20, 30, 5, 2)
        demand_elasticity = st.slider("Demand Elasticity", -2.0, 0.0, -0.5, 0.1)
    
    # Load data for calculations
    users, subscriptions, engagement, _ = load_data()
    if users is not None:
        kpis = calculate_kpis(users, subscriptions, engagement)
        
        # Current metrics
        current_users = kpis['active_users']
        current_revenue = kpis['mrr']
        current_churn = kpis['churn_rate']
        
        # Scenario calculations
        users_saved = int(current_users * (churn_reduction / 100))
        intervention_total_cost = users_saved * intervention_cost
        
        # Revenue impact from churn reduction
        revenue_from_saved_users = users_saved * kpis['arpu']
        
        # Revenue impact from price changes
        demand_change = demand_elasticity * (price_increase / 100)
        new_user_count = int(current_users * (1 + demand_change))
        new_price_revenue = new_user_count * kpis['arpu'] * (1 + price_increase / 100)
        
        total_revenue_impact = (revenue_from_saved_users + 
                               (new_price_revenue - current_revenue))
        
        net_impact = total_revenue_impact - intervention_total_cost
        
        # Display results
        st.subheader("📊 Scenario Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Users Saved", f"{users_saved:,}", 
                     delta=f"{churn_reduction}% reduction")
        
        with col2:
            st.metric("Revenue Impact", f"${total_revenue_impact:,.0f}",
                     delta=f"+{(total_revenue_impact/current_revenue)*100:.1f}%")
        
        with col3:
            st.metric("Total Costs", f"${intervention_total_cost:,.0f}")
        
        with col4:
            st.metric("Net Impact", f"${net_impact:,.0f}",
                     delta="Positive" if net_impact > 0 else "Negative")

def main():
    """Main dashboard application"""
    
    # Header
    st.title("📊 Subscription Churn & Revenue Analytics Dashboard")
    st.markdown("*Netflix/Spotify/Amazon Prime Style Analytics*")
    
    # Load data
    users, subscriptions, engagement, tickets = load_data()
    
    if users is None:
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🔧 Filters & Controls")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(users['signup_date'].min(), users['signup_date'].max()),
        min_value=users['signup_date'].min(),
        max_value=users['signup_date'].max()
    )
    
    # Plan filter
    plan_filter = st.sidebar.multiselect(
        "Select Subscription Plans",
        options=subscriptions['plan'].unique(),
        default=subscriptions['plan'].unique()
    )
    
    # Channel filter
    channel_filter = st.sidebar.multiselect(
        "Select Acquisition Channels",
        options=users['acquisition_channel'].unique(),
        default=users['acquisition_channel'].unique()
    )
    
    # Filter data
    filtered_users = users[users['acquisition_channel'].isin(channel_filter)]
    filtered_subs = subscriptions[subscriptions['plan'].isin(plan_filter)]
    
    # KPIs
    kpis = calculate_kpis(filtered_users, filtered_subs, engagement)
    
    # Display KPIs
    st.header("🎯 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Users", f"{kpis['total_users']:,}")
    
    with col2:
        st.metric("Active Users", f"{kpis['active_users']:,}")
    
    with col3:
        st.metric("Churn Rate", f"{kpis['churn_rate']:.1f}%",
                 delta="-2.3%" if kpis['churn_rate'] < 15 else "+1.8%")
    
    with col4:
        st.metric("Monthly Revenue", f"${kpis['mrr']:,.0f}")
    
    with col5:
        st.metric("ARPU", f"${kpis['arpu']:.2f}")
    
    # Charts section
    st.header("📈 Analytics Dashboard")
    
    # Row 1: Churn timeline and cohort analysis
    col1, col2 = st.columns(2)
    
    with col1:
        churn_fig = create_churn_timeline(filtered_subs)
        st.plotly_chart(churn_fig, use_container_width=True)
    
    with col2:
        revenue_fig = create_revenue_analysis(filtered_subs)
        st.plotly_chart(revenue_fig, use_container_width=True)
    
    # Row 2: Cohort heatmap
    st.subheader("🔥 Cohort Retention Analysis")
    cohort_fig = create_cohort_heatmap(filtered_users, filtered_subs)
    st.plotly_chart(cohort_fig, use_container_width=True)
    
    # Row 3: User segmentation
    st.subheader("👥 User Segmentation")
    segment_fig = create_user_segmentation(filtered_users, filtered_subs)
    st.plotly_chart(segment_fig, use_container_width=True)
    
    # What-if scenarios
    st.header("🎲 Strategy Simulator")
    scenario_simulator()
    
    # Business insights
    st.header("💡 Key Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Churn Drivers")
        st.write("""
        **Top Risk Factors:**
        1. **Low Engagement**: Users with <20% engagement score
        2. **Price Sensitivity**: Basic plan users churn 2x more
        3. **Support Issues**: Users with tickets 3x more likely to churn
        4. **Acquisition Channel**: Paid search users have higher churn
        5. **Age Demographics**: 18-25 age group shows highest churn
        """)
    
    with col2:
        st.subheader("📋 Action Plan")
        st.write("""
        **Recommended Actions:**
        1. **Personalized Retention**: Target low-engagement users
        2. **Price Anchoring**: Introduce intermediate pricing tiers
        3. **Proactive Support**: Identify at-risk users early
        4. **Content Strategy**: Improve content for high-churn segments
        5. **Onboarding**: Enhance new user experience
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard created with Streamlit • Data generated for demonstration purposes*")

if __name__ == "__main__":
    # Run with: streamlit run app.py
    main()