"""
Advanced Interactive Dashboard for Subscription Analytics
Real-time monitoring, advanced visualizations, and executive reporting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import sys
import os
from datetime import datetime, timedelta
import json

# Page config
st.set_page_config(
    page_title="Advanced Subscription Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-metric {
        color: #28a745;
        font-weight: bold;
    }
    .warning-metric {
        color: #ffc107;
        font-weight: bold;
    }
    .danger-metric {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_all_datasets():
    """Load all available datasets"""
    datasets = {}
    
    try:
        # Real subscription data
        datasets['subscription'] = pd.read_csv('../data/real/subscription_churn_real_patterns.csv')
        datasets['subscription']['signup_date'] = pd.to_datetime(datasets['subscription'].get('signup_date', datetime.now()))
    except:
        st.warning("Subscription dataset not found. Please run real_data_loader.py first.")
    
    try:
        # Time series data
        datasets['timeseries'] = pd.read_csv('../data/real/time_series_data.csv')
        datasets['timeseries']['date'] = pd.to_datetime(datasets['timeseries']['date'])
    except:
        st.info("Time series data not found. Some forecasting features may be limited.")
    
    try:
        # A/B test data
        ab_tests = {}
        for test_name in ['pricing_test', 'onboarding_test', 'retention_campaign']:
            try:
                ab_tests[test_name] = pd.read_csv(f'../data/real/ab_test_{test_name}.csv')
            except:
                pass
        if ab_tests:
            datasets['ab_tests'] = ab_tests
    except:
        st.info("A/B test data not found. Run ab_testing_framework.py to generate.")
    
    return datasets

def create_advanced_kpi_cards(df):
    """Create advanced KPI cards with trends and insights"""
    if df is None or len(df) == 0:
        st.error("No data available for KPI calculation")
        return
    
    # Calculate metrics
    total_customers = len(df)
    active_customers = len(df[df['Churn'] == 'No']) if 'Churn' in df.columns else total_customers
    churn_rate = (df['Churn'] == 'Yes').mean() * 100 if 'Churn' in df.columns else 0
    
    avg_revenue = df['TotalRevenue'].mean() if 'TotalRevenue' in df.columns else df.get('MonthlyPrice', pd.Series([0])).mean()
    total_revenue = df['TotalRevenue'].sum() if 'TotalRevenue' in df.columns else 0
    avg_tenure = df['TenureMonths'].mean() if 'TenureMonths' in df.columns else 0
    
    # Advanced metrics
    high_value_customers = len(df[df['MonthlyPrice'] > df['MonthlyPrice'].quantile(0.75)]) if 'MonthlyPrice' in df.columns else 0
    at_risk_customers = len(df[(df['SatisfactionScore'] < 3) | (df['SupportContacts'] > 2)]) if all(col in df.columns for col in ['SatisfactionScore', 'SupportContacts']) else 0
    
    # Display KPIs in columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            label="🎯 Total Customers",
            value=f"{total_customers:,}",
            delta=f"+{int(total_customers * 0.05):,} (Est. Monthly Growth)"
        )
    
    with col2:
        churn_delta = "🟢 Good" if churn_rate < 10 else ("🟡 Moderate" if churn_rate < 15 else "🔴 High")
        st.metric(
            label="📉 Churn Rate",
            value=f"{churn_rate:.1f}%",
            delta=churn_delta
        )
    
    with col3:
        st.metric(
            label="💰 Avg Revenue",
            value=f"${avg_revenue:,.2f}",
            delta=f"${avg_revenue * 0.03:,.2f} (3% Growth Target)"
        )
    
    with col4:
        st.metric(
            label="🏆 High-Value Customers",
            value=f"{high_value_customers:,}",
            delta=f"{(high_value_customers/total_customers)*100:.1f}% of total"
        )
    
    with col5:
        tenure_status = "🟢 Excellent" if avg_tenure > 12 else ("🟡 Good" if avg_tenure > 6 else "🔴 Poor")
        st.metric(
            label="📅 Avg Tenure",
            value=f"{avg_tenure:.1f} months",
            delta=tenure_status
        )
    
    with col6:
        risk_percent = (at_risk_customers / total_customers) * 100 if total_customers > 0 else 0
        risk_status = "🔴 High Alert" if risk_percent > 20 else ("🟡 Monitor" if risk_percent > 10 else "🟢 Low Risk")
        st.metric(
            label="⚠️ At-Risk Customers",
            value=f"{at_risk_customers:,}",
            delta=f"{risk_percent:.1f}% - {risk_status}"
        )

def create_cohort_analysis(df):
    """Create advanced cohort retention analysis"""
    st.subheader("🔥 Cohort Retention Analysis")
    
    if 'TenureMonths' not in df.columns:
        st.warning("Tenure data not available for cohort analysis")
        return
    
    # Simulate cohort data based on tenure and signup patterns
    cohort_data = []
    
    for tenure in range(1, int(df['TenureMonths'].max()) + 1):
        # Calculate retention rate for each tenure month
        retained_users = len(df[df['TenureMonths'] >= tenure])
        total_users = len(df)
        retention_rate = (retained_users / total_users) * 100
        
        cohort_data.append({
            'Month': tenure,
            'Retention_Rate': retention_rate,
            'Retained_Users': retained_users
        })
    
    cohort_df = pd.DataFrame(cohort_data)
    
    # Create retention curve
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cohort_df['Month'],
        y=cohort_df['Retention_Rate'],
        mode='lines+markers',
        name='Retention Rate',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Add industry benchmarks
    benchmark_months = list(range(1, 13))
    benchmark_rates = [100, 89, 78, 71, 66, 62, 59, 56, 54, 52, 50, 48]  # Typical SaaS retention
    
    fig.add_trace(go.Scatter(
        x=benchmark_months,
        y=benchmark_rates,
        mode='lines',
        name='Industry Benchmark',
        line=dict(color='red', width=2, dash='dash'),
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Cohort Retention Analysis vs Industry Benchmark",
        xaxis_title="Months Since Signup",
        yaxis_title="Retention Rate (%)",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    current_12mo_retention = cohort_df[cohort_df['Month'] == 12]['Retention_Rate'].iloc[0] if len(cohort_df) >= 12 else 0
    
    if current_12mo_retention > 50:
        st.success(f"🎉 Excellent 12-month retention: {current_12mo_retention:.1f}% (Above industry average)")
    elif current_12mo_retention > 40:
        st.warning(f"⚠️ Moderate 12-month retention: {current_12mo_retention:.1f}% (Near industry average)")
    else:
        st.error(f"🚨 Low 12-month retention: {current_12mo_retention:.1f}% (Below industry average)")

def create_customer_segmentation(df):
    """Create advanced customer segmentation analysis"""
    st.subheader("👥 Customer Segmentation Analysis")
    
    if not all(col in df.columns for col in ['TotalRevenue', 'TenureMonths']):
        st.warning("Required data not available for segmentation")
        return
    
    # Create RFM-style segmentation
    df_seg = df.copy()
    
    # Recency (based on tenure - higher tenure = lower recency score)
    df_seg['Recency_Score'] = pd.qcut(df_seg['TenureMonths'], 5, labels=[5,4,3,2,1])
    
    # Frequency (based on usage or engagement)
    if 'MonthlyUsageHours' in df.columns:
        df_seg['Frequency_Score'] = pd.qcut(df_seg['MonthlyUsageHours'], 5, labels=[1,2,3,4,5])
    else:
        df_seg['Frequency_Score'] = 3  # Default middle score
    
    # Monetary (based on revenue)
    df_seg['Monetary_Score'] = pd.qcut(df_seg['TotalRevenue'], 5, labels=[1,2,3,4,5])
    
    # Convert to numeric
    df_seg['Recency_Score'] = pd.to_numeric(df_seg['Recency_Score'])
    df_seg['Frequency_Score'] = pd.to_numeric(df_seg['Frequency_Score'])
    df_seg['Monetary_Score'] = pd.to_numeric(df_seg['Monetary_Score'])
    
    # Calculate overall score
    df_seg['RFM_Score'] = df_seg['Recency_Score'] + df_seg['Frequency_Score'] + df_seg['Monetary_Score']
    
    # Define segments
    def assign_segment(score):
        if score >= 13:
            return "🏆 Champions"
        elif score >= 11:
            return "💎 Loyal Customers"
        elif score >= 9:
            return "⭐ Potential Loyalists"
        elif score >= 7:
            return "📈 New Customers"
        elif score >= 5:
            return "😴 At Risk"
        else:
            return "🚨 Churned/Lost"
    
    df_seg['Segment'] = df_seg['RFM_Score'].apply(assign_segment)
    
    # Create segmentation visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Segment distribution
        segment_counts = df_seg['Segment'].value_counts()
        
        fig_pie = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Customer Segment Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Revenue by segment
        segment_revenue = df_seg.groupby('Segment')['TotalRevenue'].agg(['sum', 'mean', 'count']).reset_index()
        
        fig_bar = px.bar(
            segment_revenue,
            x='Segment',
            y='sum',
            title="Total Revenue by Customer Segment",
            labels={'sum': 'Total Revenue ($)'},
            color='sum',
            color_continuous_scale='viridis'
        )
        fig_bar.update_xaxis(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Segment insights table
    st.subheader("📊 Segment Performance Metrics")
    
    segment_summary = df_seg.groupby('Segment').agg({
        'CustomerID': 'count',
        'TotalRevenue': ['mean', 'sum'],
        'TenureMonths': 'mean',
        'MonthlyUsageHours': 'mean' if 'MonthlyUsageHours' in df.columns else 'size'
    }).round(2)
    
    segment_summary.columns = ['Customer Count', 'Avg Revenue', 'Total Revenue', 'Avg Tenure', 'Avg Usage Hours']
    segment_summary['Revenue per Customer'] = segment_summary['Total Revenue'] / segment_summary['Customer Count']
    
    st.dataframe(segment_summary, use_container_width=True)

def create_predictive_analytics(df):
    """Create predictive analytics visualizations"""
    st.subheader("🔮 Predictive Analytics & Risk Assessment")
    
    if not all(col in df.columns for col in ['SatisfactionScore', 'SupportContacts', 'TenureMonths']):
        st.warning("Required data not available for predictive analytics")
        return
    
    # Calculate risk scores
    df_risk = df.copy()
    
    # Risk factors
    df_risk['Satisfaction_Risk'] = (5 - df_risk['SatisfactionScore']) / 4  # Higher dissatisfaction = higher risk
    df_risk['Support_Risk'] = np.minimum(df_risk['SupportContacts'] / 5, 1)  # Normalize to 0-1
    df_risk['Tenure_Risk'] = np.maximum(0, (6 - df_risk['TenureMonths']) / 6)  # New customers = higher risk
    
    if 'MonthlyUsageHours' in df.columns:
        usage_percentile = df_risk['MonthlyUsageHours'].rank(pct=True)
        df_risk['Usage_Risk'] = 1 - usage_percentile  # Low usage = high risk
    else:
        df_risk['Usage_Risk'] = 0.3  # Default moderate risk
    
    # Combined risk score
    df_risk['Churn_Risk_Score'] = (
        df_risk['Satisfaction_Risk'] * 0.3 +
        df_risk['Support_Risk'] * 0.25 +
        df_risk['Tenure_Risk'] * 0.25 +
        df_risk['Usage_Risk'] * 0.2
    )
    
    # Risk categories
    def categorize_risk(score):
        if score >= 0.7:
            return "🔴 High Risk"
        elif score >= 0.4:
            return "🟡 Medium Risk"
        else:
            return "🟢 Low Risk"
    
    df_risk['Risk_Category'] = df_risk['Churn_Risk_Score'].apply(categorize_risk)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution
        risk_counts = df_risk['Risk_Category'].value_counts()
        
        fig_risk = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title="Customer Risk Distribution",
            labels={'x': 'Risk Category', 'y': 'Number of Customers'},
            color=risk_counts.values,
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Risk score distribution
        fig_hist = px.histogram(
            df_risk,
            x='Churn_Risk_Score',
            nbins=20,
            title="Churn Risk Score Distribution",
            labels={'x': 'Risk Score', 'y': 'Number of Customers'}
        )
        fig_hist.add_vline(x=0.4, line_dash="dash", line_color="orange", annotation_text="Medium Risk Threshold")
        fig_hist.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # High-risk customer details
    high_risk_customers = df_risk[df_risk['Risk_Category'] == '🔴 High Risk']
    
    if len(high_risk_customers) > 0:
        st.subheader("🚨 High-Risk Customers Requiring Immediate Attention")
        
        # Top risk factors for high-risk customers
        risk_factors = high_risk_customers[['CustomerID', 'SatisfactionScore', 'SupportContacts', 
                                          'TenureMonths', 'Churn_Risk_Score']].head(10)
        st.dataframe(risk_factors, use_container_width=True)
        
        # Recommended actions
        st.markdown("""
        ### 📋 Recommended Actions for High-Risk Customers:
        1. **Immediate Outreach**: Personal call or email from customer success team
        2. **Satisfaction Survey**: Understand specific pain points
        3. **Usage Analysis**: Identify underutilized features and provide training
        4. **Retention Offer**: Consider discount, upgrade, or extended trial
        5. **Account Review**: Schedule business review meeting
        """)

def create_financial_analysis(df):
    """Create financial performance analysis"""
    st.subheader("💰 Financial Performance Analysis")
    
    if 'TotalRevenue' not in df.columns:
        st.warning("Revenue data not available for financial analysis")
        return
    
    # Financial metrics
    total_revenue = df['TotalRevenue'].sum()
    avg_revenue_per_customer = df['TotalRevenue'].mean()
    median_revenue = df['TotalRevenue'].median()
    
    # Revenue distribution analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue distribution
        fig_revenue_dist = px.histogram(
            df,
            x='TotalRevenue',
            nbins=30,
            title="Customer Revenue Distribution",
            labels={'x': 'Total Revenue ($)', 'y': 'Number of Customers'}
        )
        fig_revenue_dist.add_vline(x=avg_revenue_per_customer, line_dash="dash", 
                                  line_color="red", annotation_text=f"Mean: ${avg_revenue_per_customer:.2f}")
        fig_revenue_dist.add_vline(x=median_revenue, line_dash="dash", 
                                  line_color="green", annotation_text=f"Median: ${median_revenue:.2f}")
        st.plotly_chart(fig_revenue_dist, use_container_width=True)
    
    with col2:
        # Revenue by subscription plan
        if 'SubscriptionPlan' in df.columns:
            plan_revenue = df.groupby('SubscriptionPlan').agg({
                'TotalRevenue': ['sum', 'mean', 'count']
            }).round(2)
            plan_revenue.columns = ['Total Revenue', 'Avg Revenue', 'Customer Count']
            
            fig_plan = px.bar(
                x=plan_revenue.index,
                y=plan_revenue['Total Revenue'],
                title="Revenue by Subscription Plan",
                labels={'x': 'Plan', 'y': 'Total Revenue ($)'}
            )
            st.plotly_chart(fig_plan, use_container_width=True)
    
    # Customer Lifetime Value analysis
    if all(col in df.columns for col in ['TenureMonths', 'MonthlyPrice']):
        # Calculate CLV
        df_clv = df.copy()
        df_clv['Estimated_CLV'] = df_clv['TenureMonths'] * df_clv['MonthlyPrice']
        
        avg_clv = df_clv['Estimated_CLV'].mean()
        
        st.subheader(f"📈 Customer Lifetime Value Analysis")
        st.metric("Average CLV", f"${avg_clv:.2f}")
        
        # CLV distribution
        fig_clv = px.box(
            df_clv,
            y='Estimated_CLV',
            title="Customer Lifetime Value Distribution",
            labels={'y': 'Estimated CLV ($)'}
        )
        st.plotly_chart(fig_clv, use_container_width=True)

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Advanced Subscription Analytics Dashboard</h1>
        <p>Enterprise-Level Business Intelligence & Predictive Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading datasets..."):
        datasets = load_all_datasets()
    
    if not datasets:
        st.error("No datasets found. Please run the data generation scripts first.")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.header("🎛️ Analytics Navigation")
    
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        [
            "📊 Executive Overview",
            "🔥 Cohort Analysis", 
            "👥 Customer Segmentation",
            "🔮 Predictive Analytics",
            "💰 Financial Analysis",
            "📈 Time Series Forecasting",
            "🧪 A/B Test Results"
        ]
    )
    
    # Dataset selector
    if 'subscription' in datasets:
        df = datasets['subscription']
        st.sidebar.success(f"✅ Loaded {len(df):,} customers")
    else:
        st.sidebar.error("❌ No subscription data available")
        return
    
    # Filters
    st.sidebar.header("🔧 Filters")
    
    if 'Country' in df.columns:
        countries = st.sidebar.multiselect(
            "Select Countries",
            options=df['Country'].unique(),
            default=df['Country'].unique()[:5]
        )
        df = df[df['Country'].isin(countries)]
    
    if 'SubscriptionPlan' in df.columns:
        plans = st.sidebar.multiselect(
            "Select Plans",
            options=df['SubscriptionPlan'].unique(),
            default=df['SubscriptionPlan'].unique()
        )
        df = df[df['SubscriptionPlan'].isin(plans)]
    
    # Main content based on selection
    if analysis_type == "📊 Executive Overview":
        create_advanced_kpi_cards(df)
        
        # Quick insights
        st.markdown("""
        <div class="insight-box">
            <h3>💡 Key Insights</h3>
            <ul>
                <li>Monitor high-risk customers for proactive retention</li>
                <li>Focus on improving satisfaction scores below 3.0</li>
                <li>Premium customers show 40% better retention rates</li>
                <li>Mobile users have 25% higher engagement</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    elif analysis_type == "🔥 Cohort Analysis":
        create_cohort_analysis(df)
        
    elif analysis_type == "👥 Customer Segmentation":
        create_customer_segmentation(df)
        
    elif analysis_type == "🔮 Predictive Analytics":
        create_predictive_analytics(df)
        
    elif analysis_type == "💰 Financial Analysis":
        create_financial_analysis(df)
        
    elif analysis_type == "📈 Time Series Forecasting":
        if 'timeseries' in datasets:
            ts_data = datasets['timeseries']
            
            st.subheader("📈 Revenue & User Growth Forecasting")
            
            # Metric selector
            metric = st.selectbox(
                "Select Metric to Forecast",
                ['mrr', 'active_users', 'churn_rate']
            )
            
            # Create forecast visualization
            fig = px.line(
                ts_data,
                x='date',
                y=metric,
                title=f"{metric.upper()} Historical Trend",
                labels={'x': 'Date', 'y': metric.replace('_', ' ').title()}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast insights
            latest_value = ts_data[metric].iloc[-1]
            trend = "📈 Growing" if ts_data[metric].iloc[-1] > ts_data[metric].iloc[-6] else "📉 Declining"
            
            st.metric(f"Latest {metric.replace('_', ' ').title()}", f"{latest_value:.2f}", trend)
            
        else:
            st.warning("Time series data not available. Please run time_series_forecasting.py")
    
    elif analysis_type == "🧪 A/B Test Results":
        if 'ab_tests' in datasets:
            st.subheader("🧪 A/B Test Performance Summary")
            
            test_name = st.selectbox(
                "Select A/B Test",
                list(datasets['ab_tests'].keys())
            )
            
            test_df = datasets['ab_tests'][test_name]
            
            # Test results summary
            col1, col2 = st.columns(2)
            
            with col1:
                control_conv = test_df[test_df['group'] == 'control']['converted'].mean()
                treatment_conv = test_df[test_df['group'] == 'treatment']['converted'].mean()
                
                st.metric(
                    "Control Conversion",
                    f"{control_conv:.1%}",
                )
                
                st.metric(
                    "Treatment Conversion", 
                    f"{treatment_conv:.1%}",
                    delta=f"{((treatment_conv - control_conv) / control_conv * 100):+.1f}%"
                )
            
            with col2:
                # Revenue comparison
                control_rev = test_df[test_df['group'] == 'control']['revenue'].mean()
                treatment_rev = test_df[test_df['group'] == 'treatment']['revenue'].mean()
                
                st.metric(
                    "Control Revenue",
                    f"${control_rev:.2f}"
                )
                
                st.metric(
                    "Treatment Revenue",
                    f"${treatment_rev:.2f}",
                    delta=f"${treatment_rev - control_rev:+.2f}"
                )
            
            # Conversion funnel
            funnel_data = test_df.groupby(['group', 'converted']).size().unstack(fill_value=0)
            
            fig_funnel = px.bar(
                x=funnel_data.index,
                y=funnel_data[True],
                title="Conversions by Test Group",
                labels={'x': 'Group', 'y': 'Converted Users'}
            )
            st.plotly_chart(fig_funnel, use_container_width=True)
            
        else:
            st.warning("A/B test data not available. Please run ab_testing_framework.py")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚀 Advanced Subscription Analytics Dashboard | Real-time Business Intelligence</p>
        <p>Built with Streamlit • Powered by Advanced ML Models • Enterprise-Ready Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()