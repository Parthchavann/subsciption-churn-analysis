"""
Automated Reporting System for Subscription Analytics
Generates executive reports, alerts, and scheduled insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
from jinja2 import Template
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings('ignore')

class AutomatedReporter:
    def __init__(self):
        self.reports = {}
        self.alerts = []
        self.metrics_cache = {}
        
    def load_latest_data(self):
        """Load the most recent data for reporting"""
        print("📊 Loading latest data for automated reporting...")
        
        try:
            # Subscription data
            self.subscription_data = pd.read_csv('../data/real/subscription_churn_real_patterns.csv')
            
            # Time series data if available
            try:
                self.timeseries_data = pd.read_csv('../data/real/time_series_data.csv')
                self.timeseries_data['date'] = pd.to_datetime(self.timeseries_data['date'])
            except:
                self.timeseries_data = None
            
            print(f"✅ Loaded {len(self.subscription_data):,} customer records")
            return True
            
        except FileNotFoundError:
            print("❌ Data files not found. Please run data generation scripts first.")
            return False
    
    def calculate_key_metrics(self):
        """Calculate key business metrics for reporting"""
        print("📈 Calculating key business metrics...")
        
        df = self.subscription_data
        
        # Basic metrics
        total_customers = len(df)
        active_customers = len(df[df['Churn'] == 'No'])
        churned_customers = len(df[df['Churn'] == 'Yes'])
        churn_rate = (churned_customers / total_customers) * 100
        
        # Revenue metrics
        total_revenue = df['TotalRevenue'].sum()
        avg_revenue_per_customer = df['TotalRevenue'].mean()
        monthly_revenue = df['MonthlyPrice'].sum()  # Approximate MRR
        
        # Advanced metrics
        avg_tenure = df['TenureMonths'].mean()
        avg_satisfaction = df['SatisfactionScore'].mean()
        high_value_customers = len(df[df['MonthlyPrice'] > df['MonthlyPrice'].quantile(0.8)])
        
        # Risk assessment
        at_risk_customers = len(df[
            (df['SatisfactionScore'] < 3) | 
            (df['SupportContacts'] > 2) |
            (df['TenureMonths'] < 3)
        ])
        
        # Plan distribution
        plan_distribution = df['SubscriptionPlan'].value_counts().to_dict()
        
        # Geographic distribution
        country_distribution = df['Country'].value_counts().head(5).to_dict()
        
        self.metrics_cache = {
            'calculation_date': datetime.now().isoformat(),
            'customer_metrics': {
                'total_customers': total_customers,
                'active_customers': active_customers,
                'churned_customers': churned_customers,
                'churn_rate': round(churn_rate, 2),
                'at_risk_customers': at_risk_customers,
                'high_value_customers': high_value_customers
            },
            'revenue_metrics': {
                'total_revenue': round(total_revenue, 2),
                'monthly_revenue': round(monthly_revenue, 2),
                'avg_revenue_per_customer': round(avg_revenue_per_customer, 2),
                'revenue_per_active_customer': round(total_revenue / active_customers, 2) if active_customers > 0 else 0
            },
            'engagement_metrics': {
                'avg_tenure_months': round(avg_tenure, 2),
                'avg_satisfaction_score': round(avg_satisfaction, 2),
                'avg_monthly_usage': round(df['MonthlyUsageHours'].mean(), 2),
                'avg_feature_usage': round(df['FeatureUsageRate'].mean(), 3)
            },
            'segmentation': {
                'plan_distribution': plan_distribution,
                'country_distribution': country_distribution
            }
        }
        
        print("✅ Metrics calculation complete")
        return self.metrics_cache
    
    def generate_alert_conditions(self):
        """Check for alert conditions and generate alerts"""
        print("🚨 Checking for alert conditions...")
        
        metrics = self.metrics_cache
        alerts = []
        
        # Churn rate alerts
        churn_rate = metrics['customer_metrics']['churn_rate']
        if churn_rate > 20:
            alerts.append({
                'type': 'CRITICAL',
                'category': 'Churn',
                'message': f"Critical churn rate: {churn_rate:.1f}% (Threshold: 20%)",
                'recommended_action': "Immediate retention campaign required",
                'priority': 1
            })
        elif churn_rate > 15:
            alerts.append({
                'type': 'WARNING',
                'category': 'Churn',
                'message': f"High churn rate: {churn_rate:.1f}% (Threshold: 15%)",
                'recommended_action': "Review retention strategies",
                'priority': 2
            })
        
        # At-risk customers
        at_risk_count = metrics['customer_metrics']['at_risk_customers']
        total_customers = metrics['customer_metrics']['total_customers']
        at_risk_percentage = (at_risk_count / total_customers) * 100
        
        if at_risk_percentage > 25:
            alerts.append({
                'type': 'WARNING',
                'category': 'Customer Risk',
                'message': f"{at_risk_count:,} customers at risk ({at_risk_percentage:.1f}%)",
                'recommended_action': "Proactive customer outreach needed",
                'priority': 2
            })
        
        # Satisfaction score alerts
        avg_satisfaction = metrics['engagement_metrics']['avg_satisfaction_score']
        if avg_satisfaction < 3.5:
            alerts.append({
                'type': 'WARNING',
                'category': 'Customer Satisfaction',
                'message': f"Low satisfaction score: {avg_satisfaction:.2f}/5.0",
                'recommended_action': "Customer feedback analysis required",
                'priority': 2
            })
        
        # Revenue decline (if time series data available)
        if self.timeseries_data is not None and len(self.timeseries_data) > 6:
            recent_revenue = self.timeseries_data['mrr'].tail(3).mean()
            previous_revenue = self.timeseries_data['mrr'].iloc[-6:-3].mean()
            revenue_change = ((recent_revenue - previous_revenue) / previous_revenue) * 100
            
            if revenue_change < -5:
                alerts.append({
                    'type': 'CRITICAL',
                    'category': 'Revenue',
                    'message': f"Revenue declining: {revenue_change:.1f}% over last 3 months",
                    'recommended_action': "Revenue recovery plan needed",
                    'priority': 1
                })
        
        self.alerts = sorted(alerts, key=lambda x: x['priority'])
        print(f"🔔 Generated {len(alerts)} alerts")
        return alerts
    
    def create_executive_summary_html(self):
        """Create HTML executive summary report"""
        print("📄 Generating executive summary report...")
        
        metrics = self.metrics_cache
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Subscription Analytics - Executive Summary</title>
            <style>
                body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; color: #333; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 20px; text-align: center; border-radius: 10px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                               gap: 15px; margin: 20px 0; }
                .metric-card { background: #f8f9fa; border: 1px solid #dee2e6; padding: 15px; 
                              border-radius: 8px; text-align: center; }
                .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
                .metric-label { font-size: 0.9em; color: #6c757d; margin-top: 5px; }
                .alert-critical { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; 
                                 padding: 10px; margin: 10px 0; border-radius: 5px; }
                .alert-warning { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; 
                                padding: 10px; margin: 10px 0; border-radius: 5px; }
                .insight-box { background: #e7f3ff; border-left: 4px solid #007bff; 
                              padding: 15px; margin: 15px 0; }
                .recommendation { background: #f0f9ff; border: 1px solid #bee5eb; 
                                 padding: 15px; margin: 10px 0; border-radius: 5px; }
                table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Subscription Analytics Executive Summary</h1>
                <p>Generated on {{ report_date }}</p>
            </div>
            
            <h2>🎯 Key Performance Indicators</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{{ "{:,}".format(metrics.customer_metrics.total_customers) }}</div>
                    <div class="metric-label">Total Customers</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "{:.1f}%".format(metrics.customer_metrics.churn_rate) }}</div>
                    <div class="metric-label">Churn Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${{ "{:,.0f}".format(metrics.revenue_metrics.monthly_revenue) }}</div>
                    <div class="metric-label">Monthly Revenue</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${{ "{:.2f}".format(metrics.revenue_metrics.avg_revenue_per_customer) }}</div>
                    <div class="metric-label">Avg Revenue per Customer</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "{:.1f}".format(metrics.engagement_metrics.avg_satisfaction_score) }}/5</div>
                    <div class="metric-label">Avg Satisfaction</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "{:,}".format(metrics.customer_metrics.at_risk_customers) }}</div>
                    <div class="metric-label">At-Risk Customers</div>
                </div>
            </div>
            
            {% if alerts %}
            <h2>🚨 Critical Alerts</h2>
            {% for alert in alerts %}
                <div class="alert-{{ alert.type.lower() }}">
                    <strong>{{ alert.type }}: {{ alert.category }}</strong><br>
                    {{ alert.message }}<br>
                    <em>Recommended Action: {{ alert.recommended_action }}</em>
                </div>
            {% endfor %}
            {% endif %}
            
            <h2>💡 Key Insights</h2>
            <div class="insight-box">
                <h3>Customer Health</h3>
                <ul>
                    <li>{{ "{:.1f}%".format((metrics.customer_metrics.at_risk_customers / metrics.customer_metrics.total_customers) * 100) }} of customers are at risk of churning</li>
                    <li>Average customer tenure: {{ "{:.1f}".format(metrics.engagement_metrics.avg_tenure_months) }} months</li>
                    <li>{{ "{:,}".format(metrics.customer_metrics.high_value_customers) }} high-value customers (top 20% by revenue)</li>
                </ul>
            </div>
            
            <div class="insight-box">
                <h3>Revenue Performance</h3>
                <ul>
                    <li>Total revenue: ${{ "{:,.2f}".format(metrics.revenue_metrics.total_revenue) }}</li>
                    <li>Revenue per active customer: ${{ "{:.2f}".format(metrics.revenue_metrics.revenue_per_active_customer) }}</li>
                    <li>Monthly recurring revenue: ${{ "{:,.2f}".format(metrics.revenue_metrics.monthly_revenue) }}</li>
                </ul>
            </div>
            
            <h2>📋 Recommendations</h2>
            <div class="recommendation">
                <h3>Immediate Actions (Next 30 Days)</h3>
                <ul>
                    <li>Launch proactive retention campaign for {{ "{:,}".format(metrics.customer_metrics.at_risk_customers) }} at-risk customers</li>
                    <li>Implement customer satisfaction improvement program</li>
                    <li>Review pricing strategy for basic plan customers</li>
                    <li>Enhance onboarding process to improve early-stage retention</li>
                </ul>
            </div>
            
            <div class="recommendation">
                <h3>Strategic Initiatives (Next 90 Days)</h3>
                <ul>
                    <li>Develop customer success automation workflows</li>
                    <li>Implement predictive churn scoring model</li>
                    <li>Launch feature adoption campaigns</li>
                    <li>Establish customer health score monitoring</li>
                </ul>
            </div>
            
            <h2>📊 Subscription Plan Performance</h2>
            <table>
                <thead>
                    <tr><th>Plan</th><th>Customers</th><th>Percentage</th></tr>
                </thead>
                <tbody>
                    {% for plan, count in metrics.segmentation.plan_distribution.items() %}
                    <tr>
                        <td>{{ plan }}</td>
                        <td>{{ "{:,}".format(count) }}</td>
                        <td>{{ "{:.1f}%".format((count / metrics.customer_metrics.total_customers) * 100) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            
            <div style="margin-top: 30px; text-align: center; color: #666;">
                <p><em>This report was automatically generated by the Subscription Analytics System</em></p>
                <p>Next report scheduled: {{ next_report_date }}</p>
            </div>
        </body>
        </html>
        """
        
        template = Template(html_template)
        
        html_content = template.render(
            metrics=type('obj', (object,), metrics),  # Convert dict to object for dot notation
            alerts=self.alerts,
            report_date=datetime.now().strftime("%B %d, %Y at %I:%M %p"),
            next_report_date=(datetime.now() + timedelta(days=7)).strftime("%B %d, %Y")
        )
        
        # Save report
        report_filename = f"../presentation/executive_report_{datetime.now().strftime('%Y%m%d')}.html"
        with open(report_filename, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Executive report saved: {report_filename}")
        return html_content, report_filename
    
    def create_operational_dashboard_data(self):
        """Create data for operational dashboard"""
        print("📊 Generating operational dashboard data...")
        
        df = self.subscription_data
        
        # Daily/Weekly operational metrics
        operational_data = {
            'date_generated': datetime.now().isoformat(),
            'customer_health': {
                'new_signups_last_30_days': len(df[df['TenureMonths'] <= 1]),  # Approximate new customers
                'cancellations_last_30_days': len(df[df['Churn'] == 'Yes']),
                'payment_failures': len(df[df['PaymentFailures'] > 0]),
                'support_tickets_high_priority': len(df[df['SupportContacts'] > 3])
            },
            'revenue_health': {
                'mrr_trend': 'stable',  # Would calculate from time series data
                'revenue_at_risk': df[df['Churn'] == 'Yes']['TotalRevenue'].sum(),
                'expansion_revenue': df[df['MonthlyPrice'] > 20]['TotalRevenue'].sum(),
                'contraction_risk': len(df[(df['SatisfactionScore'] < 3) & (df['MonthlyPrice'] > 15)])
            },
            'engagement_health': {
                'low_engagement_users': len(df[df['MonthlyUsageHours'] < 5]),
                'feature_adoption_rate': df['FeatureUsageRate'].mean(),
                'support_satisfaction': df[df['SupportContacts'] > 0]['SatisfactionScore'].mean(),
                'product_sentiment': 'positive' if df['SatisfactionScore'].mean() > 4 else 'mixed'
            }
        }
        
        # Save operational data
        with open('../data/operational_metrics.json', 'w') as f:
            json.dump(operational_data, f, indent=4, default=str)
        
        return operational_data
    
    def schedule_report_generation(self):
        """Set up automated report scheduling"""
        print("⏰ Setting up automated report scheduling...")
        
        schedule_config = {
            'executive_summary': {
                'frequency': 'weekly',
                'day': 'monday',
                'time': '09:00',
                'recipients': ['ceo@company.com', 'cfo@company.com', 'head-of-growth@company.com']
            },
            'operational_dashboard': {
                'frequency': 'daily',
                'time': '08:00',
                'recipients': ['operations@company.com', 'customer-success@company.com']
            },
            'alerts': {
                'frequency': 'real-time',
                'threshold_checks': ['churn_rate', 'revenue_decline', 'satisfaction_drop'],
                'recipients': ['on-call@company.com']
            }
        }
        
        # Save scheduling configuration
        with open('../data/report_schedule.json', 'w') as f:
            json.dump(schedule_config, f, indent=4)
        
        print("✅ Report scheduling configured")
        return schedule_config
    
    def export_insights_to_json(self):
        """Export all insights to structured JSON for API consumption"""
        print("📤 Exporting insights to JSON...")
        
        insights_export = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'data_as_of': datetime.now().isoformat(),
                'report_version': '2.0',
                'data_sources': ['subscription_data', 'timeseries_data']
            },
            'executive_summary': {
                'key_metrics': self.metrics_cache,
                'alerts': self.alerts,
                'recommendations': [
                    "Implement predictive churn model",
                    "Launch retention campaign for at-risk customers",
                    "Improve customer onboarding experience",
                    "Develop customer success automation"
                ]
            },
            'detailed_analysis': {
                'customer_segments': self._analyze_customer_segments(),
                'revenue_analysis': self._analyze_revenue_patterns(),
                'churn_analysis': self._analyze_churn_patterns()
            }
        }
        
        # Export to JSON
        export_filename = f"../data/insights_export_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(export_filename, 'w') as f:
            json.dump(insights_export, f, indent=4, default=str)
        
        print(f"✅ Insights exported: {export_filename}")
        return insights_export
    
    def _analyze_customer_segments(self):
        """Analyze customer segments for detailed reporting"""
        df = self.subscription_data
        
        # Create segments based on revenue and tenure
        segments = {}
        
        # High-value customers
        high_value = df[df['MonthlyPrice'] > df['MonthlyPrice'].quantile(0.8)]
        segments['high_value'] = {
            'count': len(high_value),
            'avg_tenure': high_value['TenureMonths'].mean(),
            'churn_rate': (high_value['Churn'] == 'Yes').mean() * 100,
            'satisfaction': high_value['SatisfactionScore'].mean()
        }
        
        # New customers (tenure < 3 months)
        new_customers = df[df['TenureMonths'] < 3]
        segments['new_customers'] = {
            'count': len(new_customers),
            'churn_rate': (new_customers['Churn'] == 'Yes').mean() * 100,
            'satisfaction': new_customers['SatisfactionScore'].mean()
        }
        
        # Loyal customers (tenure > 12 months)
        loyal_customers = df[df['TenureMonths'] > 12]
        segments['loyal_customers'] = {
            'count': len(loyal_customers),
            'churn_rate': (loyal_customers['Churn'] == 'Yes').mean() * 100,
            'avg_revenue': loyal_customers['TotalRevenue'].mean()
        }
        
        return segments
    
    def _analyze_revenue_patterns(self):
        """Analyze revenue patterns for reporting"""
        df = self.subscription_data
        
        revenue_analysis = {
            'total_revenue': df['TotalRevenue'].sum(),
            'revenue_per_plan': df.groupby('SubscriptionPlan')['TotalRevenue'].sum().to_dict(),
            'revenue_concentration': {
                'top_20_percent_customers': df.nlargest(int(len(df) * 0.2), 'TotalRevenue')['TotalRevenue'].sum(),
                'bottom_80_percent_customers': df.nsmallest(int(len(df) * 0.8), 'TotalRevenue')['TotalRevenue'].sum()
            }
        }
        
        return revenue_analysis
    
    def _analyze_churn_patterns(self):
        """Analyze churn patterns for reporting"""
        df = self.subscription_data
        
        churned_customers = df[df['Churn'] == 'Yes']
        
        churn_analysis = {
            'overall_churn_rate': (len(churned_customers) / len(df)) * 100,
            'churn_by_plan': df.groupby('SubscriptionPlan')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).to_dict(),
            'churn_by_tenure': {
                'early_stage': ((df['TenureMonths'] < 3) & (df['Churn'] == 'Yes')).sum(),
                'mid_stage': ((df['TenureMonths'].between(3, 12)) & (df['Churn'] == 'Yes')).sum(),
                'mature': ((df['TenureMonths'] > 12) & (df['Churn'] == 'Yes')).sum()
            },
            'churn_drivers': {
                'low_satisfaction': len(churned_customers[churned_customers['SatisfactionScore'] < 3]),
                'high_support_contacts': len(churned_customers[churned_customers['SupportContacts'] > 2]),
                'payment_issues': len(churned_customers[churned_customers['PaymentFailures'] > 0])
            }
        }
        
        return churn_analysis
    
    def run_automated_reporting(self):
        """Run the complete automated reporting pipeline"""
        print("🚀 AUTOMATED REPORTING SYSTEM")
        print("=" * 60)
        
        # Load data
        if not self.load_latest_data():
            return
        
        # Calculate metrics
        self.calculate_key_metrics()
        
        # Generate alerts
        self.generate_alert_conditions()
        
        # Create reports
        html_report, report_file = self.create_executive_summary_html()
        operational_data = self.create_operational_dashboard_data()
        insights_export = self.export_insights_to_json()
        schedule_config = self.schedule_report_generation()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 AUTOMATED REPORTING COMPLETE")
        print("=" * 60)
        
        print(f"\n📋 Reports Generated:")
        print(f"  📄 Executive Summary: {report_file}")
        print(f"  📊 Operational Data: ../data/operational_metrics.json")
        print(f"  📤 Insights Export: Multiple JSON files")
        
        print(f"\n🚨 Alerts Generated: {len(self.alerts)}")
        for alert in self.alerts[:3]:  # Show top 3 alerts
            print(f"  {alert['type']}: {alert['message']}")
        
        print(f"\n📈 Key Metrics:")
        print(f"  Total Customers: {self.metrics_cache['customer_metrics']['total_customers']:,}")
        print(f"  Churn Rate: {self.metrics_cache['customer_metrics']['churn_rate']:.1f}%")
        print(f"  Monthly Revenue: ${self.metrics_cache['revenue_metrics']['monthly_revenue']:,.2f}")
        print(f"  At-Risk Customers: {self.metrics_cache['customer_metrics']['at_risk_customers']:,}")
        
        print("\n✨ Automated Reporting System Ready!")
        print("🔄 Reports can be scheduled for daily/weekly/monthly generation")
        
        return {
            'reports': {
                'executive_summary': report_file,
                'operational_data': operational_data,
                'insights_export': insights_export
            },
            'alerts': self.alerts,
            'metrics': self.metrics_cache
        }

if __name__ == "__main__":
    reporter = AutomatedReporter()
    results = reporter.run_automated_reporting()