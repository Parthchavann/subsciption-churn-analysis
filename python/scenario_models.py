"""
What-If Scenario Models for Subscription Business
Advanced simulation for churn reduction and revenue optimization strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import json
from datetime import datetime, timedelta

class SubscriptionScenarioModel:
    def __init__(self):
        self.baseline_metrics = {}
        self.scenario_results = {}
        
    def load_baseline_data(self):
        """Load baseline subscription data"""
        try:
            self.users = pd.read_csv('../data/raw/users.csv')
            self.subscriptions = pd.read_csv('../data/raw/subscriptions.csv')
            self.engagement = pd.read_csv('../data/raw/engagement.csv')
            self.tickets = pd.read_csv('../data/raw/support_tickets.csv')
            
            # Convert dates
            self.users['signup_date'] = pd.to_datetime(self.users['signup_date'])
            self.subscriptions['billing_date'] = pd.to_datetime(self.subscriptions['billing_date'])
            
            print("Baseline data loaded successfully!")
            return True
        except FileNotFoundError:
            print("Error: Data files not found. Run data_generation.py first.")
            return False
    
    def calculate_baseline_metrics(self):
        """Calculate current business metrics"""
        
        # User metrics
        total_users = len(self.users)
        active_users = len(self.subscriptions[self.subscriptions['status'] == 'active']['user_id'].unique())
        churned_users = len(self.subscriptions[self.subscriptions['status'] == 'churned']['user_id'].unique())
        churn_rate = (churned_users / total_users) * 100
        
        # Revenue metrics
        monthly_revenue = self.subscriptions[self.subscriptions['status'] == 'active']['price'].sum()
        total_revenue = self.subscriptions['price'].sum()
        arpu = monthly_revenue / active_users if active_users > 0 else 0
        
        # Customer Lifetime Value (simplified)
        avg_customer_lifetime = self.subscriptions.groupby('user_id').size().mean()
        customer_ltv = arpu * avg_customer_lifetime
        
        # Engagement metrics
        avg_engagement = self.engagement['engagement_score'].mean()
        
        self.baseline_metrics = {
            'total_users': total_users,
            'active_users': active_users,
            'churned_users': churned_users,
            'churn_rate': churn_rate,
            'monthly_revenue': monthly_revenue,
            'total_revenue': total_revenue,
            'arpu': arpu,
            'customer_ltv': customer_ltv,
            'avg_engagement': avg_engagement
        }
        
        return self.baseline_metrics
    
    def simulate_churn_reduction(self, reduction_percentage, intervention_cost_per_user, 
                                target_segments=None, months_horizon=12):
        """
        Simulate the impact of churn reduction initiatives
        
        Args:
            reduction_percentage: Target churn reduction (e.g., 10 for 10%)
            intervention_cost_per_user: Cost of intervention per user
            target_segments: Specific user segments to target
            months_horizon: Projection horizon in months
        """
        
        current_churn_rate = self.baseline_metrics['churn_rate']
        active_users = self.baseline_metrics['active_users']
        monthly_revenue = self.baseline_metrics['monthly_revenue']
        arpu = self.baseline_metrics['arpu']
        
        # Calculate intervention parameters
        new_churn_rate = current_churn_rate * (1 - reduction_percentage / 100)
        monthly_churn_reduction = (current_churn_rate - new_churn_rate) / 100
        users_saved_monthly = int(active_users * monthly_churn_reduction)
        
        # Cost calculations
        total_users_to_target = int(active_users * 0.3)  # Target 30% of users at risk
        total_intervention_cost = total_users_to_target * intervention_cost_per_user
        monthly_intervention_cost = total_intervention_cost / 12  # Spread over year
        
        # Revenue impact over time
        results = []
        cumulative_users_saved = 0
        cumulative_revenue_impact = 0
        cumulative_costs = 0
        
        for month in range(1, months_horizon + 1):
            # Monthly users saved
            monthly_saved = users_saved_monthly
            cumulative_users_saved += monthly_saved
            
            # Revenue from saved users (they continue paying)
            monthly_revenue_impact = monthly_saved * arpu
            cumulative_revenue_impact += monthly_revenue_impact
            
            # Monthly costs
            cumulative_costs += monthly_intervention_cost
            
            # Net impact
            net_monthly_impact = monthly_revenue_impact - monthly_intervention_cost
            net_cumulative_impact = cumulative_revenue_impact - cumulative_costs
            
            results.append({
                'month': month,
                'users_saved_monthly': monthly_saved,
                'cumulative_users_saved': cumulative_users_saved,
                'monthly_revenue_impact': monthly_revenue_impact,
                'cumulative_revenue_impact': cumulative_revenue_impact,
                'monthly_costs': monthly_intervention_cost,
                'cumulative_costs': cumulative_costs,
                'net_monthly_impact': net_monthly_impact,
                'net_cumulative_impact': net_cumulative_impact,
                'roi': (net_cumulative_impact / cumulative_costs * 100) if cumulative_costs > 0 else 0
            })
        
        scenario_summary = {
            'intervention_type': 'Churn Reduction',
            'parameters': {
                'reduction_target': reduction_percentage,
                'cost_per_user': intervention_cost_per_user,
                'users_targeted': total_users_to_target
            },
            'total_impact': {
                'users_saved': cumulative_users_saved,
                'revenue_gained': cumulative_revenue_impact,
                'total_costs': cumulative_costs,
                'net_impact': net_cumulative_impact,
                'roi_percentage': results[-1]['roi']
            },
            'timeline': results
        }
        
        return scenario_summary
    
    def simulate_price_optimization(self, price_change_percentage, demand_elasticity=-0.5,
                                   plan_segments=None, months_horizon=12):
        """
        Simulate impact of pricing changes on revenue and user base
        
        Args:
            price_change_percentage: Price change (e.g., 5 for 5% increase)
            demand_elasticity: Price elasticity of demand (negative number)
            plan_segments: Specific plans to apply changes to
            months_horizon: Projection horizon
        """
        
        active_users = self.baseline_metrics['active_users']
        monthly_revenue = self.baseline_metrics['monthly_revenue']
        arpu = self.baseline_metrics['arpu']
        
        # Calculate demand response
        demand_change_percentage = demand_elasticity * price_change_percentage
        user_change = int(active_users * (demand_change_percentage / 100))
        new_user_count = active_users + user_change
        
        # New pricing
        new_arpu = arpu * (1 + price_change_percentage / 100)
        new_monthly_revenue = new_user_count * new_arpu
        
        # Revenue impact
        monthly_revenue_impact = new_monthly_revenue - monthly_revenue
        
        results = []
        cumulative_revenue_impact = 0
        
        for month in range(1, months_horizon + 1):
            cumulative_revenue_impact += monthly_revenue_impact
            
            # Additional churn from price sensitivity (first 6 months)
            if month <= 6:
                additional_churn_rate = abs(price_change_percentage) * 0.1  # 0.1% additional churn per 1% price increase
                additional_churned_users = int(active_users * (additional_churn_rate / 100))
            else:
                additional_churned_users = 0
            
            results.append({
                'month': month,
                'new_user_count': new_user_count - (additional_churned_users * min(month, 6)),
                'new_monthly_revenue': new_monthly_revenue - (additional_churned_users * new_arpu),
                'monthly_revenue_impact': monthly_revenue_impact,
                'cumulative_revenue_impact': cumulative_revenue_impact,
                'additional_churn': additional_churned_users if month <= 6 else 0
            })
        
        scenario_summary = {
            'intervention_type': 'Price Optimization',
            'parameters': {
                'price_change': price_change_percentage,
                'demand_elasticity': demand_elasticity,
                'affected_users': active_users
            },
            'total_impact': {
                'user_change': user_change,
                'new_arpu': new_arpu,
                'monthly_revenue_change': monthly_revenue_impact,
                'annual_revenue_impact': cumulative_revenue_impact
            },
            'timeline': results
        }
        
        return scenario_summary
    
    def simulate_engagement_improvement(self, engagement_lift_percentage, conversion_rate_improvement,
                                       implementation_cost, months_horizon=12):
        """
        Simulate impact of improving user engagement on retention and upselling
        """
        
        active_users = self.baseline_metrics['active_users']
        arpu = self.baseline_metrics['arpu']
        current_churn_rate = self.baseline_metrics['churn_rate']
        
        # Engagement-driven improvements
        # Higher engagement typically leads to lower churn
        churn_improvement_rate = engagement_lift_percentage * 0.3  # 30% of engagement improvement translates to churn reduction
        new_churn_rate = current_churn_rate * (1 - churn_improvement_rate / 100)
        
        # Users retained due to better engagement
        users_retained_monthly = int(active_users * (current_churn_rate - new_churn_rate) / 100)
        
        # Upselling opportunity from higher engagement
        upsell_users = int(active_users * (conversion_rate_improvement / 100))
        upsell_revenue_per_user = arpu * 0.5  # Average 50% revenue increase from upsell
        
        results = []
        cumulative_retention_revenue = 0
        cumulative_upsell_revenue = 0
        cumulative_costs = implementation_cost
        
        monthly_implementation_cost = implementation_cost / months_horizon
        
        for month in range(1, months_horizon + 1):
            # Retention revenue
            monthly_retention_revenue = users_retained_monthly * arpu
            cumulative_retention_revenue += monthly_retention_revenue
            
            # Upsell revenue (gradual rollout)
            if month >= 3:  # Upselling starts after 3 months
                monthly_upsell_revenue = (upsell_users / 9) * upsell_revenue_per_user  # Spread over 9 months
                cumulative_upsell_revenue += monthly_upsell_revenue
            else:
                monthly_upsell_revenue = 0
            
            # Costs
            cumulative_costs += monthly_implementation_cost
            
            # Total impact
            total_monthly_revenue = monthly_retention_revenue + monthly_upsell_revenue
            total_cumulative_revenue = cumulative_retention_revenue + cumulative_upsell_revenue
            net_impact = total_cumulative_revenue - cumulative_costs
            
            results.append({
                'month': month,
                'retention_revenue': monthly_retention_revenue,
                'upsell_revenue': monthly_upsell_revenue,
                'total_monthly_revenue': total_monthly_revenue,
                'cumulative_revenue': total_cumulative_revenue,
                'cumulative_costs': cumulative_costs,
                'net_impact': net_impact,
                'roi': (net_impact / cumulative_costs * 100) if cumulative_costs > 0 else 0
            })
        
        scenario_summary = {
            'intervention_type': 'Engagement Improvement',
            'parameters': {
                'engagement_lift': engagement_lift_percentage,
                'conversion_improvement': conversion_rate_improvement,
                'implementation_cost': implementation_cost
            },
            'total_impact': {
                'users_retained': users_retained_monthly * months_horizon,
                'users_upsold': upsell_users,
                'retention_revenue': cumulative_retention_revenue,
                'upsell_revenue': cumulative_upsell_revenue,
                'total_costs': cumulative_costs,
                'net_impact': results[-1]['net_impact'],
                'roi_percentage': results[-1]['roi']
            },
            'timeline': results
        }
        
        return scenario_summary
    
    def compare_scenarios(self, scenarios):
        """Compare multiple scenarios side by side"""
        
        comparison_data = []
        
        for scenario_name, scenario_data in scenarios.items():
            total_impact = scenario_data['total_impact']
            
            comparison_data.append({
                'scenario': scenario_name,
                'type': scenario_data['intervention_type'],
                'net_impact': total_impact.get('net_impact', 0),
                'roi': total_impact.get('roi_percentage', 0),
                'revenue_impact': total_impact.get('revenue_gained', 
                                total_impact.get('annual_revenue_impact', 
                                total_impact.get('retention_revenue', 0) + total_impact.get('upsell_revenue', 0))),
                'cost': total_impact.get('total_costs', total_impact.get('implementation_cost', 0))
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        return comparison_df
    
    def visualize_scenario_comparison(self, scenarios):
        """Create visualization comparing different scenarios"""
        
        comparison_df = self.compare_scenarios(scenarios)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Net Impact Comparison', 'ROI Comparison',
                          'Revenue vs Cost', 'Timeline Comparison'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Net Impact
        fig.add_trace(
            go.Bar(x=comparison_df['scenario'], y=comparison_df['net_impact'],
                   name='Net Impact', marker_color='lightblue'),
            row=1, col=1
        )
        
        # ROI
        fig.add_trace(
            go.Bar(x=comparison_df['scenario'], y=comparison_df['roi'],
                   name='ROI %', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Revenue vs Cost scatter
        fig.add_trace(
            go.Scatter(x=comparison_df['cost'], y=comparison_df['revenue_impact'],
                      mode='markers+text', text=comparison_df['scenario'],
                      textposition='top center', name='Revenue vs Cost',
                      marker=dict(size=12, color='orange')),
            row=2, col=1
        )
        
        # Timeline comparison (if timeline data available)
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (scenario_name, scenario_data) in enumerate(scenarios.items()):
            if 'timeline' in scenario_data:
                timeline_df = pd.DataFrame(scenario_data['timeline'])
                fig.add_trace(
                    go.Scatter(x=timeline_df['month'], y=timeline_df.get('net_cumulative_impact', 
                                                                        timeline_df.get('cumulative_revenue_impact', 0)),
                              mode='lines', name=scenario_name,
                              line=dict(color=colors[i % len(colors)])),
                    row=2, col=2
                )
        
        fig.update_layout(height=800, title_text="Scenario Analysis Dashboard")
        return fig
    
    def run_comprehensive_analysis(self):
        """Run comprehensive what-if analysis with multiple scenarios"""
        
        print("=== SUBSCRIPTION BUSINESS SCENARIO ANALYSIS ===")
        
        # Load data and calculate baseline
        if not self.load_baseline_data():
            return
        
        self.calculate_baseline_metrics()
        
        print(f"\nBaseline Metrics:")
        print(f"Active Users: {self.baseline_metrics['active_users']:,}")
        print(f"Monthly Revenue: ${self.baseline_metrics['monthly_revenue']:,.2f}")
        print(f"Churn Rate: {self.baseline_metrics['churn_rate']:.1f}%")
        print(f"ARPU: ${self.baseline_metrics['arpu']:.2f}")
        
        # Define scenarios
        scenarios = {}
        
        print("\n1. Analyzing Churn Reduction Scenario...")
        scenarios['aggressive_retention'] = self.simulate_churn_reduction(
            reduction_percentage=15,
            intervention_cost_per_user=25,
            months_horizon=12
        )
        
        print("2. Analyzing Conservative Churn Reduction...")
        scenarios['conservative_retention'] = self.simulate_churn_reduction(
            reduction_percentage=8,
            intervention_cost_per_user=12,
            months_horizon=12
        )
        
        print("3. Analyzing Price Optimization...")
        scenarios['price_increase'] = self.simulate_price_optimization(
            price_change_percentage=8,
            demand_elasticity=-0.6,
            months_horizon=12
        )
        
        print("4. Analyzing Engagement Improvement...")
        scenarios['engagement_boost'] = self.simulate_engagement_improvement(
            engagement_lift_percentage=25,
            conversion_rate_improvement=3,
            implementation_cost=50000,
            months_horizon=12
        )
        
        # Compare scenarios
        comparison_df = self.compare_scenarios(scenarios)
        
        print("\n=== SCENARIO COMPARISON ===")
        print(comparison_df.round(2))
        
        # Find best scenario
        best_scenario = comparison_df.loc[comparison_df['roi'].idxmax()]
        print(f"\nBest ROI Scenario: {best_scenario['scenario']} ({best_scenario['roi']:.1f}% ROI)")
        
        # Create visualizations
        print("\n5. Creating scenario visualizations...")
        fig = self.visualize_scenario_comparison(scenarios)
        fig.write_html("../presentation/scenario_analysis.html")
        
        # Save detailed results
        with open('../data/scenario_results.json', 'w') as f:
            json.dump(scenarios, f, indent=4, default=str)
        
        print("\nAnalysis complete!")
        print("Results saved to:")
        print("- ../presentation/scenario_analysis.html")
        print("- ../data/scenario_results.json")
        
        return scenarios, comparison_df

if __name__ == "__main__":
    model = SubscriptionScenarioModel()
    scenarios, comparison = model.run_comprehensive_analysis()