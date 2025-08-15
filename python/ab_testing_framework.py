"""
A/B Testing Framework for Subscription Optimization
Statistical testing for churn reduction experiments and revenue optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical testing libraries
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import ttest_power
import seaborn as sns

class ABTestingFramework:
    def __init__(self):
        self.experiments = {}
        self.results = {}
        
    def power_analysis(self, baseline_rate, effect_size, alpha=0.05, power=0.80):
        """Calculate required sample size for A/B test"""
        
        # Calculate minimum detectable effect
        treatment_rate = baseline_rate * (1 + effect_size)
        
        # Effect size for power calculation
        effect_size_cohen = abs(treatment_rate - baseline_rate) / np.sqrt(baseline_rate * (1 - baseline_rate))
        
        # Sample size calculation
        sample_size = ttest_power(effect_size_cohen, power=power, alpha=alpha, alternative='two-sided')
        
        # For proportion test (more accurate for conversion rates)
        p1, p2 = baseline_rate, treatment_rate
        p_pooled = (p1 + p2) / 2
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * p_pooled * (1 - p_pooled) * ((z_alpha + z_beta) / (p2 - p1)) ** 2
        
        return int(np.ceil(n))
    
    def generate_ab_test_data(self, test_name, sample_size_per_group=5000):
        """Generate realistic A/B test data for subscription experiments"""
        print(f"🧪 Generating A/B test data for: {test_name}")
        
        np.random.seed(42)
        
        # Different test scenarios
        if test_name == "pricing_test":
            # Test: 20% price increase vs. current pricing
            control_price = 14.99
            treatment_price = 17.99
            
            # Price sensitivity affects conversion and churn
            control_conversion = 0.15  # 15% trial-to-paid conversion
            treatment_conversion = 0.12  # Lower due to higher price
            
            control_churn = 0.08  # 8% monthly churn
            treatment_churn = 0.10  # Higher churn due to price sensitivity
            
            # Revenue impact
            control_revenue = control_price
            treatment_revenue = treatment_price
            
        elif test_name == "onboarding_test":
            # Test: New onboarding flow vs. current
            control_price = treatment_price = 14.99
            
            # Better onboarding improves conversion and retention
            control_conversion = 0.12
            treatment_conversion = 0.18  # 50% relative improvement
            
            control_churn = 0.10
            treatment_churn = 0.07  # Better retention
            
            control_revenue = treatment_revenue = control_price
            
        elif test_name == "retention_campaign":
            # Test: Proactive retention emails vs. no intervention
            control_price = treatment_price = 14.99
            
            # Same conversion, different churn
            control_conversion = treatment_conversion = 0.15
            
            control_churn = 0.12  # Higher baseline churn
            treatment_churn = 0.08  # Retention campaign reduces churn
            
            control_revenue = treatment_revenue = control_price
            
        else:
            # Default test
            control_price = treatment_price = 14.99
            control_conversion = 0.15
            treatment_conversion = 0.17
            control_churn = 0.10
            treatment_churn = 0.08
            control_revenue = treatment_revenue = control_price
        
        # Generate user data
        users = []
        for i in range(sample_size_per_group * 2):
            # Random assignment
            group = 'treatment' if i >= sample_size_per_group else 'control'
            
            # User characteristics
            age = np.random.normal(35, 10)
            age = max(18, min(65, age))
            
            country = np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'], p=[0.4, 0.2, 0.15, 0.15, 0.1])
            device = np.random.choice(['Mobile', 'Desktop', 'Tablet'], p=[0.6, 0.3, 0.1])
            
            # Outcomes based on group
            if group == 'control':
                converted = np.random.random() < control_conversion
                if converted:
                    churned = np.random.random() < control_churn
                    revenue = control_revenue if not churned else 0
                else:
                    churned = True
                    revenue = 0
            else:  # treatment
                converted = np.random.random() < treatment_conversion
                if converted:
                    churned = np.random.random() < treatment_churn
                    revenue = treatment_revenue if not churned else 0
                else:
                    churned = True
                    revenue = 0
            
            # Engagement metrics (affected by group)
            base_engagement = 0.6
            if group == 'treatment' and test_name == "onboarding_test":
                base_engagement = 0.75  # Better onboarding = higher engagement
            
            engagement_score = np.random.beta(base_engagement * 10, (1-base_engagement) * 10)
            usage_hours = np.random.gamma(2, 15) if converted else np.random.gamma(1, 5)
            
            user = {
                'user_id': f'{test_name}_{i:06d}',
                'group': group,
                'age': int(age),
                'country': country,
                'device': device,
                'converted': converted,
                'churned': churned,
                'revenue': revenue,
                'engagement_score': round(engagement_score, 3),
                'usage_hours': round(usage_hours, 1),
                'test_start_date': datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 90)),
                'test_duration_days': 30  # 30-day test
            }
            users.append(user)
        
        df = pd.DataFrame(users)
        
        print(f"✅ Generated {len(df):,} users ({sample_size_per_group:,} per group)")
        print(f"📊 Control conversion: {df[df['group']=='control']['converted'].mean():.1%}")
        print(f"📊 Treatment conversion: {df[df['group']=='treatment']['converted'].mean():.1%}")
        
        return df
    
    def run_conversion_test(self, df, test_name):
        """Run statistical test for conversion rate differences"""
        print(f"📈 Running conversion rate test for: {test_name}")
        
        # Separate control and treatment groups
        control = df[df['group'] == 'control']
        treatment = df[df['group'] == 'treatment']
        
        # Conversion rates
        control_conv_rate = control['converted'].mean()
        treatment_conv_rate = treatment['converted'].mean()
        
        # Statistical test using z-test for proportions
        control_conversions = control['converted'].sum()
        treatment_conversions = treatment['converted'].sum()
        control_total = len(control)
        treatment_total = len(treatment)
        
        # Two-proportion z-test
        counts = np.array([treatment_conversions, control_conversions])
        nobs = np.array([treatment_total, control_total])
        
        stat, pvalue = proportions_ztest(counts, nobs)
        
        # Effect size (relative improvement)
        relative_improvement = (treatment_conv_rate - control_conv_rate) / control_conv_rate
        absolute_improvement = treatment_conv_rate - control_conv_rate
        
        # Confidence interval for difference
        se_diff = np.sqrt(
            (control_conv_rate * (1 - control_conv_rate) / control_total) +
            (treatment_conv_rate * (1 - treatment_conv_rate) / treatment_total)
        )
        ci_lower = absolute_improvement - 1.96 * se_diff
        ci_upper = absolute_improvement + 1.96 * se_diff
        
        results = {
            'test_type': 'conversion_rate',
            'control_rate': control_conv_rate,
            'treatment_rate': treatment_conv_rate,
            'absolute_improvement': absolute_improvement,
            'relative_improvement': relative_improvement,
            'p_value': pvalue,
            'is_significant': pvalue < 0.05,
            'confidence_interval': (ci_lower, ci_upper),
            'sample_size_control': control_total,
            'sample_size_treatment': treatment_total
        }
        
        return results
    
    def run_revenue_test(self, df, test_name):
        """Run statistical test for revenue differences"""
        print(f"💰 Running revenue test for: {test_name}")
        
        # Separate groups
        control_revenue = df[df['group'] == 'control']['revenue']
        treatment_revenue = df[df['group'] == 'treatment']['revenue']
        
        # Revenue metrics
        control_mean = control_revenue.mean()
        treatment_mean = treatment_revenue.mean()
        
        # Statistical test - t-test for means
        stat, pvalue = ttest_ind(treatment_revenue, control_revenue)
        
        # Effect size
        relative_improvement = (treatment_mean - control_mean) / control_mean if control_mean > 0 else 0
        absolute_improvement = treatment_mean - control_mean
        
        # Confidence interval
        se_diff = np.sqrt(
            control_revenue.var() / len(control_revenue) +
            treatment_revenue.var() / len(treatment_revenue)
        )
        ci_lower = absolute_improvement - 1.96 * se_diff
        ci_upper = absolute_improvement + 1.96 * se_diff
        
        results = {
            'test_type': 'revenue',
            'control_revenue': control_mean,
            'treatment_revenue': treatment_mean,
            'absolute_improvement': absolute_improvement,
            'relative_improvement': relative_improvement,
            'p_value': pvalue,
            'is_significant': pvalue < 0.05,
            'confidence_interval': (ci_lower, ci_upper)
        }
        
        return results
    
    def run_churn_test(self, df, test_name):
        """Run statistical test for churn rate differences"""
        print(f"📉 Running churn rate test for: {test_name}")
        
        # Only look at converted users for churn analysis
        converted_users = df[df['converted'] == True]
        
        if len(converted_users) == 0:
            return {'test_type': 'churn', 'error': 'No converted users to analyze'}
        
        # Separate groups
        control = converted_users[converted_users['group'] == 'control']
        treatment = converted_users[converted_users['group'] == 'treatment']
        
        if len(control) == 0 or len(treatment) == 0:
            return {'test_type': 'churn', 'error': 'Insufficient data in one or both groups'}
        
        # Churn rates
        control_churn_rate = control['churned'].mean()
        treatment_churn_rate = treatment['churned'].mean()
        
        # Statistical test
        control_churns = control['churned'].sum()
        treatment_churns = treatment['churned'].sum()
        
        counts = np.array([treatment_churns, control_churns])
        nobs = np.array([len(treatment), len(control)])
        
        stat, pvalue = proportions_ztest(counts, nobs)
        
        # Effect size (note: negative is good for churn)
        relative_improvement = (control_churn_rate - treatment_churn_rate) / control_churn_rate if control_churn_rate > 0 else 0
        absolute_improvement = control_churn_rate - treatment_churn_rate  # Positive means churn reduction
        
        results = {
            'test_type': 'churn',
            'control_churn_rate': control_churn_rate,
            'treatment_churn_rate': treatment_churn_rate,
            'absolute_improvement': absolute_improvement,  # Positive = churn reduction
            'relative_improvement': relative_improvement,  # Positive = % churn reduction
            'p_value': pvalue,
            'is_significant': pvalue < 0.05
        }
        
        return results
    
    def calculate_business_impact(self, df, results, test_name):
        """Calculate business impact of the A/B test results"""
        print(f"💼 Calculating business impact for: {test_name}")
        
        total_users = len(df) // 2  # Users per group
        
        # Get results
        conv_results = results.get('conversion_rate', {})
        revenue_results = results.get('revenue', {})
        churn_results = results.get('churn', {})
        
        # Business impact calculations
        impact = {}
        
        # Conversion impact
        if 'absolute_improvement' in conv_results:
            additional_conversions = conv_results['absolute_improvement'] * total_users
            impact['additional_conversions_per_month'] = int(additional_conversions)
        
        # Revenue impact
        if 'absolute_improvement' in revenue_results:
            additional_revenue_per_user = revenue_results['absolute_improvement']
            total_additional_revenue = additional_revenue_per_user * total_users
            impact['additional_monthly_revenue'] = total_additional_revenue
            impact['annual_revenue_impact'] = total_additional_revenue * 12
        
        # Churn impact (retention)
        if 'absolute_improvement' in churn_results and churn_results['absolute_improvement'] > 0:
            churn_reduction = churn_results['absolute_improvement']
            # Assume average customer value for retention calculation
            avg_customer_value = df[df['converted']==True]['revenue'].mean()
            retained_customers = churn_reduction * total_users
            retention_value = retained_customers * avg_customer_value * 12  # Annual value
            impact['customers_retained_monthly'] = int(retained_customers)
            impact['retention_value_annual'] = retention_value
        
        return impact
    
    def create_ab_test_dashboard(self, df, results, test_name):
        """Create comprehensive A/B test results dashboard"""
        print(f"📊 Creating A/B test dashboard for: {test_name}")
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Conversion Rate by Group',
                'Revenue Distribution',
                'Churn Rate Comparison',
                'Sample Size & Power',
                'Statistical Significance',
                'Business Impact Summary'
            ),
            specs=[[{"type": "bar"}, {"type": "box"}, {"type": "bar"}],
                   [{"type": "table"}, {"type": "bar"}, {"type": "table"}]]
        )
        
        # Conversion rate comparison
        conv_data = df.groupby('group')['converted'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=conv_data['group'],
                y=conv_data['converted'],
                name='Conversion Rate',
                marker_color=['lightblue', 'lightcoral'],
                text=[f"{x:.1%}" for x in conv_data['converted']],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Revenue distribution
        for group in ['control', 'treatment']:
            group_data = df[df['group'] == group]['revenue']
            fig.add_trace(
                go.Box(
                    y=group_data,
                    name=f'{group.title()} Revenue',
                    boxpoints='outliers'
                ),
                row=1, col=2
            )
        
        # Churn rate comparison (for converted users)
        converted_df = df[df['converted'] == True]
        if len(converted_df) > 0:
            churn_data = converted_df.groupby('group')['churned'].mean().reset_index()
            fig.add_trace(
                go.Bar(
                    x=churn_data['group'],
                    y=churn_data['churned'],
                    name='Churn Rate',
                    marker_color=['lightgreen', 'lightpink'],
                    text=[f"{x:.1%}" for x in churn_data['churned']],
                    textposition='auto'
                ),
                row=1, col=3
            )
        
        # Statistical significance summary
        sig_data = []
        for test_type, result in results.items():
            if 'p_value' in result:
                sig_data.append({
                    'Test': test_type.replace('_', ' ').title(),
                    'P-value': f"{result['p_value']:.4f}",
                    'Significant': '✅ Yes' if result['is_significant'] else '❌ No'
                })
        
        if sig_data:
            fig.add_trace(
                go.Bar(
                    x=[d['Test'] for d in sig_data],
                    y=[float(d['P-value']) for d in sig_data],
                    name='P-values',
                    marker_color='orange'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"A/B Test Results Dashboard - {test_name.replace('_', ' ').title()}",
            showlegend=False
        )
        
        # Save dashboard
        fig.write_html(f"../presentation/ab_test_dashboard_{test_name}.html")
        
        return fig
    
    def run_comprehensive_ab_tests(self):
        """Run comprehensive A/B testing analysis"""
        print("🚀 COMPREHENSIVE A/B TESTING FRAMEWORK")
        print("=" * 60)
        
        # Define test scenarios
        test_scenarios = [
            "pricing_test",
            "onboarding_test", 
            "retention_campaign"
        ]
        
        all_results = {}
        
        for test_name in test_scenarios:
            print(f"\n🧪 Running A/B Test: {test_name.replace('_', ' ').title()}")
            print("-" * 50)
            
            # Generate test data
            df = self.generate_ab_test_data(test_name)
            
            # Store experiment data
            self.experiments[test_name] = df
            
            # Run statistical tests
            test_results = {}
            
            # Conversion test
            conv_results = self.run_conversion_test(df, test_name)
            test_results['conversion_rate'] = conv_results
            
            # Revenue test
            revenue_results = self.run_revenue_test(df, test_name)
            test_results['revenue'] = revenue_results
            
            # Churn test
            churn_results = self.run_churn_test(df, test_name)
            if 'error' not in churn_results:
                test_results['churn'] = churn_results
            
            # Business impact
            business_impact = self.calculate_business_impact(df, test_results, test_name)
            test_results['business_impact'] = business_impact
            
            # Store results
            self.results[test_name] = test_results
            all_results[test_name] = test_results
            
            # Create dashboard
            dashboard = self.create_ab_test_dashboard(df, test_results, test_name)
            
            # Save data
            df.to_csv(f'../data/real/ab_test_{test_name}.csv', index=False)
        
        # Summary results
        print("\n" + "=" * 60)
        print("🎯 A/B TESTING RESULTS SUMMARY")
        print("=" * 60)
        
        for test_name, results in all_results.items():
            print(f"\n📊 {test_name.replace('_', ' ').title()}:")
            
            if 'conversion_rate' in results:
                conv = results['conversion_rate']
                print(f"  Conversion: {conv['relative_improvement']:+.1%} " +
                      f"({'✅ Significant' if conv['is_significant'] else '❌ Not Significant'})")
            
            if 'revenue' in results:
                rev = results['revenue']
                print(f"  Revenue: ${rev['absolute_improvement']:+.2f} per user " +
                      f"({'✅ Significant' if rev['is_significant'] else '❌ Not Significant'})")
            
            if 'churn' in results:
                churn = results['churn']
                print(f"  Churn Reduction: {churn['relative_improvement']:+.1%} " +
                      f"({'✅ Significant' if churn['is_significant'] else '❌ Not Significant'})")
            
            if 'business_impact' in results:
                impact = results['business_impact']
                if 'annual_revenue_impact' in impact:
                    print(f"  💰 Annual Revenue Impact: ${impact['annual_revenue_impact']:,.0f}")
        
        print(f"\n📁 Files saved:")
        for test_name in test_scenarios:
            print(f"  - ../data/real/ab_test_{test_name}.csv")
            print(f"  - ../presentation/ab_test_dashboard_{test_name}.html")
        
        print("\n✨ A/B Testing Framework Complete!")
        return all_results

if __name__ == "__main__":
    ab_tester = ABTestingFramework()
    results = ab_tester.run_comprehensive_ab_tests()