"""
Advanced Customer Lifetime Value (CLV) Prediction
Enterprise-grade CLV modeling with cohort analysis and revenue forecasting
"""
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

class CustomerLifetimeValueAnalyzer:
    """Advanced CLV analysis and prediction system"""
    
    def __init__(self, data_path="data/processed/"):
        self.data_path = data_path
        self.customers_data = []
        self.subscriptions_data = []
        
    def load_real_data(self):
        """Load real customer and subscription data"""
        try:
            # Load customers
            with open(f"{self.data_path}real_customers.csv", 'r') as file:
                reader = csv.DictReader(file)
                self.customers_data = list(reader)
                
            # Load subscriptions  
            with open(f"{self.data_path}real_subscriptions.csv", 'r') as file:
                reader = csv.DictReader(file)
                self.subscriptions_data = list(reader)
                
            print(f"✅ Loaded {len(self.customers_data)} customers and {len(self.subscriptions_data)} subscriptions")
            
        except FileNotFoundError:
            print("❌ Real data files not found. Run download_real_data.py first.")
            return False
        return True
        
    def calculate_traditional_clv(self):
        """Calculate CLV using traditional methods with real data"""
        if not self.customers_data:
            self.load_real_data()
            
        clv_results = []
        
        for customer in self.customers_data:
            user_id = customer['user_id']
            monthly_charges = float(customer['monthly_charges'])
            tenure_months = int(customer['tenure_months'])
            total_charges = float(customer['total_charges'])
            churn_status = int(customer['churn'])
            
            # Calculate CLV metrics
            traditional_clv = monthly_charges * tenure_months
            
            # Adjust for churn
            if churn_status == 1:
                # Customer churned - use actual values
                actual_clv = total_charges
            else:
                # Customer active - predict future value
                predicted_future_months = 12  # Predict 12 more months
                predicted_clv = total_charges + (monthly_charges * predicted_future_months)
                actual_clv = predicted_clv
                
            clv_results.append({
                'user_id': user_id,
                'monthly_revenue': monthly_charges,
                'tenure_months': tenure_months,
                'total_revenue': total_charges,
                'traditional_clv': traditional_clv,
                'predicted_clv': actual_clv,
                'is_churned': churn_status,
                'contract_type': customer.get('contract', 'Month-to-month'),
                'payment_method': customer.get('payment_method', 'Electronic check')
            })
            
        # Calculate aggregate metrics
        total_customers = len(clv_results)
        total_revenue = sum(c['total_revenue'] for c in clv_results)
        avg_clv = sum(c['predicted_clv'] for c in clv_results) / total_customers
        avg_monthly_revenue = sum(c['monthly_revenue'] for c in clv_results) / total_customers
        churn_rate = sum(c['is_churned'] for c in clv_results) / total_customers
        
        return {
            'customer_clv': clv_results,
            'total_customers': total_customers,
            'total_revenue': total_revenue,
            'average_clv': avg_clv,
            'avg_monthly_revenue': avg_monthly_revenue,
            'churn_rate': churn_rate
        }
        
    def segment_customers_by_clv(self):
        """Segment customers by CLV potential"""
        clv_data = self.calculate_traditional_clv()
        customers = clv_data['customer_clv']
        
        # Sort by predicted CLV
        customers.sort(key=lambda x: x['predicted_clv'], reverse=True)
        
        # Create quartile segments
        total_customers = len(customers)
        q1 = total_customers // 4
        q2 = total_customers // 2
        q3 = (3 * total_customers) // 4
        
        segments = {}
        
        for i, customer in enumerate(customers):
            if i < q1:
                segment = 'Premium Value'
            elif i < q2:
                segment = 'High Value'
            elif i < q3:
                segment = 'Medium Value'
            else:
                segment = 'Low Value'
                
            customer['clv_segment'] = segment
            
            if segment not in segments:
                segments[segment] = []
            segments[segment].append(customer)
            
        # Calculate segment statistics
        segment_stats = {}
        for segment, customers_list in segments.items():
            segment_stats[segment] = {
                'count': len(customers_list),
                'avg_clv': sum(c['predicted_clv'] for c in customers_list) / len(customers_list),
                'avg_monthly_revenue': sum(c['monthly_revenue'] for c in customers_list) / len(customers_list),
                'avg_tenure': sum(c['tenure_months'] for c in customers_list) / len(customers_list),
                'churn_rate': sum(c['is_churned'] for c in customers_list) / len(customers_list)
            }
            
        return {
            'segments': segments,
            'segment_stats': segment_stats,
            'all_customers': customers
        }
        
    def analyze_clv_by_contract_type(self):
        """Analyze CLV by contract type"""
        clv_data = self.calculate_traditional_clv()
        customers = clv_data['customer_clv']
        
        contract_analysis = {}
        
        for customer in customers:
            contract_type = customer['contract_type']
            
            if contract_type not in contract_analysis:
                contract_analysis[contract_type] = []
            contract_analysis[contract_type].append(customer)
            
        # Calculate statistics by contract type
        contract_stats = {}
        for contract_type, customers_list in contract_analysis.items():
            contract_stats[contract_type] = {
                'count': len(customers_list),
                'avg_clv': sum(c['predicted_clv'] for c in customers_list) / len(customers_list),
                'avg_monthly_revenue': sum(c['monthly_revenue'] for c in customers_list) / len(customers_list),
                'avg_tenure': sum(c['tenure_months'] for c in customers_list) / len(customers_list),
                'churn_rate': sum(c['is_churned'] for c in customers_list) / len(customers_list),
                'total_revenue': sum(c['total_revenue'] for c in customers_list)
            }
            
        return contract_stats
        
    def predict_revenue_impact(self):
        """Predict revenue impact of churn reduction strategies"""
        clv_data = self.calculate_traditional_clv()
        segment_data = self.segment_customers_by_clv()
        
        # Current state
        current_revenue = clv_data['total_revenue']
        current_churn_rate = clv_data['churn_rate']
        
        # Scenarios
        scenarios = {
            'reduce_churn_10%': {
                'churn_reduction': 0.1,
                'description': 'Reduce churn by 10%'
            },
            'reduce_churn_15%': {
                'churn_reduction': 0.15,
                'description': 'Reduce churn by 15%'
            },
            'premium_retention': {
                'churn_reduction': 0.2,
                'target_segment': 'Premium Value',
                'description': 'Focus on premium customer retention'
            }
        }
        
        impact_analysis = {}
        
        for scenario_name, scenario in scenarios.items():
            # Calculate impact
            if 'target_segment' in scenario:
                # Targeted approach
                target_customers = segment_data['segments'][scenario['target_segment']]
                affected_revenue = sum(c['predicted_clv'] for c in target_customers)
                churn_reduction_impact = affected_revenue * scenario['churn_reduction']
            else:
                # Overall approach
                churn_reduction_impact = current_revenue * scenario['churn_reduction']
                
            # Estimated costs (industry benchmarks)
            retention_cost_per_customer = 50  # $50 per customer retention effort
            affected_customers = clv_data['total_customers']
            total_retention_cost = affected_customers * retention_cost_per_customer
            
            # ROI calculation
            net_benefit = churn_reduction_impact - total_retention_cost
            roi = (net_benefit / total_retention_cost) if total_retention_cost > 0 else 0
            
            impact_analysis[scenario_name] = {
                'description': scenario['description'],
                'revenue_impact': churn_reduction_impact,
                'retention_cost': total_retention_cost,
                'net_benefit': net_benefit,
                'roi': roi,
                'payback_months': (total_retention_cost / (churn_reduction_impact / 12)) if churn_reduction_impact > 0 else float('inf')
            }
            
        return {
            'current_state': {
                'total_revenue': current_revenue,
                'churn_rate': current_churn_rate,
                'customers': clv_data['total_customers']
            },
            'scenarios': impact_analysis
        }
        
    def generate_clv_insights(self):
        """Generate comprehensive CLV insights and recommendations"""
        clv_data = self.calculate_traditional_clv()
        segments = self.segment_customers_by_clv()
        contracts = self.analyze_clv_by_contract_type()
        revenue_impact = self.predict_revenue_impact()
        
        # Key insights
        insights = {
            'executive_summary': {
                'total_customers': clv_data['total_customers'],
                'total_revenue': round(clv_data['total_revenue'], 2),
                'average_clv': round(clv_data['average_clv'], 2),
                'churn_rate': round(clv_data['churn_rate'] * 100, 1),
                'avg_monthly_revenue': round(clv_data['avg_monthly_revenue'], 2)
            },
            'segment_insights': {},
            'contract_insights': contracts,
            'revenue_opportunities': revenue_impact,
            'key_findings': [],
            'recommendations': []
        }
        
        # Segment insights
        for segment, stats in segments['segment_stats'].items():
            insights['segment_insights'][segment] = {
                'customer_count': stats['count'],
                'avg_clv': round(stats['avg_clv'], 2),
                'churn_rate': round(stats['churn_rate'] * 100, 1),
                'avg_tenure': round(stats['avg_tenure'], 1)
            }
            
        # Key findings
        best_contract = max(contracts.keys(), key=lambda x: contracts[x]['avg_clv'])
        worst_churn_segment = max(segments['segment_stats'].keys(), 
                                key=lambda x: segments['segment_stats'][x]['churn_rate'])
        
        insights['key_findings'] = [
            f"Total revenue at risk: ${clv_data['total_revenue']:,.2f}",
            f"Current churn rate: {clv_data['churn_rate']*100:.1f}%",
            f"Best performing contract: {best_contract} (${contracts[best_contract]['avg_clv']:.2f} avg CLV)",
            f"Highest risk segment: {worst_churn_segment} ({segments['segment_stats'][worst_churn_segment]['churn_rate']*100:.1f}% churn)",
            f"Premium segment: {segments['segment_stats']['Premium Value']['count']} customers"
        ]
        
        # Recommendations
        insights['recommendations'] = [
            f"Focus retention on {worst_churn_segment} segment customers",
            f"Promote {best_contract} contracts to increase CLV",
            "Implement predictive churn alerts for Premium Value customers",
            f"Target 15% churn reduction for ${revenue_impact['scenarios']['reduce_churn_15%']['revenue_impact']:,.2f} impact",
            "Develop customer success programs for high-value segments"
        ]
        
        # Save insights
        with open('data/clv_insights_real.json', 'w') as f:
            json.dump(insights, f, indent=2, default=str)
            
        return insights
        
    def create_executive_summary(self):
        """Create executive-ready CLV summary"""
        insights = self.generate_clv_insights()
        
        summary = f"""
# Customer Lifetime Value Analysis - Executive Summary

## Key Metrics
- **Total Customers**: {insights['executive_summary']['total_customers']:,}
- **Total Revenue**: ${insights['executive_summary']['total_revenue']:,.2f}
- **Average CLV**: ${insights['executive_summary']['average_clv']:,.2f}
- **Current Churn Rate**: {insights['executive_summary']['churn_rate']}%
- **Average Monthly Revenue**: ${insights['executive_summary']['avg_monthly_revenue']:.2f}

## Revenue Opportunities
"""
        
        for scenario_name, scenario in insights['revenue_opportunities']['scenarios'].items():
            summary += f"- **{scenario['description']}**: ${scenario['revenue_impact']:,.2f} revenue impact (ROI: {scenario['roi']*100:.1f}%)\n"
            
        summary += "\n## Key Recommendations\n"
        for rec in insights['recommendations']:
            summary += f"- {rec}\n"
            
        with open('docs/clv_executive_summary.md', 'w') as f:
            f.write(summary)
            
        return summary

if __name__ == "__main__":
    print("🔍 Starting Advanced CLV Analysis with Real Data...")
    
    analyzer = CustomerLifetimeValueAnalyzer()
    
    if analyzer.load_real_data():
        print("\n📊 Calculating CLV metrics...")
        insights = analyzer.generate_clv_insights()
        
        print("\n📈 Creating executive summary...")
        summary = analyzer.create_executive_summary()
        
        print("\n" + "="*60)
        print("CLV ANALYSIS COMPLETE - REAL DATA")
        print("="*60)
        print(f"📊 Total Customers: {insights['executive_summary']['total_customers']:,}")
        print(f"💰 Total Revenue: ${insights['executive_summary']['total_revenue']:,.2f}")
        print(f"🎯 Average CLV: ${insights['executive_summary']['average_clv']:,.2f}")
        print(f"⚠️  Churn Rate: {insights['executive_summary']['churn_rate']}%")
        
        print(f"\n🚀 Revenue Opportunities:")
        for scenario_name, scenario in insights['revenue_opportunities']['scenarios'].items():
            print(f"   • {scenario['description']}: ${scenario['revenue_impact']:,.2f}")
            
        print(f"\n📁 Files Generated:")
        print(f"   • data/clv_insights_real.json")
        print(f"   • docs/clv_executive_summary.md")
        
    else:
        print("❌ Failed to load real data. Please run download_real_data.py first.")