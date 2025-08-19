#!/usr/bin/env python3
"""
💰 Advanced Financial Modeling & Scenario Planning
Enterprise-grade financial analytics with Monte Carlo simulations and risk modeling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from scipy import stats
from scipy.optimize import minimize
import sqlite3
import uuid
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, lognorm, beta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScenarioType(Enum):
    OPTIMISTIC = "optimistic"
    REALISTIC = "realistic"
    PESSIMISTIC = "pessimistic"
    STRESS_TEST = "stress_test"

@dataclass
class FinancialMetrics:
    """Financial performance metrics"""
    revenue: float
    cost: float
    profit: float
    margin: float
    roi: float
    payback_period: float
    npv: float
    irr: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%

@dataclass
class ScenarioAssumptions:
    """Scenario planning assumptions"""
    churn_rate_change: float
    acquisition_cost_change: float
    ltv_change: float
    market_growth: float
    competitive_pressure: float
    economic_factor: float
    pricing_flexibility: float

class FinancialModeling:
    """Advanced Financial Modeling and Risk Analysis"""
    
    def __init__(self, db_path: str = "data/financial_models.db"):
        self.db_path = db_path
        self.initialize_database()
        
    def initialize_database(self):
        """Initialize financial modeling database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create scenarios table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS financial_scenarios (
            scenario_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            scenario_type TEXT,
            assumptions TEXT,
            results TEXT,
            created_at TEXT
        )
        ''')
        
        # Create monte carlo results
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS monte_carlo_simulations (
            simulation_id TEXT PRIMARY KEY,
            scenario_id TEXT,
            iteration INTEGER,
            revenue REAL,
            cost REAL,
            profit REAL,
            roi REAL,
            npv REAL,
            simulation_date TEXT,
            FOREIGN KEY (scenario_id) REFERENCES financial_scenarios (scenario_id)
        )
        ''')
        
        # Create risk metrics
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS risk_metrics (
            metric_id TEXT PRIMARY KEY,
            scenario_id TEXT,
            metric_name TEXT,
            value REAL,
            confidence_level REAL,
            calculated_at TEXT,
            FOREIGN KEY (scenario_id) REFERENCES financial_scenarios (scenario_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_base_financial_model(self) -> Dict[str, Any]:
        """Create base financial model from subscription data"""
        # Load subscription data
        try:
            conn = sqlite3.connect("data/subscription_data.db")
            customers_df = pd.read_sql_query("SELECT * FROM customers", conn)
            subscriptions_df = pd.read_sql_query("SELECT * FROM subscriptions", conn)
            conn.close()
        except:
            # Generate synthetic data if real data not available
            customers_df, subscriptions_df = self._generate_synthetic_data()
        
        # Calculate base metrics
        total_customers = len(customers_df)
        monthly_revenue = subscriptions_df['monthly_amount'].sum()
        avg_ltv = customers_df['lifetime_value'].mean()
        avg_acquisition_cost = 50  # Assumed CAC
        
        # Calculate churn rate
        active_customers = subscriptions_df[subscriptions_df['subscription_status'] == 'active']
        churn_rate = 1 - (len(active_customers) / total_customers)
        
        base_model = {
            'total_customers': total_customers,
            'monthly_revenue': monthly_revenue,
            'annual_revenue': monthly_revenue * 12,
            'avg_ltv': avg_ltv,
            'avg_acquisition_cost': avg_acquisition_cost,
            'churn_rate': churn_rate,
            'gross_margin': 0.75,  # Assumed 75% gross margin
            'discount_rate': 0.10,  # 10% discount rate for NPV
        }
        
        return base_model
    
    def run_scenario_analysis(self, scenarios: Dict[str, ScenarioAssumptions], 
                            years: int = 5) -> Dict[str, FinancialMetrics]:
        """Run comprehensive scenario analysis"""
        base_model = self.create_base_financial_model()
        results = {}
        
        for scenario_name, assumptions in scenarios.items():
            logger.info(f"Running scenario: {scenario_name}")
            
            # Project financials over time periods
            projections = []
            for year in range(years):
                projection = self._calculate_yearly_projection(
                    base_model, assumptions, year
                )
                projections.append(projection)
            
            # Calculate aggregate metrics
            metrics = self._calculate_scenario_metrics(projections, base_model)
            results[scenario_name] = metrics
            
            # Save to database
            self._save_scenario(scenario_name, assumptions, metrics)
        
        return results
    
    def _calculate_yearly_projection(self, base_model: Dict[str, Any], 
                                   assumptions: ScenarioAssumptions, year: int) -> Dict[str, float]:
        """Calculate financial projection for a specific year"""
        # Apply compound effects over time
        compound_factor = (1 + assumptions.market_growth) ** year
        churn_impact = (1 + assumptions.churn_rate_change) ** year
        
        # Customer base evolution
        customer_retention = (1 - base_model['churn_rate'] * churn_impact)
        customers = base_model['total_customers'] * (customer_retention ** year) * compound_factor
        
        # Revenue calculation
        avg_revenue_per_customer = (base_model['monthly_revenue'] / base_model['total_customers']) * 12
        pricing_adjustment = (1 + assumptions.pricing_flexibility * year * 0.1)
        revenue = customers * avg_revenue_per_customer * pricing_adjustment
        
        # Cost calculation
        acquisition_cost = base_model['avg_acquisition_cost'] * (1 + assumptions.acquisition_cost_change)
        new_customers = customers * base_model['churn_rate'] * churn_impact
        acquisition_costs = new_customers * acquisition_cost
        
        # Operating costs (scaled with customer base)
        operating_costs = revenue * (1 - base_model['gross_margin'])
        total_costs = acquisition_costs + operating_costs
        
        # Economic and competitive adjustments
        economic_adjustment = 1 + assumptions.economic_factor
        competitive_adjustment = 1 - assumptions.competitive_pressure
        
        revenue *= economic_adjustment * competitive_adjustment
        total_costs *= economic_adjustment
        
        return {
            'year': year + 1,
            'customers': customers,
            'revenue': revenue,
            'costs': total_costs,
            'profit': revenue - total_costs,
            'acquisition_cost': acquisition_cost,
            'ltv': base_model['avg_ltv'] * (1 + assumptions.ltv_change)
        }
    
    def _calculate_scenario_metrics(self, projections: List[Dict[str, float]], 
                                  base_model: Dict[str, Any]) -> FinancialMetrics:
        """Calculate comprehensive financial metrics for scenario"""
        revenues = [p['revenue'] for p in projections]
        costs = [p['costs'] for p in projections]
        profits = [p['profit'] for p in projections]
        
        # Basic metrics
        total_revenue = sum(revenues)
        total_costs = sum(costs)
        total_profit = sum(profits)
        avg_margin = total_profit / total_revenue if total_revenue > 0 else 0
        
        # ROI calculation
        initial_investment = costs[0] if costs else 0
        roi = (total_profit - initial_investment) / initial_investment if initial_investment > 0 else 0
        
        # Payback period (simplified)
        cumulative_profit = 0
        payback_period = len(projections)
        for i, profit in enumerate(profits):
            cumulative_profit += profit
            if cumulative_profit > initial_investment:
                payback_period = i + 1
                break
        
        # NPV calculation
        discount_rate = base_model['discount_rate']
        npv = sum([profit / ((1 + discount_rate) ** (i + 1)) 
                  for i, profit in enumerate(profits)])
        
        # IRR calculation (simplified)
        irr = self._calculate_irr([-initial_investment] + profits)
        
        # Risk metrics (using profit volatility)
        profit_std = np.std(profits)
        var_95 = np.percentile(profits, 5)  # 5th percentile as VaR
        cvar_95 = np.mean([p for p in profits if p <= var_95])
        
        return FinancialMetrics(
            revenue=total_revenue,
            cost=total_costs,
            profit=total_profit,
            margin=avg_margin,
            roi=roi,
            payback_period=payback_period,
            npv=npv,
            irr=irr,
            var_95=var_95,
            cvar_95=cvar_95
        )
    
    def _calculate_irr(self, cash_flows: List[float]) -> float:
        """Calculate Internal Rate of Return"""
        def npv_func(rate):
            return sum([cf / ((1 + rate) ** i) for i, cf in enumerate(cash_flows)])
        
        try:
            result = minimize(lambda x: abs(npv_func(x[0])), x0=[0.1], 
                            bounds=[(0.001, 1.0)], method='L-BFGS-B')
            return result.x[0] if result.success else 0.0
        except:
            return 0.0
    
    def run_monte_carlo_simulation(self, scenario_assumptions: ScenarioAssumptions,
                                 num_simulations: int = 10000) -> Dict[str, Any]:
        """Run Monte Carlo simulation for risk analysis"""
        base_model = self.create_base_financial_model()
        simulation_results = []
        
        scenario_id = str(uuid.uuid4())
        
        logger.info(f"Running {num_simulations} Monte Carlo simulations...")
        
        for iteration in range(num_simulations):
            # Add random variations to assumptions
            varied_assumptions = self._add_random_variations(scenario_assumptions)
            
            # Calculate 5-year projection
            projections = []
            for year in range(5):
                projection = self._calculate_yearly_projection(
                    base_model, varied_assumptions, year
                )
                projections.append(projection)
            
            # Calculate metrics for this iteration
            total_revenue = sum([p['revenue'] for p in projections])
            total_costs = sum([p['costs'] for p in projections])
            total_profit = sum([p['profit'] for p in projections])
            roi = (total_profit - total_costs) / total_costs if total_costs > 0 else 0
            
            # NPV calculation
            discount_rate = base_model['discount_rate']
            profits = [p['profit'] for p in projections]
            npv = sum([profit / ((1 + discount_rate) ** (i + 1)) 
                      for i, profit in enumerate(profits)])
            
            result = {
                'iteration': iteration,
                'revenue': total_revenue,
                'cost': total_costs,
                'profit': total_profit,
                'roi': roi,
                'npv': npv
            }
            
            simulation_results.append(result)
            
            # Save to database every 1000 iterations
            if (iteration + 1) % 1000 == 0:
                self._save_monte_carlo_batch(scenario_id, simulation_results[-1000:])
        
        # Analyze results
        df_results = pd.DataFrame(simulation_results)
        
        analysis = {
            'scenario_id': scenario_id,
            'num_simulations': num_simulations,
            'revenue': {
                'mean': df_results['revenue'].mean(),
                'std': df_results['revenue'].std(),
                'percentiles': {
                    '5th': df_results['revenue'].quantile(0.05),
                    '25th': df_results['revenue'].quantile(0.25),
                    '50th': df_results['revenue'].quantile(0.50),
                    '75th': df_results['revenue'].quantile(0.75),
                    '95th': df_results['revenue'].quantile(0.95)
                }
            },
            'profit': {
                'mean': df_results['profit'].mean(),
                'std': df_results['profit'].std(),
                'percentiles': {
                    '5th': df_results['profit'].quantile(0.05),
                    '25th': df_results['profit'].quantile(0.25),
                    '50th': df_results['profit'].quantile(0.50),
                    '75th': df_results['profit'].quantile(0.75),
                    '95th': df_results['profit'].quantile(0.95)
                }
            },
            'roi': {
                'mean': df_results['roi'].mean(),
                'std': df_results['roi'].std(),
                'probability_positive': (df_results['roi'] > 0).mean()
            },
            'npv': {
                'mean': df_results['npv'].mean(),
                'std': df_results['npv'].std(),
                'probability_positive': (df_results['npv'] > 0).mean()
            },
            'risk_metrics': {
                'var_95_profit': df_results['profit'].quantile(0.05),
                'cvar_95_profit': df_results[df_results['profit'] <= df_results['profit'].quantile(0.05)]['profit'].mean(),
                'var_95_revenue': df_results['revenue'].quantile(0.05),
                'max_loss': df_results['profit'].min(),
                'probability_of_loss': (df_results['profit'] < 0).mean()
            }
        }
        
        return analysis
    
    def _add_random_variations(self, assumptions: ScenarioAssumptions) -> ScenarioAssumptions:
        """Add random variations to scenario assumptions"""
        # Define variation ranges (as percentages of base assumption)
        variation_range = 0.2  # 20% variation
        
        return ScenarioAssumptions(
            churn_rate_change=assumptions.churn_rate_change * (1 + np.random.normal(0, variation_range)),
            acquisition_cost_change=assumptions.acquisition_cost_change * (1 + np.random.normal(0, variation_range)),
            ltv_change=assumptions.ltv_change * (1 + np.random.normal(0, variation_range)),
            market_growth=assumptions.market_growth * (1 + np.random.normal(0, variation_range)),
            competitive_pressure=assumptions.competitive_pressure * (1 + np.random.normal(0, variation_range)),
            economic_factor=assumptions.economic_factor * (1 + np.random.normal(0, variation_range)),
            pricing_flexibility=assumptions.pricing_flexibility * (1 + np.random.normal(0, variation_range))
        )
    
    def _save_scenario(self, name: str, assumptions: ScenarioAssumptions, 
                      metrics: FinancialMetrics):
        """Save scenario results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        scenario_id = str(uuid.uuid4())
        cursor.execute('''
        INSERT INTO financial_scenarios VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            scenario_id, name, "scenario_analysis",
            json.dumps(asdict(assumptions)), json.dumps(asdict(metrics)),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _save_monte_carlo_batch(self, scenario_id: str, results: List[Dict[str, Any]]):
        """Save Monte Carlo simulation batch to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for result in results:
            cursor.execute('''
            INSERT INTO monte_carlo_simulations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), scenario_id, result['iteration'],
                result['revenue'], result['cost'], result['profit'],
                result['roi'], result['npv'], datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    def _generate_synthetic_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic data for modeling"""
        np.random.seed(42)
        
        # Generate customers
        n_customers = 10000
        customers_data = {
            'customer_id': [f'cust_{i}' for i in range(n_customers)],
            'signup_date': pd.date_range('2022-01-01', periods=n_customers, freq='1H'),
            'lifetime_value': np.random.lognormal(mean=5, sigma=0.5, size=n_customers),
            'acquisition_channel': np.random.choice(['organic', 'paid', 'referral'], n_customers),
            'customer_segment': np.random.choice(['premium', 'standard', 'basic'], n_customers)
        }
        customers_df = pd.DataFrame(customers_data)
        
        # Generate subscriptions
        subscriptions_data = {
            'customer_id': customers_df['customer_id'],
            'monthly_amount': np.random.normal(50, 15, n_customers),
            'subscription_status': np.random.choice(['active', 'churned'], n_customers, p=[0.85, 0.15]),
            'plan_type': np.random.choice(['basic', 'pro', 'enterprise'], n_customers)
        }
        subscriptions_df = pd.DataFrame(subscriptions_data)
        
        return customers_df, subscriptions_df
    
    def generate_financial_report(self, scenario_results: Dict[str, FinancialMetrics],
                                monte_carlo_analysis: Dict[str, Any] = None) -> str:
        """Generate comprehensive financial analysis report"""
        report = []
        report.append("# 💰 Comprehensive Financial Analysis Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## 📊 Executive Summary")
        report.append("")
        
        best_scenario = max(scenario_results.items(), key=lambda x: x[1].npv)
        worst_scenario = min(scenario_results.items(), key=lambda x: x[1].npv)
        
        report.append(f"**Best Case Scenario:** {best_scenario[0]}")
        report.append(f"- NPV: ${best_scenario[1].npv:,.0f}")
        report.append(f"- ROI: {best_scenario[1].roi:.1%}")
        report.append(f"- Payback Period: {best_scenario[1].payback_period:.1f} years")
        report.append("")
        
        report.append(f"**Worst Case Scenario:** {worst_scenario[0]}")
        report.append(f"- NPV: ${worst_scenario[1].npv:,.0f}")
        report.append(f"- ROI: {worst_scenario[1].roi:.1%}")
        report.append(f"- Payback Period: {worst_scenario[1].payback_period:.1f} years")
        report.append("")
        
        # Detailed Scenario Analysis
        report.append("## 🎯 Scenario Analysis")
        report.append("")
        
        for scenario_name, metrics in scenario_results.items():
            report.append(f"### {scenario_name.title()} Scenario")
            report.append(f"- **Total Revenue:** ${metrics.revenue:,.0f}")
            report.append(f"- **Total Costs:** ${metrics.cost:,.0f}")
            report.append(f"- **Net Profit:** ${metrics.profit:,.0f}")
            report.append(f"- **Profit Margin:** {metrics.margin:.1%}")
            report.append(f"- **Return on Investment:** {metrics.roi:.1%}")
            report.append(f"- **Net Present Value:** ${metrics.npv:,.0f}")
            report.append(f"- **Internal Rate of Return:** {metrics.irr:.1%}")
            report.append(f"- **Value at Risk (95%):** ${metrics.var_95:,.0f}")
            report.append("")
        
        # Monte Carlo Analysis
        if monte_carlo_analysis:
            report.append("## 🎲 Monte Carlo Risk Analysis")
            report.append("")
            report.append(f"**Simulations Run:** {monte_carlo_analysis['num_simulations']:,}")
            report.append("")
            
            profit_stats = monte_carlo_analysis['profit']
            report.append("### Profit Distribution")
            report.append(f"- **Expected Profit:** ${profit_stats['mean']:,.0f}")
            report.append(f"- **Standard Deviation:** ${profit_stats['std']:,.0f}")
            report.append(f"- **5th Percentile:** ${profit_stats['percentiles']['5th']:,.0f}")
            report.append(f"- **95th Percentile:** ${profit_stats['percentiles']['95th']:,.0f}")
            report.append("")
            
            risk_stats = monte_carlo_analysis['risk_metrics']
            report.append("### Risk Metrics")
            report.append(f"- **Probability of Loss:** {risk_stats['probability_of_loss']:.1%}")
            report.append(f"- **Maximum Potential Loss:** ${risk_stats['max_loss']:,.0f}")
            report.append(f"- **Value at Risk (95%):** ${risk_stats['var_95_profit']:,.0f}")
            report.append(f"- **Conditional VaR (95%):** ${risk_stats['cvar_95_profit']:,.0f}")
            report.append("")
        
        # Recommendations
        report.append("## 💡 Strategic Recommendations")
        report.append("")
        
        if best_scenario[1].npv > 0:
            report.append("**Investment Recommendation: PROCEED**")
            report.append("- Positive NPV across most scenarios indicates strong investment potential")
            report.append("- Focus on optimistic scenario drivers to maximize returns")
        else:
            report.append("**Investment Recommendation: CAUTION**")
            report.append("- Negative NPV in base cases requires risk mitigation strategies")
            report.append("- Consider scenario planning to improve outlook")
        
        report.append("")
        report.append("**Risk Management:**")
        report.append("- Monitor key metrics identified in sensitivity analysis")
        report.append("- Implement early warning systems for scenario deterioration")
        report.append("- Maintain financial reserves for worst-case scenarios")
        
        return "\n".join(report)

def run_comprehensive_analysis():
    """Run comprehensive financial modeling analysis"""
    fm = FinancialModeling()
    
    # Define scenarios
    scenarios = {
        'optimistic': ScenarioAssumptions(
            churn_rate_change=-0.30,  # 30% reduction in churn
            acquisition_cost_change=-0.20,  # 20% reduction in CAC
            ltv_change=0.40,  # 40% increase in LTV
            market_growth=0.15,  # 15% market growth
            competitive_pressure=0.05,  # Low competitive pressure
            economic_factor=0.10,  # Positive economic conditions
            pricing_flexibility=0.20  # Good pricing power
        ),
        'realistic': ScenarioAssumptions(
            churn_rate_change=-0.10,  # 10% reduction in churn
            acquisition_cost_change=0.05,  # 5% increase in CAC
            ltv_change=0.15,  # 15% increase in LTV
            market_growth=0.08,  # 8% market growth
            competitive_pressure=0.10,  # Moderate competitive pressure
            economic_factor=0.00,  # Neutral economic conditions
            pricing_flexibility=0.05  # Limited pricing power
        ),
        'pessimistic': ScenarioAssumptions(
            churn_rate_change=0.20,  # 20% increase in churn
            acquisition_cost_change=0.25,  # 25% increase in CAC
            ltv_change=-0.20,  # 20% decrease in LTV
            market_growth=0.02,  # 2% market growth
            competitive_pressure=0.20,  # High competitive pressure
            economic_factor=-0.10,  # Negative economic conditions
            pricing_flexibility=-0.10  # Pricing pressure
        )
    }
    
    # Run scenario analysis
    print("🚀 Running scenario analysis...")
    scenario_results = fm.run_scenario_analysis(scenarios)
    
    # Run Monte Carlo simulation for realistic scenario
    print("🎲 Running Monte Carlo simulation...")
    monte_carlo_results = fm.run_monte_carlo_simulation(scenarios['realistic'])
    
    # Generate report
    report = fm.generate_financial_report(scenario_results, monte_carlo_results)
    
    print("✅ Financial modeling complete!")
    print("\n" + "="*50)
    print(report)
    print("="*50)
    
    return scenario_results, monte_carlo_results

if __name__ == "__main__":
    run_comprehensive_analysis()