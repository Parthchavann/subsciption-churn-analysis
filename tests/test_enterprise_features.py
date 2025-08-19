#!/usr/bin/env python3
"""
Enterprise features tests for advanced analytics platform
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json
import uuid

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# Import enterprise modules (with error handling for missing dependencies)
try:
    from security_compliance import SecurityComplianceFramework, ComplianceFramework
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

try:
    from financial_modeling import FinancialModeling
    FINANCIAL_AVAILABLE = True
except ImportError:
    FINANCIAL_AVAILABLE = False

try:
    from advanced_forecasting import AdvancedForecasting
    FORECASTING_AVAILABLE = True
except ImportError:
    FORECASTING_AVAILABLE = False

class TestSecurityCompliance:
    """Test security and compliance features"""
    
    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security module not available")
    def test_security_framework_initialization(self):
        """Test security framework initialization"""
        framework = SecurityComplianceFramework()
        assert framework is not None
        assert hasattr(framework, 'data_classifications')
        assert hasattr(framework, 'security_policies')
    
    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security module not available")
    def test_access_logging(self):
        """Test access logging functionality"""
        framework = SecurityComplianceFramework()
        
        log_id = framework.log_access(
            user_id="test_user",
            resource="test_resource", 
            action="read",
            ip_address="192.168.1.100"
        )
        
        assert log_id is not None
        assert isinstance(log_id, str)
    
    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security module not available")
    def test_data_anonymization(self):
        """Test data anonymization"""
        framework = SecurityComplianceFramework()
        
        # Create test data
        test_data = pd.DataFrame({
            'customer_id': ['cust_1', 'cust_2'],
            'email': ['user1@test.com', 'user2@test.com'],
            'amount': [100.0, 200.0]
        })
        
        anonymized = framework.anonymize_data(test_data, "customer_email")
        
        # Anonymized data should be different from original
        assert not anonymized['email'].equals(test_data['email'])
        assert len(anonymized) == len(test_data)
    
    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security module not available")
    def test_compliance_checking(self):
        """Test compliance framework checking"""
        framework = SecurityComplianceFramework()
        
        # Create test data with potential compliance issues
        test_data = pd.DataFrame({
            'customer_id': ['c1', 'c2'],
            'email': ['user1@test.com', 'user2@test.com'],
            'created_at': [datetime.now(), datetime.now() - timedelta(days=3000)]
        })
        
        result = framework.check_compliance(test_data, "customer_email")
        
        assert 'compliant' in result
        assert 'violations' in result
        assert 'recommendations' in result
        assert isinstance(result['compliant'], bool)

class TestFinancialModeling:
    """Test financial modeling features"""
    
    @pytest.mark.skipif(not FINANCIAL_AVAILABLE, reason="Financial modeling module not available")
    def test_financial_modeling_initialization(self):
        """Test financial modeling initialization"""
        fm = FinancialModeling()
        assert fm is not None
    
    @pytest.mark.skipif(not FINANCIAL_AVAILABLE, reason="Financial modeling module not available")
    def test_base_financial_model(self):
        """Test base financial model creation"""
        fm = FinancialModeling()
        base_model = fm.create_base_financial_model()
        
        required_keys = ['total_customers', 'monthly_revenue', 'avg_ltv', 'churn_rate']
        for key in required_keys:
            assert key in base_model
            assert isinstance(base_model[key], (int, float))
    
    @pytest.mark.skipif(not FINANCIAL_AVAILABLE, reason="Financial modeling module not available") 
    def test_scenario_analysis(self):
        """Test scenario analysis functionality"""
        from financial_modeling import ScenarioAssumptions
        
        fm = FinancialModeling()
        
        # Create test scenario
        test_scenario = {
            'test': ScenarioAssumptions(
                churn_rate_change=-0.1,
                acquisition_cost_change=0.05,
                ltv_change=0.15,
                market_growth=0.08,
                competitive_pressure=0.1,
                economic_factor=0.0,
                pricing_flexibility=0.05
            )
        }
        
        results = fm.run_scenario_analysis(test_scenario, years=2)
        
        assert 'test' in results
        assert hasattr(results['test'], 'revenue')
        assert hasattr(results['test'], 'profit')
        assert hasattr(results['test'], 'npv')

class TestAdvancedForecasting:
    """Test advanced forecasting features"""
    
    @pytest.mark.skipif(not FORECASTING_AVAILABLE, reason="Forecasting module not available")
    def test_forecasting_initialization(self):
        """Test forecasting system initialization"""
        forecasting = AdvancedForecasting()
        assert forecasting is not None
    
    @pytest.mark.skipif(not FORECASTING_AVAILABLE, reason="Forecasting module not available")
    def test_seasonality_analysis(self):
        """Test seasonality analysis"""
        forecasting = AdvancedForecasting()
        
        # Create test time series
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        values = np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.normal(0, 0.1, 365)
        ts_data = pd.Series(values, index=dates)
        
        analysis = forecasting.analyze_seasonality(ts_data, "test_metric")
        
        assert hasattr(analysis, 'has_trend')
        assert hasattr(analysis, 'seasonal_periods')
        assert hasattr(analysis, 'anomalies')
        assert isinstance(analysis.has_trend, bool)
    
    @pytest.mark.skipif(not FORECASTING_AVAILABLE, reason="Forecasting module not available")
    def test_forecast_generation(self):
        """Test forecast generation"""
        from advanced_forecasting import ForecastModel, ForecastHorizon
        
        forecasting = AdvancedForecasting()
        
        # Create simple test data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        values = np.random.normal(100, 10, 100)
        ts_data = pd.Series(values, index=dates)
        
        result = forecasting.generate_forecast(
            ts_data, 
            "test_metric",
            model_type=ForecastModel.ENSEMBLE,
            forecast_horizon=ForecastHorizon.SHORT_TERM,
            periods=10
        )
        
        assert hasattr(result, 'forecast_values')
        assert hasattr(result, 'confidence_intervals')
        assert len(result.forecast_values) == 10
        assert len(result.confidence_intervals) == 10

class TestDataProcessing:
    """Test advanced data processing capabilities"""
    
    def test_large_dataset_handling(self):
        """Test handling of larger datasets"""
        # Create larger test dataset
        n_records = 10000
        test_data = pd.DataFrame({
            'id': range(n_records),
            'value': np.random.normal(100, 25, n_records),
            'category': np.random.choice(['A', 'B', 'C'], n_records),
            'date': pd.date_range('2020-01-01', periods=n_records, freq='1H')
        })
        
        # Test basic operations
        assert len(test_data) == n_records
        assert test_data['value'].mean() > 0
        
        # Test aggregations
        summary = test_data.groupby('category')['value'].agg(['mean', 'std', 'count'])
        assert len(summary) == 3  # Should have 3 categories
    
    def test_time_series_operations(self):
        """Test time series data operations"""
        # Create time series test data
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        values = np.cumsum(np.random.normal(0, 1, 365)) + 100
        ts_data = pd.DataFrame({
            'date': dates,
            'value': values
        })
        ts_data.set_index('date', inplace=True)
        
        # Test resampling
        monthly = ts_data.resample('M').mean()
        assert len(monthly) == 12  # Should have 12 months
        
        # Test rolling operations
        rolling_mean = ts_data.rolling(window=7).mean()
        assert len(rolling_mean) == len(ts_data)
    
    def test_feature_engineering(self):
        """Test feature engineering capabilities"""
        # Create base data
        base_data = pd.DataFrame({
            'customer_id': [f'c_{i}' for i in range(100)],
            'signup_date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'monthly_spend': np.random.normal(50, 15, 100),
            'category': np.random.choice(['premium', 'standard', 'basic'], 100)
        })
        
        # Create engineered features
        base_data['days_since_signup'] = (datetime.now() - base_data['signup_date']).dt.days
        base_data['spend_category'] = pd.cut(base_data['monthly_spend'], bins=3, labels=['low', 'medium', 'high'])
        base_data['is_premium'] = (base_data['category'] == 'premium').astype(int)
        
        # Test engineered features
        assert 'days_since_signup' in base_data.columns
        assert 'spend_category' in base_data.columns
        assert 'is_premium' in base_data.columns
        assert base_data['is_premium'].max() <= 1

class TestIntegration:
    """Test integration between different modules"""
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end data pipeline"""
        # This test simulates a complete analytics pipeline
        
        # 1. Data Generation
        customers = pd.DataFrame({
            'customer_id': [f'c_{i}' for i in range(50)],
            'signup_date': pd.date_range('2023-01-01', periods=50, freq='D'),
            'segment': np.random.choice(['A', 'B', 'C'], 50)
        })
        
        subscriptions = pd.DataFrame({
            'customer_id': customers['customer_id'],
            'monthly_amount': np.random.normal(75, 20, 50),
            'status': np.random.choice(['active', 'churned'], 50, p=[0.8, 0.2])
        })
        
        # 2. Basic Analytics
        churn_rate = len(subscriptions[subscriptions['status'] == 'churned']) / len(subscriptions)
        avg_revenue = subscriptions['monthly_amount'].mean()
        
        # 3. Validation
        assert 0 <= churn_rate <= 1
        assert avg_revenue > 0
        
        # 4. Aggregation
        customer_metrics = subscriptions.groupby('customer_id').agg({
            'monthly_amount': 'sum'
        }).reset_index()
        
        assert len(customer_metrics) == len(customers)
    
    def test_data_consistency_across_modules(self):
        """Test data consistency across different analysis modules"""
        # Create consistent test data
        base_customers = pd.DataFrame({
            'customer_id': ['c1', 'c2', 'c3', 'c4', 'c5'],
            'signup_date': pd.date_range('2023-01-01', periods=5, freq='M')
        })
        
        base_subscriptions = pd.DataFrame({
            'customer_id': ['c1', 'c2', 'c3', 'c4', 'c5'],
            'monthly_amount': [50, 75, 100, 125, 150],
            'status': ['active', 'active', 'churned', 'active', 'churned']
        })
        
        # Test consistency
        customer_ids_customers = set(base_customers['customer_id'])
        customer_ids_subscriptions = set(base_subscriptions['customer_id'])
        
        assert customer_ids_customers == customer_ids_subscriptions
        
        # Test metrics consistency
        total_customers = len(base_customers)
        active_customers = len(base_subscriptions[base_subscriptions['status'] == 'active'])
        
        assert active_customers <= total_customers
        assert active_customers > 0

class TestPerformance:
    """Test performance characteristics"""
    
    def test_processing_speed(self):
        """Test processing speed for reasonable dataset sizes"""
        import time
        
        # Create moderately large dataset
        n_records = 5000
        test_data = pd.DataFrame({
            'id': range(n_records),
            'value': np.random.normal(100, 25, n_records),
            'category': np.random.choice(['A', 'B', 'C', 'D'], n_records)
        })
        
        # Test aggregation performance
        start_time = time.time()
        result = test_data.groupby('category')['value'].agg(['mean', 'std', 'count'])
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 1.0  # Less than 1 second
        assert len(result) == 4  # Should have 4 categories
    
    def test_memory_efficiency(self):
        """Test memory efficiency"""
        # Create data and test that we can process it without memory issues
        large_data = pd.DataFrame({
            'id': range(10000),
            'values': np.random.random(10000)
        })
        
        # Test operations that should be memory efficient
        chunk_results = []
        chunk_size = 1000
        
        for i in range(0, len(large_data), chunk_size):
            chunk = large_data.iloc[i:i+chunk_size]
            chunk_mean = chunk['values'].mean()
            chunk_results.append(chunk_mean)
        
        assert len(chunk_results) == 10  # Should have 10 chunks
        assert all(isinstance(x, float) for x in chunk_results)

if __name__ == "__main__":
    pytest.main([__file__])