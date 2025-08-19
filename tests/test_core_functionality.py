#!/usr/bin/env python3
"""
Core functionality tests for subscription churn analysis platform
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from churn_analysis import ChurnAnalysis
from customer_lifetime_value import CLVAnalysis
from data_generation import DataGenerator

class TestDataGeneration:
    """Test data generation functionality"""
    
    def test_data_generator_initialization(self):
        """Test that DataGenerator initializes correctly"""
        generator = DataGenerator()
        assert generator is not None
    
    def test_generate_customers(self):
        """Test customer data generation"""
        generator = DataGenerator()
        customers = generator.generate_customers(100)
        
        assert len(customers) == 100
        assert 'customer_id' in customers.columns
        assert 'signup_date' in customers.columns
        assert 'acquisition_channel' in customers.columns
    
    def test_generate_subscriptions(self):
        """Test subscription data generation"""
        generator = DataGenerator()
        customers = generator.generate_customers(50)
        subscriptions = generator.generate_subscriptions(customers)
        
        assert len(subscriptions) >= 50  # Should have at least one subscription per customer
        assert 'customer_id' in subscriptions.columns
        assert 'monthly_amount' in subscriptions.columns
        assert 'subscription_status' in subscriptions.columns

class TestChurnAnalysis:
    """Test churn analysis functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        generator = DataGenerator()
        customers = generator.generate_customers(200)
        subscriptions = generator.generate_subscriptions(customers)
        return customers, subscriptions
    
    def test_churn_analysis_initialization(self, sample_data):
        """Test ChurnAnalysis initialization"""
        customers, subscriptions = sample_data
        analyzer = ChurnAnalysis(customers, subscriptions)
        assert analyzer is not None
    
    def test_calculate_churn_rate(self, sample_data):
        """Test churn rate calculation"""
        customers, subscriptions = sample_data
        analyzer = ChurnAnalysis(customers, subscriptions)
        
        churn_rate = analyzer.calculate_churn_rate()
        assert 0 <= churn_rate <= 1  # Churn rate should be between 0 and 1
        assert isinstance(churn_rate, float)
    
    def test_customer_segmentation(self, sample_data):
        """Test customer segmentation"""
        customers, subscriptions = sample_data
        analyzer = ChurnAnalysis(customers, subscriptions)
        
        segments = analyzer.segment_customers()
        assert isinstance(segments, pd.DataFrame)
        assert len(segments) > 0
        assert 'segment' in segments.columns

class TestCLVAnalysis:
    """Test Customer Lifetime Value analysis"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        generator = DataGenerator()
        customers = generator.generate_customers(100)
        subscriptions = generator.generate_subscriptions(customers)
        return customers, subscriptions
    
    def test_clv_initialization(self, sample_data):
        """Test CLVAnalysis initialization"""
        customers, subscriptions = sample_data
        clv_analyzer = CLVAnalysis(customers, subscriptions)
        assert clv_analyzer is not None
    
    def test_calculate_clv(self, sample_data):
        """Test CLV calculation"""
        customers, subscriptions = sample_data
        clv_analyzer = CLVAnalysis(customers, subscriptions)
        
        clv_results = clv_analyzer.calculate_clv()
        assert isinstance(clv_results, pd.DataFrame)
        assert 'customer_id' in clv_results.columns
        assert 'predicted_clv' in clv_results.columns
        assert all(clv_results['predicted_clv'] >= 0)  # CLV should be non-negative

class TestDataValidation:
    """Test data validation and quality checks"""
    
    def test_data_types(self):
        """Test that generated data has correct types"""
        generator = DataGenerator()
        customers = generator.generate_customers(50)
        
        # Check data types
        assert customers['customer_id'].dtype == 'object'
        assert pd.api.types.is_datetime64_any_dtype(customers['signup_date'])
    
    def test_data_completeness(self):
        """Test that generated data is complete"""
        generator = DataGenerator()
        customers = generator.generate_customers(100)
        subscriptions = generator.generate_subscriptions(customers)
        
        # Check for missing values in critical columns
        assert customers['customer_id'].notna().all()
        assert customers['signup_date'].notna().all()
        assert subscriptions['customer_id'].notna().all()
        assert subscriptions['monthly_amount'].notna().all()
    
    def test_data_consistency(self):
        """Test data consistency between customers and subscriptions"""
        generator = DataGenerator()
        customers = generator.generate_customers(50)
        subscriptions = generator.generate_subscriptions(customers)
        
        # All subscription customer_ids should exist in customers
        customer_ids = set(customers['customer_id'])
        subscription_customer_ids = set(subscriptions['customer_id'])
        assert subscription_customer_ids.issubset(customer_ids)

class TestMetricsCalculation:
    """Test business metrics calculations"""
    
    def test_revenue_calculations(self):
        """Test revenue metric calculations"""
        # Create simple test data
        subscriptions = pd.DataFrame({
            'customer_id': ['c1', 'c2', 'c3'],
            'monthly_amount': [50.0, 75.0, 100.0],
            'subscription_status': ['active', 'active', 'churned']
        })
        
        # Test total revenue
        total_revenue = subscriptions['monthly_amount'].sum()
        assert total_revenue == 225.0
        
        # Test active revenue
        active_revenue = subscriptions[subscriptions['subscription_status'] == 'active']['monthly_amount'].sum()
        assert active_revenue == 125.0
    
    def test_churn_metrics(self):
        """Test churn metric calculations"""
        # Create test data
        customers = pd.DataFrame({
            'customer_id': ['c1', 'c2', 'c3', 'c4'],
            'signup_date': pd.date_range('2023-01-01', periods=4, freq='M')
        })
        
        subscriptions = pd.DataFrame({
            'customer_id': ['c1', 'c2', 'c3', 'c4'],
            'subscription_status': ['active', 'active', 'churned', 'churned'],
            'monthly_amount': [50, 75, 100, 25]
        })
        
        # Calculate churn rate
        total_customers = len(customers)
        churned_customers = len(subscriptions[subscriptions['subscription_status'] == 'churned'])
        churn_rate = churned_customers / total_customers
        
        assert churn_rate == 0.5  # 2 out of 4 customers churned

class TestAdvancedFeatures:
    """Test advanced platform features"""
    
    def test_ml_model_compatibility(self):
        """Test that data is compatible with ML models"""
        generator = DataGenerator()
        customers = generator.generate_customers(100)
        
        # Check that we can create feature matrices
        numeric_features = customers.select_dtypes(include=[np.number])
        assert len(numeric_features.columns) > 0
        
        # Check for ML-ready format
        assert not numeric_features.isnull().all().any()
    
    def test_date_handling(self):
        """Test date handling in data processing"""
        generator = DataGenerator()
        customers = generator.generate_customers(50)
        
        # Test date operations
        recent_customers = customers[customers['signup_date'] > datetime.now() - timedelta(days=365)]
        assert isinstance(recent_customers, pd.DataFrame)
    
    def test_aggregation_functions(self):
        """Test data aggregation capabilities"""
        generator = DataGenerator()
        customers = generator.generate_customers(100)
        subscriptions = generator.generate_subscriptions(customers)
        
        # Test groupby operations
        customer_summary = subscriptions.groupby('customer_id').agg({
            'monthly_amount': ['sum', 'mean', 'count']
        })
        
        assert len(customer_summary) > 0
        assert customer_summary.notna().all().all()

if __name__ == "__main__":
    pytest.main([__file__])