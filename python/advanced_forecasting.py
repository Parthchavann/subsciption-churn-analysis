#!/usr/bin/env python3
"""
🔮 Advanced Time Series Forecasting & Predictive Analytics
Enterprise-grade forecasting with multiple models, uncertainty quantification, and business scenarios
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import logging
from scipy import stats
from scipy.optimize import minimize
import sqlite3
import uuid
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Mock advanced forecasting libraries (in production use actual libraries)
class MockForecastModels:
    @staticmethod
    def prophet_forecast(data, periods=12):
        # Mock Prophet forecast
        trend = np.linspace(data[-1], data[-1] * 1.1, periods)
        seasonal = np.sin(np.arange(periods) * 2 * np.pi / 12) * data[-1] * 0.05
        noise = np.random.normal(0, data[-1] * 0.02, periods)
        return {
            'forecast': trend + seasonal + noise,
            'lower': trend + seasonal + noise - data[-1] * 0.1,
            'upper': trend + seasonal + noise + data[-1] * 0.1
        }
    
    @staticmethod
    def arima_forecast(data, periods=12):
        # Mock ARIMA forecast
        last_val = data[-1]
        trend = 0.01
        forecast = []
        for i in range(periods):
            next_val = last_val * (1 + trend) + np.random.normal(0, last_val * 0.03)
            forecast.append(next_val)
            last_val = next_val
        return {
            'forecast': np.array(forecast),
            'confidence': 0.85
        }
    
    @staticmethod
    def lstm_forecast(data, periods=12):
        # Mock LSTM forecast
        base = data[-1]
        growth_rate = 0.02
        forecast = []
        for i in range(periods):
            val = base * ((1 + growth_rate) ** (i + 1)) + np.random.normal(0, base * 0.02)
            forecast.append(val)
        return {
            'forecast': np.array(forecast),
            'accuracy': 0.92
        }

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastModel(Enum):
    PROPHET = "prophet"
    ARIMA = "arima"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"
    LINEAR_REGRESSION = "linear_regression"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"

class ForecastHorizon(Enum):
    SHORT_TERM = "short_term"  # 1-3 months
    MEDIUM_TERM = "medium_term"  # 3-12 months
    LONG_TERM = "long_term"  # 1-3 years

@dataclass
class ForecastResult:
    """Forecast result with confidence intervals"""
    model_type: ForecastModel
    forecast_horizon: ForecastHorizon
    forecast_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    dates: List[datetime]
    accuracy_metrics: Dict[str, float]
    model_confidence: float
    business_scenarios: Dict[str, List[float]]

@dataclass
class SeasonalityAnalysis:
    """Seasonality analysis result"""
    has_trend: bool
    trend_strength: float
    seasonal_periods: List[int]
    seasonal_strength: Dict[int, float]
    anomalies: List[Dict[str, Any]]
    change_points: List[datetime]

class AdvancedForecasting:
    """Advanced Time Series Forecasting System"""
    
    def __init__(self, db_path: str = "data/forecasting.db"):
        self.db_path = db_path
        self.mock_models = MockForecastModels()
        self.initialize_database()
        
    def initialize_database(self):
        """Initialize forecasting database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create forecasts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS forecasts (
            forecast_id TEXT PRIMARY KEY,
            metric_name TEXT,
            model_type TEXT,
            forecast_horizon TEXT,
            forecast_date TEXT,
            target_date TEXT,
            forecast_value REAL,
            confidence_lower REAL,
            confidence_upper REAL,
            model_confidence REAL,
            created_at TEXT
        )
        ''')
        
        # Create model performance table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            performance_id TEXT PRIMARY KEY,
            model_type TEXT,
            metric_name TEXT,
            mae REAL,
            mse REAL,
            mape REAL,
            accuracy_score REAL,
            evaluation_date TEXT
        )
        ''')
        
        # Create seasonality analysis table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS seasonality_analysis (
            analysis_id TEXT PRIMARY KEY,
            metric_name TEXT,
            has_trend INTEGER,
            trend_strength REAL,
            seasonal_periods TEXT,
            seasonal_strength TEXT,
            anomalies TEXT,
            change_points TEXT,
            analyzed_at TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def analyze_seasonality(self, data: pd.Series, metric_name: str) -> SeasonalityAnalysis:
        """Comprehensive seasonality and trend analysis"""
        # Trend analysis
        x = np.arange(len(data))
        slope, _, r_value, _, _ = stats.linregress(x, data)
        has_trend = abs(r_value) > 0.3
        trend_strength = abs(r_value)
        
        # Seasonal decomposition (simplified)
        seasonal_periods = self._detect_seasonal_periods(data)
        seasonal_strength = {}
        
        for period in seasonal_periods:
            if len(data) >= period * 2:
                seasonal_component = self._extract_seasonal_component(data, period)
                strength = np.std(seasonal_component) / np.std(data)
                seasonal_strength[period] = strength
        
        # Anomaly detection
        anomalies = self._detect_anomalies(data, metric_name)
        
        # Change point detection
        change_points = self._detect_change_points(data)
        
        analysis = SeasonalityAnalysis(
            has_trend=has_trend,
            trend_strength=trend_strength,
            seasonal_periods=seasonal_periods,
            seasonal_strength=seasonal_strength,
            anomalies=anomalies,
            change_points=change_points
        )
        
        # Save to database
        self._save_seasonality_analysis(metric_name, analysis)
        
        return analysis
    
    def _detect_seasonal_periods(self, data: pd.Series) -> List[int]:
        """Detect seasonal periods in time series"""
        # Common business periods
        candidates = [7, 30, 90, 365]  # Weekly, monthly, quarterly, yearly
        detected_periods = []
        
        for period in candidates:
            if len(data) >= period * 2:  # Need at least 2 cycles
                # Calculate autocorrelation at the period lag
                if len(data) > period:
                    corr = self._autocorrelation(data, period)
                    if corr > 0.3:  # Threshold for significant seasonality
                        detected_periods.append(period)
        
        return detected_periods
    
    def _autocorrelation(self, data: pd.Series, lag: int) -> float:
        """Calculate autocorrelation at specific lag"""
        if len(data) <= lag:
            return 0.0
        
        # Simplified autocorrelation calculation
        x1 = data[:-lag].values
        x2 = data[lag:].values
        
        if len(x1) == 0 or len(x2) == 0:
            return 0.0
        
        correlation = np.corrcoef(x1, x2)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _extract_seasonal_component(self, data: pd.Series, period: int) -> np.ndarray:
        """Extract seasonal component for given period"""
        n_cycles = len(data) // period
        if n_cycles == 0:
            return np.array([])
        
        # Reshape data into cycles and average
        cycles = []
        for i in range(n_cycles):
            cycle_data = data[i*period:(i+1)*period].values
            if len(cycle_data) == period:
                cycles.append(cycle_data)
        
        if cycles:
            seasonal_pattern = np.mean(cycles, axis=0)
            # Repeat pattern to match original data length
            full_seasonal = np.tile(seasonal_pattern, len(data) // period + 1)[:len(data)]
            return full_seasonal
        
        return np.zeros(len(data))
    
    def _detect_anomalies(self, data: pd.Series, metric_name: str) -> List[Dict[str, Any]]:
        """Detect anomalies in time series"""
        anomalies = []
        
        # Z-score based anomaly detection
        z_scores = np.abs(stats.zscore(data.dropna()))
        anomaly_threshold = 3
        
        anomaly_indices = np.where(z_scores > anomaly_threshold)[0]
        
        for idx in anomaly_indices:
            if idx < len(data):
                anomalies.append({
                    'index': int(idx),
                    'value': float(data.iloc[idx]),
                    'z_score': float(z_scores[idx]),
                    'severity': 'high' if z_scores[idx] > 4 else 'medium',
                    'type': 'statistical_outlier'
                })
        
        # Add trend-based anomalies
        if len(data) > 10:
            rolling_mean = data.rolling(window=5, center=True).mean()
            rolling_std = data.rolling(window=5, center=True).std()
            
            for i in range(len(data)):
                if pd.notna(rolling_mean.iloc[i]) and pd.notna(rolling_std.iloc[i]):
                    if abs(data.iloc[i] - rolling_mean.iloc[i]) > 2 * rolling_std.iloc[i]:
                        anomalies.append({
                            'index': i,
                            'value': float(data.iloc[i]),
                            'expected': float(rolling_mean.iloc[i]),
                            'deviation': float(abs(data.iloc[i] - rolling_mean.iloc[i])),
                            'severity': 'medium',
                            'type': 'trend_deviation'
                        })
        
        return anomalies[:10]  # Limit to top 10 anomalies
    
    def _detect_change_points(self, data: pd.Series) -> List[datetime]:
        """Detect structural change points in time series"""
        change_points = []
        
        if len(data) < 20:  # Need sufficient data
            return change_points
        
        # Sliding window approach to detect changes in mean/variance
        window_size = max(10, len(data) // 10)
        
        for i in range(window_size, len(data) - window_size):
            # Compare before and after windows
            before = data[i-window_size:i]
            after = data[i:i+window_size]
            
            # Statistical test for difference in means
            if len(before) > 0 and len(after) > 0:
                t_stat, p_value = stats.ttest_ind(before, after)
                
                if p_value < 0.01:  # Significant change
                    # Convert index to datetime if possible
                    if hasattr(data.index, 'to_pydatetime'):
                        change_date = data.index[i].to_pydatetime()
                    else:
                        change_date = datetime.now() - timedelta(days=len(data)-i)
                    
                    change_points.append(change_date)
        
        return change_points[:5]  # Limit to top 5 change points
    
    def generate_forecast(self, data: pd.Series, metric_name: str,
                         model_type: ForecastModel = ForecastModel.ENSEMBLE,
                         forecast_horizon: ForecastHorizon = ForecastHorizon.MEDIUM_TERM,
                         periods: int = 12) -> ForecastResult:
        """Generate comprehensive forecast with multiple models"""
        
        # Determine forecast periods based on horizon
        if forecast_horizon == ForecastHorizon.SHORT_TERM:
            periods = min(periods, 3)
        elif forecast_horizon == ForecastHorizon.MEDIUM_TERM:
            periods = min(periods, 12)
        else:  # LONG_TERM
            periods = min(periods, 36)
        
        # Generate forecasts with different models
        forecasts = {}
        
        if model_type == ForecastModel.ENSEMBLE or model_type == ForecastModel.PROPHET:
            forecasts['prophet'] = self.mock_models.prophet_forecast(data.values, periods)
        
        if model_type == ForecastModel.ENSEMBLE or model_type == ForecastModel.ARIMA:
            forecasts['arima'] = self.mock_models.arima_forecast(data.values, periods)
        
        if model_type == ForecastModel.ENSEMBLE or model_type == ForecastModel.LSTM:
            forecasts['lstm'] = self.mock_models.lstm_forecast(data.values, periods)
        
        # Ensemble forecast (if multiple models)
        if model_type == ForecastModel.ENSEMBLE and len(forecasts) > 1:
            ensemble_forecast = self._create_ensemble_forecast(forecasts)
            final_forecast = ensemble_forecast['forecast']
            confidence_intervals = list(zip(ensemble_forecast['lower'], ensemble_forecast['upper']))
            model_confidence = ensemble_forecast['confidence']
        else:
            # Use single model
            model_key = list(forecasts.keys())[0]
            model_result = forecasts[model_key]
            final_forecast = model_result['forecast']
            
            # Create confidence intervals
            if 'lower' in model_result and 'upper' in model_result:
                confidence_intervals = list(zip(model_result['lower'], model_result['upper']))
            else:
                # Generate confidence intervals based on historical volatility
                std_dev = np.std(data.values[-20:])  # Use last 20 points
                confidence_intervals = [(f - 1.96*std_dev, f + 1.96*std_dev) for f in final_forecast]
            
            model_confidence = model_result.get('confidence', model_result.get('accuracy', 0.8))
        
        # Generate forecast dates
        if hasattr(data.index, 'freq') and data.index.freq:
            last_date = data.index[-1]
            freq = data.index.freq
        else:
            last_date = datetime.now()
            freq = 'D'  # Default to daily
        
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                     periods=periods, freq=freq).tolist()
        
        # Calculate accuracy metrics (on historical data)
        accuracy_metrics = self._calculate_accuracy_metrics(data, model_type)
        
        # Generate business scenarios
        business_scenarios = self._generate_business_scenarios(final_forecast, data.values)
        
        result = ForecastResult(
            model_type=model_type,
            forecast_horizon=forecast_horizon,
            forecast_values=final_forecast.tolist(),
            confidence_intervals=confidence_intervals,
            dates=forecast_dates,
            accuracy_metrics=accuracy_metrics,
            model_confidence=model_confidence,
            business_scenarios=business_scenarios
        )
        
        # Save to database
        self._save_forecast_result(metric_name, result)
        
        return result
    
    def _create_ensemble_forecast(self, forecasts: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """Create ensemble forecast from multiple models"""
        # Simple averaging ensemble (in production, use weighted averaging based on accuracy)
        forecast_arrays = []
        confidence_scores = []
        
        for model_name, forecast_data in forecasts.items():
            forecast_arrays.append(forecast_data['forecast'])
            confidence_scores.append(forecast_data.get('confidence', forecast_data.get('accuracy', 0.8)))
        
        # Weighted average based on model confidence
        weights = np.array(confidence_scores) / np.sum(confidence_scores)
        ensemble_forecast = np.average(forecast_arrays, axis=0, weights=weights)
        
        # Calculate ensemble confidence intervals
        all_forecasts = np.array(forecast_arrays)
        ensemble_std = np.std(all_forecasts, axis=0)
        ensemble_lower = ensemble_forecast - 1.96 * ensemble_std
        ensemble_upper = ensemble_forecast + 1.96 * ensemble_std
        
        # Overall ensemble confidence
        ensemble_confidence = np.mean(confidence_scores)
        
        return {
            'forecast': ensemble_forecast,
            'lower': ensemble_lower,
            'upper': ensemble_upper,
            'confidence': ensemble_confidence
        }
    
    def _calculate_accuracy_metrics(self, data: pd.Series, model_type: ForecastModel) -> Dict[str, float]:
        """Calculate forecast accuracy metrics using historical data"""
        if len(data) < 10:
            return {'mae': 0.0, 'mse': 0.0, 'mape': 0.0, 'accuracy_score': 0.8}
        
        # Split data for validation
        split_point = int(len(data) * 0.8)
        train_data = data[:split_point]
        test_data = data[split_point:]
        
        if len(test_data) == 0:
            return {'mae': 0.0, 'mse': 0.0, 'mape': 0.0, 'accuracy_score': 0.8}
        
        # Generate forecast for test period
        if model_type == ForecastModel.PROPHET:
            forecast_result = self.mock_models.prophet_forecast(train_data.values, len(test_data))
            forecast = forecast_result['forecast']
        elif model_type == ForecastModel.ARIMA:
            forecast_result = self.mock_models.arima_forecast(train_data.values, len(test_data))
            forecast = forecast_result['forecast']
        elif model_type == ForecastModel.LSTM:
            forecast_result = self.mock_models.lstm_forecast(train_data.values, len(test_data))
            forecast = forecast_result['forecast']
        else:  # Ensemble
            forecasts = {
                'prophet': self.mock_models.prophet_forecast(train_data.values, len(test_data)),
                'arima': self.mock_models.arima_forecast(train_data.values, len(test_data)),
                'lstm': self.mock_models.lstm_forecast(train_data.values, len(test_data))
            }
            ensemble_result = self._create_ensemble_forecast(forecasts)
            forecast = ensemble_result['forecast']
        
        # Calculate metrics
        actual = test_data.values
        forecast = forecast[:len(actual)]  # Ensure same length
        
        mae = np.mean(np.abs(actual - forecast))
        mse = np.mean((actual - forecast) ** 2)
        mape = np.mean(np.abs((actual - forecast) / np.maximum(actual, 1e-8))) * 100
        
        # Accuracy score (1 - normalized MAPE)
        accuracy_score = max(0, 1 - mape / 100)
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'mape': float(mape),
            'accuracy_score': float(accuracy_score)
        }
    
    def _generate_business_scenarios(self, forecast: np.ndarray, historical_data: np.ndarray) -> Dict[str, List[float]]:
        """Generate business scenario forecasts"""
        scenarios = {}
        
        # Conservative scenario (10% lower growth)
        scenarios['conservative'] = (forecast * 0.9).tolist()
        
        # Optimistic scenario (20% higher growth)
        scenarios['optimistic'] = (forecast * 1.2).tolist()
        
        # Stress test scenario (based on worst historical performance)
        if len(historical_data) > 12:
            worst_period_ratio = np.min(historical_data[-12:]) / np.mean(historical_data[-12:])
            scenarios['stress_test'] = (forecast * worst_period_ratio).tolist()
        else:
            scenarios['stress_test'] = (forecast * 0.7).tolist()
        
        # Recovery scenario (gradual improvement)
        recovery_multiplier = np.linspace(0.8, 1.1, len(forecast))
        scenarios['recovery'] = (forecast * recovery_multiplier).tolist()
        
        return scenarios
    
    def _save_forecast_result(self, metric_name: str, result: ForecastResult):
        """Save forecast result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save individual forecast points
        for i, (date, value, conf_interval) in enumerate(zip(result.dates, result.forecast_values, result.confidence_intervals)):
            forecast_id = str(uuid.uuid4())
            cursor.execute('''
            INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                forecast_id, metric_name, result.model_type.value,
                result.forecast_horizon.value, datetime.now().isoformat(),
                date.isoformat(), value, conf_interval[0], conf_interval[1],
                result.model_confidence, datetime.now().isoformat()
            ))
        
        # Save model performance
        performance_id = str(uuid.uuid4())
        cursor.execute('''
        INSERT INTO model_performance VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            performance_id, result.model_type.value, metric_name,
            result.accuracy_metrics['mae'], result.accuracy_metrics['mse'],
            result.accuracy_metrics['mape'], result.accuracy_metrics['accuracy_score'],
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _save_seasonality_analysis(self, metric_name: str, analysis: SeasonalityAnalysis):
        """Save seasonality analysis to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        analysis_id = str(uuid.uuid4())
        cursor.execute('''
        INSERT INTO seasonality_analysis VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis_id, metric_name, int(analysis.has_trend),
            analysis.trend_strength, json.dumps(analysis.seasonal_periods),
            json.dumps(analysis.seasonal_strength), json.dumps(analysis.anomalies),
            json.dumps([cp.isoformat() for cp in analysis.change_points]),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def generate_forecasting_dashboard_data(self, metric_name: str) -> Dict[str, Any]:
        """Generate comprehensive dashboard data for forecasting"""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent forecasts
        forecasts_query = '''
        SELECT * FROM forecasts 
        WHERE metric_name = ? 
        ORDER BY created_at DESC 
        LIMIT 100
        '''
        forecasts_df = pd.read_sql_query(forecasts_query, conn, params=(metric_name,))
        
        # Get model performance
        performance_query = '''
        SELECT * FROM model_performance 
        WHERE metric_name = ? 
        ORDER BY evaluation_date DESC 
        LIMIT 10
        '''
        performance_df = pd.read_sql_query(performance_query, conn, params=(metric_name,))
        
        # Get seasonality analysis
        seasonality_query = '''
        SELECT * FROM seasonality_analysis 
        WHERE metric_name = ? 
        ORDER BY analyzed_at DESC 
        LIMIT 1
        '''
        seasonality_df = pd.read_sql_query(seasonality_query, conn, params=(metric_name,))
        
        conn.close()
        
        dashboard_data = {
            'metric_name': metric_name,
            'forecasts': forecasts_df.to_dict('records'),
            'model_performance': performance_df.to_dict('records'),
            'seasonality_analysis': seasonality_df.to_dict('records')[0] if not seasonality_df.empty else {},
            'summary_stats': {
                'total_forecasts': len(forecasts_df),
                'best_model': performance_df.loc[performance_df['accuracy_score'].idxmax()]['model_type'] if not performance_df.empty else 'ensemble',
                'avg_accuracy': performance_df['accuracy_score'].mean() if not performance_df.empty else 0.0,
                'forecast_horizon_days': len(forecasts_df[forecasts_df['model_type'] == 'ensemble']) if not forecasts_df.empty else 0
            },
            'generated_at': datetime.now().isoformat()
        }
        
        return dashboard_data

def create_sample_forecast_data():
    """Create sample time series data for forecasting demonstration"""
    # Generate sample subscription metrics
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
    
    # Generate base trends
    trend = np.linspace(1000, 1500, len(dates))
    
    # Add seasonality (yearly and weekly)
    yearly_seasonal = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    weekly_seasonal = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    
    # Add noise
    noise = np.random.normal(0, 30, len(dates))
    
    # Combine components
    values = trend + yearly_seasonal + weekly_seasonal + noise
    
    # Add some anomalies
    anomaly_indices = np.random.choice(len(dates), size=10, replace=False)
    values[anomaly_indices] *= np.random.uniform(0.5, 1.8, 10)
    
    # Create series
    ts_data = pd.Series(values, index=dates, name='daily_revenue')
    
    return ts_data

def run_forecasting_demo():
    """Run comprehensive forecasting demonstration"""
    forecasting = AdvancedForecasting()
    
    print("🔮 Running Advanced Forecasting Demo...")
    
    # Create sample data
    print("Generating sample time series data...")
    ts_data = create_sample_forecast_data()
    
    # Analyze seasonality
    print("Analyzing seasonality and trends...")
    seasonality_analysis = forecasting.analyze_seasonality(ts_data, "daily_revenue")
    
    print(f"✅ Seasonality Analysis:")
    print(f"  - Has trend: {seasonality_analysis.has_trend}")
    print(f"  - Trend strength: {seasonality_analysis.trend_strength:.3f}")
    print(f"  - Seasonal periods detected: {seasonality_analysis.seasonal_periods}")
    print(f"  - Anomalies found: {len(seasonality_analysis.anomalies)}")
    print(f"  - Change points: {len(seasonality_analysis.change_points)}")
    
    # Generate forecasts with different models
    models_to_test = [
        ForecastModel.PROPHET,
        ForecastModel.ARIMA,
        ForecastModel.LSTM,
        ForecastModel.ENSEMBLE
    ]
    
    forecast_results = {}
    
    for model in models_to_test:
        print(f"\nGenerating forecast with {model.value}...")
        result = forecasting.generate_forecast(
            ts_data, 
            "daily_revenue", 
            model_type=model,
            forecast_horizon=ForecastHorizon.MEDIUM_TERM,
            periods=30
        )
        forecast_results[model.value] = result
        
        print(f"✅ {model.value} forecast complete:")
        print(f"  - Model confidence: {result.model_confidence:.1%}")
        print(f"  - Accuracy score: {result.accuracy_metrics['accuracy_score']:.1%}")
        print(f"  - MAPE: {result.accuracy_metrics['mape']:.2f}%")
        print(f"  - Forecast range: ${result.forecast_values[0]:.0f} - ${result.forecast_values[-1]:.0f}")
    
    # Generate dashboard data
    dashboard_data = forecasting.generate_forecasting_dashboard_data("daily_revenue")
    
    # Display results summary
    print("\n" + "="*60)
    print("📊 FORECASTING RESULTS SUMMARY")
    print("="*60)
    
    print(f"Metric: {dashboard_data['metric_name']}")
    print(f"Total forecasts generated: {dashboard_data['summary_stats']['total_forecasts']}")
    print(f"Best performing model: {dashboard_data['summary_stats']['best_model']}")
    print(f"Average accuracy: {dashboard_data['summary_stats']['avg_accuracy']:.1%}")
    
    print("\n🎯 Business Scenarios (30-day forecast):")
    ensemble_result = forecast_results['ensemble']
    for scenario, values in ensemble_result.business_scenarios.items():
        avg_value = np.mean(values)
        print(f"  - {scenario.title()}: ${avg_value:.0f} average daily revenue")
    
    print("\n💡 Key Insights:")
    if seasonality_analysis.has_trend:
        print("  - Strong trend detected - consider trend-following strategies")
    if seasonality_analysis.seasonal_periods:
        print(f"  - Seasonal patterns found - optimize for {seasonality_analysis.seasonal_periods} day cycles")
    if len(seasonality_analysis.anomalies) > 5:
        print("  - High anomaly frequency - implement robust monitoring")
    if ensemble_result.model_confidence > 0.8:
        print("  - High forecast confidence - suitable for strategic planning")
    
    print("="*60)
    
    return forecast_results, seasonality_analysis, dashboard_data

if __name__ == "__main__":
    run_forecasting_demo()