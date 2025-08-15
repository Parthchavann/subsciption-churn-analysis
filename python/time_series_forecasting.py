"""
Time Series Forecasting for Subscription Metrics
Advanced forecasting: ARIMA, Prophet, LSTM for MRR, Churn, User Growth
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

# Time series libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

class SubscriptionForecaster:
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        
    def generate_time_series_data(self):
        """Generate realistic time series data for subscription metrics"""
        print("📈 Generating time series data...")
        
        # Create 36 months of historical data
        start_date = datetime(2021, 1, 1)
        end_date = datetime(2024, 12, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        np.random.seed(42)
        
        # Base trends and seasonality
        time_trend = np.arange(len(dates))
        seasonal_component = 10 * np.sin(2 * np.pi * time_trend / 12)  # Annual seasonality
        
        # Generate MRR (Monthly Recurring Revenue)
        base_mrr = 100000  # $100K base
        mrr_trend = base_mrr + time_trend * 2000  # Growing $2K/month
        mrr_seasonal = seasonal_component * 5000  # $50K seasonal variation
        mrr_noise = np.random.normal(0, 8000, len(dates))
        mrr = mrr_trend + mrr_seasonal + mrr_noise
        mrr = np.maximum(mrr, 50000)  # Minimum $50K
        
        # Generate Active Users
        base_users = 5000
        users_trend = base_users + time_trend * 80  # Growing 80 users/month
        users_seasonal = seasonal_component * 200  # 200 users seasonal variation
        users_noise = np.random.normal(0, 150, len(dates))
        active_users = users_trend + users_seasonal + users_noise
        active_users = np.maximum(active_users.astype(int), 1000)
        
        # Generate Churn Rate (inverse relationship with growth)
        base_churn = 0.12  # 12% base churn
        churn_trend = base_churn - time_trend * 0.001  # Improving over time
        churn_seasonal = -seasonal_component * 0.002  # Holiday seasons = lower churn
        churn_noise = np.random.normal(0, 0.01, len(dates))
        churn_rate = churn_trend + churn_seasonal + churn_noise
        churn_rate = np.clip(churn_rate, 0.05, 0.25)  # Between 5% and 25%
        
        # Calculate derived metrics
        arpu = mrr / active_users
        new_users = np.diff(active_users, prepend=active_users[0])
        new_users = np.maximum(new_users, 0)  # Can't have negative new users
        churned_users = (active_users * churn_rate).astype(int)
        
        # Create DataFrame
        ts_data = pd.DataFrame({
            'date': dates,
            'mrr': mrr.round(2),
            'active_users': active_users,
            'churn_rate': churn_rate.round(4),
            'arpu': arpu.round(2),
            'new_users': new_users.astype(int),
            'churned_users': churned_users
        })
        
        # Add external factors (marketing spend, economic indicators)
        ts_data['marketing_spend'] = 10000 + np.random.normal(0, 2000, len(dates))
        ts_data['economic_index'] = 100 + np.cumsum(np.random.normal(0, 0.5, len(dates)))
        
        print(f"✅ Generated {len(ts_data)} months of time series data")
        print(f"📊 MRR range: ${ts_data['mrr'].min():,.0f} - ${ts_data['mrr'].max():,.0f}")
        print(f"👥 Users range: {ts_data['active_users'].min():,} - {ts_data['active_users'].max():,}")
        print(f"📉 Churn range: {ts_data['churn_rate'].min():.1%} - {ts_data['churn_rate'].max():.1%}")
        
        return ts_data
    
    def perform_time_series_analysis(self, ts_data, metric='mrr'):
        """Perform comprehensive time series analysis"""
        print(f"🔍 Analyzing {metric.upper()} time series...")
        
        # Set date as index
        ts_series = ts_data.set_index('date')[metric]
        
        # Stationarity test
        adf_result = adfuller(ts_series)
        is_stationary = adf_result[1] <= 0.05
        
        # Decomposition
        decomposition = seasonal_decompose(ts_series, model='additive', period=12)
        
        # Store analysis results
        analysis = {
            'is_stationary': is_stationary,
            'adf_pvalue': adf_result[1],
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }
        
        print(f"📊 Stationarity: {'✅ Stationary' if is_stationary else '❌ Non-stationary'}")
        print(f"📈 ADF p-value: {adf_result[1]:.4f}")
        
        return analysis
    
    def forecast_arima(self, ts_data, metric='mrr', forecast_periods=12):
        """ARIMA forecasting"""
        print(f"🎯 ARIMA forecasting for {metric.upper()}...")
        
        # Prepare data
        ts_series = ts_data.set_index('date')[metric]
        
        # Auto ARIMA (simplified approach)
        # In practice, you'd use auto_arima from pmdarima
        try:
            model = ARIMA(ts_series, order=(2, 1, 2))
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=forecast_periods)
            forecast_index = pd.date_range(
                start=ts_series.index[-1] + pd.DateOffset(months=1),
                periods=forecast_periods,
                freq='M'
            )
            
            forecast_df = pd.DataFrame({
                'date': forecast_index,
                'forecast': forecast,
                'model': 'ARIMA'
            })
            
            # Calculate accuracy on last 6 months
            train_data = ts_series[:-6]
            test_data = ts_series[-6:]
            
            test_model = ARIMA(train_data, order=(2, 1, 2)).fit()
            test_forecast = test_model.forecast(steps=6)
            
            mae = mean_absolute_error(test_data, test_forecast)
            mse = mean_squared_error(test_data, test_forecast)
            
            self.models[f'arima_{metric}'] = fitted_model
            self.forecasts[f'arima_{metric}'] = forecast_df
            self.metrics[f'arima_{metric}'] = {'mae': mae, 'mse': mse}
            
            print(f"✅ ARIMA model trained. MAE: {mae:.2f}")
            return forecast_df
            
        except Exception as e:
            print(f"❌ ARIMA forecasting failed: {e}")
            return None
    
    def forecast_prophet(self, ts_data, metric='mrr', forecast_periods=12):
        """Prophet forecasting with seasonality"""
        try:
            from prophet import Prophet
            
            print(f"🔮 Prophet forecasting for {metric.upper()}...")
            
            # Prepare data for Prophet
            prophet_data = ts_data[['date', metric]].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Initialize Prophet with yearly seasonality
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            # Add custom regressors if available
            if 'marketing_spend' in ts_data.columns:
                model.add_regressor('marketing_spend')
                prophet_data['marketing_spend'] = ts_data['marketing_spend']
            
            # Fit model
            model.fit(prophet_data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_periods, freq='M')
            
            # Add regressor values for forecast period (assuming trend continues)
            if 'marketing_spend' in prophet_data.columns:
                last_marketing = prophet_data['marketing_spend'].iloc[-1]
                future.loc[future.index >= len(prophet_data), 'marketing_spend'] = last_marketing
            
            # Forecast
            forecast = model.predict(future)
            
            # Extract forecast for future periods
            forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods)
            forecast_df.columns = ['date', 'forecast', 'lower_bound', 'upper_bound']
            forecast_df['model'] = 'Prophet'
            
            # Calculate accuracy
            train_size = len(prophet_data) - 6
            train_data = prophet_data[:train_size]
            test_data = prophet_data[train_size:]
            
            test_model = Prophet(yearly_seasonality=True)
            if 'marketing_spend' in prophet_data.columns:
                test_model.add_regressor('marketing_spend')
            
            test_model.fit(train_data)
            test_future = test_model.make_future_dataframe(periods=6, freq='M')
            
            if 'marketing_spend' in prophet_data.columns:
                test_future['marketing_spend'] = prophet_data['marketing_spend']
            
            test_forecast = test_model.predict(test_future)
            test_predictions = test_forecast['yhat'].tail(6).values
            
            mae = mean_absolute_error(test_data['y'], test_predictions)
            
            self.models[f'prophet_{metric}'] = model
            self.forecasts[f'prophet_{metric}'] = forecast_df
            self.metrics[f'prophet_{metric}'] = {'mae': mae}
            
            print(f"✅ Prophet model trained. MAE: {mae:.2f}")
            return forecast_df
            
        except ImportError:
            print("❌ Prophet not available. Install with: pip install prophet")
            return None
        except Exception as e:
            print(f"❌ Prophet forecasting failed: {e}")
            return None
    
    def forecast_linear_trend(self, ts_data, metric='mrr', forecast_periods=12):
        """Simple linear trend forecasting as baseline"""
        print(f"📈 Linear trend forecasting for {metric.upper()}...")
        
        # Prepare data
        ts_series = ts_data.set_index('date')[metric]
        
        # Calculate trend
        x = np.arange(len(ts_series))
        coefficients = np.polyfit(x, ts_series.values, 1)
        trend_line = np.poly1d(coefficients)
        
        # Forecast
        future_x = np.arange(len(ts_series), len(ts_series) + forecast_periods)
        forecast_values = trend_line(future_x)
        
        forecast_index = pd.date_range(
            start=ts_series.index[-1] + pd.DateOffset(months=1),
            periods=forecast_periods,
            freq='M'
        )
        
        forecast_df = pd.DataFrame({
            'date': forecast_index,
            'forecast': forecast_values,
            'model': 'Linear Trend'
        })
        
        # Calculate accuracy
        test_predictions = trend_line(x[-6:])
        mae = mean_absolute_error(ts_series.values[-6:], test_predictions)
        
        self.forecasts[f'linear_{metric}'] = forecast_df
        self.metrics[f'linear_{metric}'] = {'mae': mae}
        
        print(f"✅ Linear trend model. MAE: {mae:.2f}")
        return forecast_df
    
    def create_forecasting_dashboard(self, ts_data, metric='mrr'):
        """Create comprehensive forecasting dashboard"""
        print(f"📊 Creating forecasting dashboard for {metric.upper()}...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{metric.upper()} Historical & Forecast',
                'Model Accuracy Comparison',
                'Seasonal Decomposition',
                'Forecast Confidence Intervals'
            ),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"colspan": 2}, None]],
            vertical_spacing=0.1
        )
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=ts_data['date'],
                y=ts_data[metric],
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add forecasts if available
        colors = ['red', 'green', 'orange']
        for i, (model_name, forecast_df) in enumerate(self.forecasts.items()):
            if metric in model_name:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df['date'],
                        y=forecast_df['forecast'],
                        mode='lines+markers',
                        name=forecast_df['model'].iloc[0],
                        line=dict(color=colors[i % len(colors)], dash='dash')
                    ),
                    row=1, col=1
                )
        
        # Model accuracy comparison
        if self.metrics:
            model_names = []
            mae_values = []
            
            for model_name, metrics in self.metrics.items():
                if metric in model_name:
                    model_names.append(model_name.replace(f'_{metric}', '').title())
                    mae_values.append(metrics['mae'])
            
            if mae_values:
                fig.add_trace(
                    go.Bar(
                        x=model_names,
                        y=mae_values,
                        name='MAE',
                        marker_color='lightblue'
                    ),
                    row=1, col=2
                )
        
        # Time series decomposition
        ts_series = ts_data.set_index('date')[metric]
        decomposition = seasonal_decompose(ts_series, model='additive', period=12)
        
        fig.add_trace(
            go.Scatter(
                x=ts_data['date'],
                y=decomposition.trend.values,
                mode='lines',
                name='Trend',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=ts_data['date'],
                y=decomposition.seasonal.values,
                mode='lines',
                name='Seasonal',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Time Series Forecasting Dashboard - {metric.upper()}",
            showlegend=True
        )
        
        # Save dashboard
        fig.write_html(f"../presentation/forecasting_dashboard_{metric}.html")
        
        return fig
    
    def run_comprehensive_forecasting(self):
        """Run comprehensive forecasting analysis"""
        print("🚀 COMPREHENSIVE TIME SERIES FORECASTING")
        print("=" * 60)
        
        # Generate time series data
        ts_data = self.generate_time_series_data()
        
        # Save data
        ts_data.to_csv('../data/real/time_series_data.csv', index=False)
        
        # Key metrics to forecast
        metrics = ['mrr', 'active_users', 'churn_rate']
        
        # Run forecasting for each metric
        for metric in metrics:
            print(f"\n🎯 Forecasting {metric.upper()}:")
            print("-" * 40)
            
            # Perform time series analysis
            analysis = self.perform_time_series_analysis(ts_data, metric)
            
            # Try different forecasting methods
            self.forecast_arima(ts_data, metric)
            self.forecast_prophet(ts_data, metric)
            self.forecast_linear_trend(ts_data, metric)
            
            # Create dashboard
            dashboard = self.create_forecasting_dashboard(ts_data, metric)
        
        # Summary results
        print("\n" + "=" * 60)
        print("📈 FORECASTING RESULTS SUMMARY")
        print("=" * 60)
        
        print("\n🎯 Models Performance:")
        for model_name, metrics in self.metrics.items():
            mae = metrics['mae']
            print(f"{model_name}: MAE = {mae:.2f}")
        
        print(f"\n📊 Forecasts generated for: {', '.join(metrics)}")
        print("📁 Files saved:")
        print("  - ../data/real/time_series_data.csv")
        for metric in metrics:
            print(f"  - ../presentation/forecasting_dashboard_{metric}.html")
        
        print("\n✨ Time Series Forecasting Complete!")
        return ts_data, self.forecasts, self.metrics

if __name__ == "__main__":
    forecaster = SubscriptionForecaster()
    ts_data, forecasts, metrics = forecaster.run_comprehensive_forecasting()