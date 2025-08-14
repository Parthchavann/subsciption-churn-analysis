"""
Advanced Churn Analysis and Prediction Model
Netflix/Spotify style subscription analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ChurnAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load all datasets"""
        try:
            self.users = pd.read_csv('../data/raw/users.csv')
            self.subscriptions = pd.read_csv('../data/raw/subscriptions.csv')
            self.engagement = pd.read_csv('../data/raw/engagement.csv')
            self.tickets = pd.read_csv('../data/raw/support_tickets.csv')
            
            # Convert dates
            self.users['signup_date'] = pd.to_datetime(self.users['signup_date'])
            self.subscriptions['billing_date'] = pd.to_datetime(self.subscriptions['billing_date'])
            self.engagement['month'] = pd.to_datetime(self.engagement['month'])
            
            if 'churn_date' in self.subscriptions.columns:
                self.subscriptions['churn_date'] = pd.to_datetime(self.subscriptions['churn_date'])
            
            print("Data loaded successfully!")
            print(f"Users: {len(self.users):,}")
            print(f"Subscriptions: {len(self.subscriptions):,}")
            print(f"Engagement records: {len(self.engagement):,}")
            print(f"Support tickets: {len(self.tickets):,}")
            
        except FileNotFoundError:
            print("Data files not found. Please run data_generation.py first.")
            return False
        return True
    
    def create_cohort_analysis(self):
        """Create cohort retention analysis"""
        # Merge user signup data with subscriptions
        user_cohorts = self.users[['user_id', 'signup_date']].copy()
        user_cohorts['cohort_month'] = user_cohorts['signup_date'].dt.to_period('M')
        
        # Add cohort info to subscriptions
        subs_cohort = self.subscriptions.merge(user_cohorts, on='user_id')
        subs_cohort['billing_month'] = subs_cohort['billing_date'].dt.to_period('M')
        
        # Calculate months since cohort start
        subs_cohort['months_since_signup'] = (
            subs_cohort['billing_month'] - subs_cohort['cohort_month']
        ).apply(lambda x: x.n)
        
        # Create cohort table
        cohort_data = subs_cohort.groupby(['cohort_month', 'months_since_signup'])['user_id'].nunique().reset_index()
        cohort_sizes = subs_cohort.groupby('cohort_month')['user_id'].nunique()
        
        cohort_table = cohort_data.pivot(index='cohort_month', 
                                        columns='months_since_signup', 
                                        values='user_id')
        
        # Calculate retention rates
        cohort_retention = cohort_table.divide(cohort_sizes, axis=0)
        
        return cohort_retention
    
    def analyze_churn_patterns(self):
        """Analyze churn patterns and trends"""
        
        # Monthly churn rate
        monthly_churn = self.subscriptions.groupby(
            self.subscriptions['billing_date'].dt.to_period('M')
        ).agg({
            'user_id': 'count',
            'status': lambda x: (x == 'churned').sum()
        }).reset_index()
        
        monthly_churn['churn_rate'] = monthly_churn['status'] / monthly_churn['user_id'] * 100
        monthly_churn['billing_date'] = monthly_churn['billing_date'].astype(str)
        
        # Churn by demographics
        user_status = self.subscriptions.groupby('user_id')['status'].last().reset_index()
        churn_demo = self.users.merge(user_status, on='user_id')
        churn_demo['churned'] = (churn_demo['status'] == 'churned').astype(int)
        
        # Churn analysis by different segments
        churn_by_age = churn_demo.groupby(pd.cut(churn_demo['age'], bins=[0, 25, 35, 50, 100]))['churned'].mean() * 100
        churn_by_channel = churn_demo.groupby('acquisition_channel')['churned'].mean() * 100
        churn_by_device = churn_demo.groupby('device_type')['churned'].mean() * 100
        
        return {
            'monthly_churn': monthly_churn,
            'churn_by_age': churn_by_age,
            'churn_by_channel': churn_by_channel,
            'churn_by_device': churn_by_device
        }
    
    def create_feature_matrix(self):
        """Create feature matrix for machine learning"""
        
        # User features
        features = self.users.copy()
        
        # Subscription features
        sub_features = self.subscriptions.groupby('user_id').agg({
            'price': ['mean', 'sum', 'count'],
            'billing_date': 'max'
        }).round(2)
        sub_features.columns = ['avg_price', 'total_spent', 'subscription_months', 'last_billing']
        
        # Engagement features
        eng_features = self.engagement.groupby('user_id').agg({
            'login_days': ['mean', 'sum'],
            'hours_watched': ['mean', 'sum'],
            'content_items': ['mean', 'sum'],
            'engagement_score': 'mean'
        }).round(2)
        eng_features.columns = ['avg_login_days', 'total_login_days', 
                               'avg_hours_watched', 'total_hours_watched',
                               'avg_content_items', 'total_content_items',
                               'avg_engagement_score']
        
        # Support ticket features
        ticket_features = self.tickets.groupby('user_id').agg({
            'ticket_id': 'count',
            'satisfaction_score': 'mean',
            'resolution_time_hours': 'mean'
        }).round(2)
        ticket_features.columns = ['num_tickets', 'avg_satisfaction', 'avg_resolution_time']
        
        # Merge all features
        features = features.merge(sub_features, on='user_id', how='left')
        features = features.merge(eng_features, on='user_id', how='left')
        features = features.merge(ticket_features, on='user_id', how='left')
        
        # Fill missing values
        features = features.fillna(0)
        
        # Target variable (churn)
        user_churn = self.subscriptions.groupby('user_id')['status'].last()
        features['churned'] = features['user_id'].map(user_churn == 'churned').astype(int)
        
        # Calculate customer lifetime
        features['customer_lifetime_days'] = (
            pd.to_datetime(features['last_billing']) - features['signup_date']
        ).dt.days.fillna(0)
        
        return features
    
    def build_churn_model(self, features_df):
        """Build machine learning model to predict churn"""
        
        # Select features for modeling
        numeric_features = ['age', 'avg_price', 'total_spent', 'subscription_months',
                           'customer_lifetime_days', 'avg_engagement_score', 
                           'total_hours_watched', 'total_login_days', 'num_tickets']
        
        categorical_features = ['gender', 'acquisition_channel', 'device_type', 'payment_method']
        
        # Prepare data
        X = features_df[numeric_features + categorical_features].copy()
        y = features_df['churned']
        
        # Encode categorical variables
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        # Scale numeric features
        X[numeric_features] = self.scaler.fit_transform(X[numeric_features])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.model.fit(X_train, y_train)
        
        # Predictions and evaluation
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': numeric_features + categorical_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Model performance
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results = {
            'model': self.model,
            'feature_importance': feature_importance,
            'auc_score': auc_score,
            'classification_report': classification_report(y_test, y_pred),
            'test_predictions': y_pred_proba,
            'test_actual': y_test
        }
        
        return results
    
    def visualize_insights(self, churn_patterns, feature_importance):
        """Create comprehensive visualizations"""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Monthly Churn Rate', 'Churn by Age Group', 
                          'Churn by Acquisition Channel', 'Churn by Device Type',
                          'Feature Importance', 'Customer Segments'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Monthly churn trend
        monthly_data = churn_patterns['monthly_churn']
        fig.add_trace(
            go.Scatter(x=monthly_data['billing_date'], 
                      y=monthly_data['churn_rate'],
                      name='Churn Rate %',
                      line=dict(color='red', width=3)),
            row=1, col=1
        )
        
        # Churn by age
        age_data = churn_patterns['churn_by_age'].reset_index()
        fig.add_trace(
            go.Bar(x=[str(x) for x in age_data['age']], 
                   y=age_data['churned'],
                   name='Churn Rate by Age',
                   marker_color='lightcoral'),
            row=1, col=2
        )
        
        # Churn by channel
        channel_data = churn_patterns['churn_by_channel'].reset_index()
        fig.add_trace(
            go.Bar(x=channel_data['acquisition_channel'], 
                   y=channel_data['churned'],
                   name='Churn Rate by Channel',
                   marker_color='skyblue'),
            row=2, col=1
        )
        
        # Churn by device
        device_data = churn_patterns['churn_by_device'].reset_index()
        fig.add_trace(
            go.Bar(x=device_data['device_type'], 
                   y=device_data['churned'],
                   name='Churn Rate by Device',
                   marker_color='lightgreen'),
            row=2, col=2
        )
        
        # Feature importance
        top_features = feature_importance.head(10)
        fig.add_trace(
            go.Bar(x=top_features['importance'], 
                   y=top_features['feature'],
                   orientation='h',
                   name='Feature Importance',
                   marker_color='orange'),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Subscription Churn Analysis Dashboard",
            showlegend=False
        )
        
        fig.show()
        
        # Save as HTML
        fig.write_html("../presentation/churn_analysis_dashboard.html")
        
        return fig
    
    def calculate_business_impact(self, features_df):
        """Calculate business impact of churn reduction strategies"""
        
        # Current metrics
        total_users = len(features_df)
        churned_users = features_df['churned'].sum()
        active_users = total_users - churned_users
        current_churn_rate = churned_users / total_users * 100
        
        # Revenue metrics
        avg_revenue_per_user = features_df['total_spent'].mean()
        total_current_revenue = features_df['total_spent'].sum()
        
        # Scenario: Reduce churn by 10%
        target_churn_reduction = 0.10
        users_saved = int(churned_users * target_churn_reduction)
        new_churn_rate = (churned_users - users_saved) / total_users * 100
        
        # Revenue impact
        additional_revenue = users_saved * avg_revenue_per_user
        revenue_increase_pct = additional_revenue / total_current_revenue * 100
        
        # Scenario: Increase revenue by 5% through price optimization
        price_increase = 0.05
        revenue_from_price_increase = total_current_revenue * price_increase
        
        impact_summary = {
            'current_metrics': {
                'total_users': total_users,
                'churned_users': churned_users,
                'current_churn_rate': round(current_churn_rate, 2),
                'current_revenue': round(total_current_revenue, 2),
                'avg_revenue_per_user': round(avg_revenue_per_user, 2)
            },
            'churn_reduction_scenario': {
                'users_saved': users_saved,
                'new_churn_rate': round(new_churn_rate, 2),
                'additional_revenue': round(additional_revenue, 2),
                'revenue_increase_pct': round(revenue_increase_pct, 2)
            },
            'price_optimization_scenario': {
                'price_increase_pct': price_increase * 100,
                'additional_revenue': round(revenue_from_price_increase, 2),
                'total_revenue_impact': round(additional_revenue + revenue_from_price_increase, 2)
            }
        }
        
        return impact_summary
    
    def run_full_analysis(self):
        """Execute complete churn analysis pipeline"""
        
        print("=== SUBSCRIPTION CHURN ANALYSIS ===")
        print("Loading data...")
        
        if not self.load_data():
            return
        
        print("\n1. Creating cohort analysis...")
        cohort_retention = self.create_cohort_analysis()
        
        print("2. Analyzing churn patterns...")
        churn_patterns = self.analyze_churn_patterns()
        
        print("3. Building feature matrix...")
        features_df = self.create_feature_matrix()
        
        print("4. Training churn prediction model...")
        model_results = self.build_churn_model(features_df)
        
        print(f"Model AUC Score: {model_results['auc_score']:.3f}")
        
        print("\n5. Calculating business impact...")
        business_impact = self.calculate_business_impact(features_df)
        
        print("\n=== BUSINESS IMPACT ANALYSIS ===")
        print(f"Current Churn Rate: {business_impact['current_metrics']['current_churn_rate']}%")
        print(f"Users at Risk: {business_impact['current_metrics']['churned_users']:,}")
        print(f"Current Revenue: ${business_impact['current_metrics']['current_revenue']:,.2f}")
        
        print("\n--- 10% Churn Reduction Scenario ---")
        print(f"Users Saved: {business_impact['churn_reduction_scenario']['users_saved']:,}")
        print(f"Additional Revenue: ${business_impact['churn_reduction_scenario']['additional_revenue']:,.2f}")
        print(f"Revenue Increase: {business_impact['churn_reduction_scenario']['revenue_increase_pct']:.1f}%")
        
        print("\n6. Creating visualizations...")
        dashboard_fig = self.visualize_insights(churn_patterns, model_results['feature_importance'])
        
        print("\n=== TOP CHURN PREDICTION FEATURES ===")
        for idx, row in model_results['feature_importance'].head(5).iterrows():
            print(f"{row['feature']}: {row['importance']:.3f}")
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("Dashboard saved to: ../presentation/churn_analysis_dashboard.html")
        
        return {
            'cohort_retention': cohort_retention,
            'churn_patterns': churn_patterns,
            'model_results': model_results,
            'business_impact': business_impact,
            'dashboard': dashboard_fig
        }

if __name__ == "__main__":
    analyzer = ChurnAnalyzer()
    results = analyzer.run_full_analysis()