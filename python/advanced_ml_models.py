"""
Advanced Machine Learning Models for Subscription Churn Prediction
Enterprise-level models: XGBoost, Neural Networks, Ensemble Methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
import shap
import warnings
warnings.filterwarnings('ignore')

class AdvancedChurnPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        
    def load_and_prepare_data(self, dataset_path=None, dataset_name='subscription'):
        """Load and prepare data for advanced modeling"""
        
        if dataset_path is None:
            # Load the real subscription dataset we created
            try:
                df = pd.read_csv('../data/real/subscription_churn_real_patterns.csv')
                print(f"✅ Loaded {dataset_name} dataset: {len(df):,} customers")
            except FileNotFoundError:
                print("❌ Dataset not found. Please run real_data_loader.py first")
                return None
        else:
            df = pd.read_csv(dataset_path)
        
        # Data preprocessing
        print("🔄 Preprocessing data for advanced models...")
        
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        # Feature engineering
        df['RevenuePerMonth'] = df['TotalRevenue'] / df['TenureMonths']
        df['UsagePerDollar'] = df['MonthlyUsageHours'] / df['MonthlyPrice']
        df['SupportContactsPerMonth'] = df['SupportContacts'] / df['TenureMonths']
        
        # Interaction features
        df['HighValueCustomer'] = ((df['MonthlyPrice'] > df['MonthlyPrice'].quantile(0.75)) & 
                                  (df['TenureMonths'] > 6)).astype(int)
        df['RiskScore'] = (df['SupportContacts'] * 0.3 + 
                          (5 - df['SatisfactionScore']) * 0.4 + 
                          df['PaymentFailures'] * 0.3)
        
        # Categorical encoding
        categorical_features = ['Gender', 'Country', 'SubscriptionPlan', 'PrimaryDevice', 'AcquisitionChannel']
        
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        # Prepare features and target
        feature_columns = [
            'Age', 'MonthlyPrice', 'TenureMonths', 'MonthlyUsageHours', 
            'FeatureUsageRate', 'SupportContacts', 'SatisfactionScore', 
            'PaymentFailures', 'RevenuePerMonth', 'UsagePerDollar', 
            'SupportContactsPerMonth', 'HighValueCustomer', 'RiskScore'
        ] + [f'{col}_encoded' for col in categorical_features]
        
        # Add AutoRenewal as numeric
        df['AutoRenewal_num'] = df['AutoRenewal'].astype(int)
        feature_columns.append('AutoRenewal_num')
        
        X = df[feature_columns]
        y = (df['Churn'] == 'Yes').astype(int)
        
        print(f"📊 Features: {len(feature_columns)}")
        print(f"🎯 Churn rate: {y.mean()*100:.1f}%")
        
        return X, y, df
    
    def train_xgboost_model(self, X_train, X_test, y_train, y_test):
        """Train XGBoost model with hyperparameter tuning"""
        print("🚀 Training XGBoost model...")
        
        # XGBoost with hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            n_jobs=-1
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=5, 
            scoring='roc_auc', n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        best_xgb = grid_search.best_estimator_
        
        # Predictions
        y_pred_xgb = best_xgb.predict(X_test)
        y_pred_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
        
        # Performance metrics
        performance = {
            'accuracy': accuracy_score(y_test, y_pred_xgb),
            'f1': f1_score(y_test, y_pred_xgb),
            'auc': roc_auc_score(y_test, y_pred_proba_xgb),
            'best_params': grid_search.best_params_
        }
        
        self.models['xgboost'] = best_xgb
        self.model_performance['xgboost'] = performance
        
        print(f"✅ XGBoost AUC: {performance['auc']:.4f}")
        return best_xgb, performance
    
    def train_neural_network(self, X_train, X_test, y_train, y_test):
        """Train Neural Network model"""
        print("🧠 Training Neural Network...")
        
        # Scale features for neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['neural_network'] = scaler
        
        # Neural network with architecture tuning
        param_grid = {
            'hidden_layer_sizes': [(100, 50), (150, 75), (200, 100, 50)],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'alpha': [0.0001, 0.001, 0.01]
        }
        
        mlp = MLPClassifier(
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        grid_search = GridSearchCV(
            mlp, param_grid, cv=3, 
            scoring='roc_auc', n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        best_mlp = grid_search.best_estimator_
        
        # Predictions
        y_pred_mlp = best_mlp.predict(X_test_scaled)
        y_pred_proba_mlp = best_mlp.predict_proba(X_test_scaled)[:, 1]
        
        performance = {
            'accuracy': accuracy_score(y_test, y_pred_mlp),
            'f1': f1_score(y_test, y_pred_mlp),
            'auc': roc_auc_score(y_test, y_pred_proba_mlp),
            'best_params': grid_search.best_params_
        }
        
        self.models['neural_network'] = best_mlp
        self.model_performance['neural_network'] = performance
        
        print(f"✅ Neural Network AUC: {performance['auc']:.4f}")
        return best_mlp, performance
    
    def train_lightgbm_model(self, X_train, X_test, y_train, y_test):
        """Train LightGBM model"""
        print("💡 Training LightGBM model...")
        
        # LightGBM parameters
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model
        lgb_model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # Predictions
        y_pred_proba_lgb = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
        y_pred_lgb = (y_pred_proba_lgb > 0.5).astype(int)
        
        performance = {
            'accuracy': accuracy_score(y_test, y_pred_lgb),
            'f1': f1_score(y_test, y_pred_lgb),
            'auc': roc_auc_score(y_test, y_pred_proba_lgb)
        }
        
        self.models['lightgbm'] = lgb_model
        self.model_performance['lightgbm'] = performance
        
        print(f"✅ LightGBM AUC: {performance['auc']:.4f}")
        return lgb_model, performance
    
    def create_ensemble_model(self, X_train, X_test, y_train, y_test):
        """Create ensemble model combining multiple algorithms"""
        print("🎭 Creating Ensemble Model...")
        
        # Base models for ensemble
        models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ]
        
        # Voting classifier
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        
        # Predictions
        y_pred_ensemble = ensemble.predict(X_test)
        y_pred_proba_ensemble = ensemble.predict_proba(X_test)[:, 1]
        
        performance = {
            'accuracy': accuracy_score(y_test, y_pred_ensemble),
            'f1': f1_score(y_test, y_pred_ensemble),
            'auc': roc_auc_score(y_test, y_pred_proba_ensemble)
        }
        
        self.models['ensemble'] = ensemble
        self.model_performance['ensemble'] = performance
        
        print(f"✅ Ensemble AUC: {performance['auc']:.4f}")
        return ensemble, performance
    
    def analyze_feature_importance(self, X, model_name='xgboost'):
        """Analyze feature importance using SHAP"""
        print(f"🔍 Analyzing feature importance for {model_name}...")
        
        model = self.models[model_name]
        
        if model_name == 'xgboost':
            # SHAP for XGBoost
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.iloc[:1000])  # Sample for speed
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(shap_values).mean(0)
            }).sort_values('importance', ascending=False)
            
        elif model_name == 'lightgbm':
            # LightGBM feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            
        else:
            # Fallback to tree-based feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                feature_importance = None
        
        self.feature_importance[model_name] = feature_importance
        return feature_importance
    
    def create_model_comparison_dashboard(self):
        """Create comprehensive model comparison dashboard"""
        print("📊 Creating Model Comparison Dashboard...")
        
        # Performance comparison
        performance_df = pd.DataFrame(self.model_performance).T
        performance_df = performance_df.round(4)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Comparison', 'AUC Scores', 
                          'F1 Scores', 'Feature Importance (Top Model)'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Model performance comparison
        models = list(performance_df.index)
        fig.add_trace(
            go.Bar(x=models, y=performance_df['auc'], name='AUC', marker_color='lightblue'),
            row=1, col=1
        )
        
        # AUC comparison
        fig.add_trace(
            go.Bar(x=models, y=performance_df['auc'], name='AUC', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # F1 comparison
        fig.add_trace(
            go.Bar(x=models, y=performance_df['f1'], name='F1', marker_color='lightcoral'),
            row=2, col=1
        )
        
        # Feature importance (best model)
        best_model = performance_df['auc'].idxmax()
        if best_model in self.feature_importance:
            top_features = self.feature_importance[best_model].head(10)
            fig.add_trace(
                go.Bar(x=top_features['importance'], y=top_features['feature'],
                       orientation='h', name='Feature Importance', marker_color='orange'),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Advanced ML Models Performance Dashboard",
            showlegend=False
        )
        
        # Save dashboard
        fig.write_html("../presentation/advanced_ml_dashboard.html")
        
        return fig, performance_df
    
    def train_all_models(self):
        """Train all advanced models and create comprehensive analysis"""
        print("🚀 ADVANCED ML MODELS TRAINING PIPELINE")
        print("=" * 60)
        
        # Load and prepare data
        X, y, df = self.load_and_prepare_data()
        if X is None:
            return
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {len(X_train):,} samples")
        print(f"📊 Test set: {len(X_test):,} samples")
        
        # Train models
        models_trained = []
        
        try:
            # XGBoost
            self.train_xgboost_model(X_train, X_test, y_train, y_test)
            models_trained.append('XGBoost')
            self.analyze_feature_importance(X, 'xgboost')
        except Exception as e:
            print(f"❌ XGBoost training failed: {e}")
        
        try:
            # Neural Network
            self.train_neural_network(X_train, X_test, y_train, y_test)
            models_trained.append('Neural Network')
        except Exception as e:
            print(f"❌ Neural Network training failed: {e}")
        
        try:
            # LightGBM
            self.train_lightgbm_model(X_train, X_test, y_train, y_test)
            models_trained.append('LightGBM')
            self.analyze_feature_importance(X, 'lightgbm')
        except Exception as e:
            print(f"❌ LightGBM training failed: {e}")
        
        try:
            # Ensemble
            self.create_ensemble_model(X_train, X_test, y_train, y_test)
            models_trained.append('Ensemble')
        except Exception as e:
            print(f"❌ Ensemble training failed: {e}")
        
        # Create dashboard
        dashboard_fig, performance_df = self.create_model_comparison_dashboard()
        
        # Results summary
        print("\n" + "=" * 60)
        print("🏆 ADVANCED ML MODELS RESULTS")
        print("=" * 60)
        
        print("\n📊 Model Performance Summary:")
        print(performance_df)
        
        if len(self.model_performance) > 0:
            best_model = max(self.model_performance.keys(), 
                           key=lambda k: self.model_performance[k]['auc'])
            best_auc = self.model_performance[best_model]['auc']
            print(f"\n🥇 Best Model: {best_model.upper()} (AUC: {best_auc:.4f})")
        
        print(f"\n📈 Models Successfully Trained: {', '.join(models_trained)}")
        print("📊 Dashboard saved to: ../presentation/advanced_ml_dashboard.html")
        
        print("\n✨ Advanced ML Pipeline Complete!")
        return self.models, self.model_performance

if __name__ == "__main__":
    predictor = AdvancedChurnPredictor()
    models, performance = predictor.train_all_models()