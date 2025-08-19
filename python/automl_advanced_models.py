#!/usr/bin/env python3
"""
🤖 Advanced AutoML & Ensemble Modeling System
Automated machine learning with hyperparameter optimization and ensemble techniques
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score,
    train_test_split, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
import joblib
import sqlite3
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelResult:
    """Model evaluation result"""
    model_name: str
    model_type: str  # 'classification' or 'regression'
    cv_score: float
    train_score: float
    test_score: float
    feature_importance: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_time: float
    prediction_time: float
    model_size: int
    timestamp: datetime

@dataclass
class AutoMLConfig:
    """AutoML configuration"""
    max_time_mins: int = 30
    max_models: int = 50
    cv_folds: int = 5
    optimization_metric: str = 'roc_auc'  # for classification
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    verbose: bool = True

class AdvancedFeatureEngineering:
    """Advanced feature engineering pipeline"""
    
    def __init__(self):
        self.feature_transformers = {}
        self.generated_features = []
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features from raw data"""
        df_enhanced = df.copy()
        
        try:
            # Time-based features if date columns exist
            date_columns = df.select_dtypes(include=['datetime64']).columns
            for col in date_columns:
                df_enhanced[f'{col}_year'] = df[col].dt.year
                df_enhanced[f'{col}_month'] = df[col].dt.month
                df_enhanced[f'{col}_day'] = df[col].dt.day
                df_enhanced[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df_enhanced[f'{col}_quarter'] = df[col].dt.quarter
                df_enhanced[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
                
                # Days since epoch for trend analysis
                df_enhanced[f'{col}_days_since_epoch'] = (df[col] - pd.Timestamp('1970-01-01')).dt.days
            
            # Numerical feature interactions
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 1:
                for i, col1 in enumerate(numerical_cols):
                    for col2 in numerical_cols[i+1:]:
                        # Ratio features
                        safe_col2 = df_enhanced[col2].replace(0, 1e-6)
                        df_enhanced[f'{col1}_div_{col2}'] = df_enhanced[col1] / safe_col2
                        
                        # Product features
                        df_enhanced[f'{col1}_mul_{col2}'] = df_enhanced[col1] * df_enhanced[col2]
                        
                        # Difference features
                        df_enhanced[f'{col1}_minus_{col2}'] = df_enhanced[col1] - df_enhanced[col2]
            
            # Statistical features per group
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for cat_col in categorical_cols:
                if cat_col in df_enhanced.columns:
                    for num_col in numerical_cols:
                        if num_col in df_enhanced.columns:
                            # Group statistics
                            group_stats = df_enhanced.groupby(cat_col)[num_col].agg(['mean', 'std', 'median'])
                            df_enhanced[f'{num_col}_mean_by_{cat_col}'] = df_enhanced[cat_col].map(group_stats['mean'])
                            df_enhanced[f'{num_col}_std_by_{cat_col}'] = df_enhanced[cat_col].map(group_stats['std'])
                            df_enhanced[f'{num_col}_median_by_{cat_col}'] = df_enhanced[cat_col].map(group_stats['median'])
            
            # Polynomial features for key variables
            key_features = ['monthly_charges', 'tenure', 'total_charges'] if 'monthly_charges' in df.columns else numerical_cols[:3]
            for feature in key_features:
                if feature in df_enhanced.columns:
                    df_enhanced[f'{feature}_squared'] = df_enhanced[feature] ** 2
                    df_enhanced[f'{feature}_cubed'] = df_enhanced[feature] ** 3
                    df_enhanced[f'{feature}_sqrt'] = np.sqrt(np.abs(df_enhanced[feature]))
                    df_enhanced[f'{feature}_log'] = np.log1p(np.abs(df_enhanced[feature]))
            
            # Binning continuous variables
            for col in numerical_cols:
                if col in df_enhanced.columns and df_enhanced[col].nunique() > 10:
                    df_enhanced[f'{col}_binned'] = pd.qcut(df_enhanced[col], q=5, labels=['low', 'med_low', 'med', 'med_high', 'high'], duplicates='drop')
            
            # Frequency encoding for categorical variables
            for col in categorical_cols:
                if col in df_enhanced.columns:
                    freq_map = df_enhanced[col].value_counts().to_dict()
                    df_enhanced[f'{col}_frequency'] = df_enhanced[col].map(freq_map)
            
            logger.info(f"Enhanced features: {df.shape[1]} -> {df_enhanced.shape[1]} features")
            return df_enhanced
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return df

class HyperparameterOptimizer:
    """Advanced hyperparameter optimization"""
    
    def __init__(self):
        self.param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            'GradientBoosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            },
            'MLP': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
    
    def optimize_model(self, model, param_grid: Dict, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5, scoring: str = 'roc_auc') -> Tuple[Any, Dict]:
        """Optimize hyperparameters for a given model"""
        try:
            # Use RandomizedSearchCV for efficiency
            search = RandomizedSearchCV(
                model, param_grid, n_iter=50, cv=cv, 
                scoring=scoring, n_jobs=-1, random_state=42
            )
            
            search.fit(X, y)
            
            return search.best_estimator_, search.best_params_
            
        except Exception as e:
            logger.error(f"Error optimizing model: {e}")
            return model, {}

class EnsembleModelBuilder:
    """Advanced ensemble model construction"""
    
    def __init__(self):
        self.base_models = {}
        self.ensemble_models = {}
        
    def create_voting_ensemble(self, models: List[Tuple[str, Any]], 
                              voting: str = 'soft') -> VotingClassifier:
        """Create voting ensemble from multiple models"""
        try:
            ensemble = VotingClassifier(estimators=models, voting=voting)
            return ensemble
        except Exception as e:
            logger.error(f"Error creating voting ensemble: {e}")
            return None
    
    def create_stacking_ensemble(self, base_models: List[Tuple[str, Any]], 
                               meta_model: Any = None) -> Any:
        """Create stacking ensemble"""
        try:
            from sklearn.ensemble import StackingClassifier
            
            if meta_model is None:
                meta_model = LogisticRegression()
            
            ensemble = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                cv=5
            )
            return ensemble
        except Exception as e:
            logger.error(f"Error creating stacking ensemble: {e}")
            return None
    
    def create_blending_ensemble(self, models: List[Any], X_blend: np.ndarray, 
                               y_blend: np.ndarray) -> Dict[str, float]:
        """Create blending weights for ensemble"""
        try:
            # Generate predictions from each model
            predictions = []
            for model in models:
                pred = model.predict_proba(X_blend)[:, 1]
                predictions.append(pred)
            
            # Optimize blending weights
            from scipy.optimize import minimize
            
            def blend_objective(weights):
                weights = weights / np.sum(weights)  # Normalize
                blended_pred = np.average(predictions, axis=0, weights=weights)
                return -roc_auc_score(y_blend, blended_pred)
            
            # Initialize equal weights
            init_weights = np.ones(len(models)) / len(models)
            
            # Constraints: weights sum to 1 and are non-negative
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(len(models))]
            
            result = minimize(blend_objective, init_weights, 
                            constraints=constraints, bounds=bounds)
            
            optimal_weights = result.x / np.sum(result.x)
            return dict(zip([f"model_{i}" for i in range(len(models))], optimal_weights))
            
        except Exception as e:
            logger.error(f"Error creating blending ensemble: {e}")
            return {}

class AutoMLPipeline:
    """Complete AutoML pipeline"""
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        self.feature_engineer = AdvancedFeatureEngineering()
        self.optimizer = HyperparameterOptimizer()
        self.ensemble_builder = EnsembleModelBuilder()
        self.models_db = "data/live/automl_models.db"
        self.best_models = []
        self.ensure_db_exists()
    
    def ensure_db_exists(self):
        """Create models database"""
        os.makedirs("data/live", exist_ok=True)
        
        conn = sqlite3.connect(self.models_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                model_type TEXT,
                cv_score REAL,
                train_score REAL,
                test_score REAL,
                feature_importance TEXT,
                hyperparameters TEXT,
                training_time REAL,
                prediction_time REAL,
                model_size INTEGER,
                timestamp TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ensemble_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ensemble_type TEXT,
                base_models TEXT,
                cv_score REAL,
                test_score REAL,
                weights TEXT,
                timestamp TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def run_automl(self, X: pd.DataFrame, y: pd.Series, 
                   problem_type: str = 'classification') -> Dict[str, Any]:
        """Run complete AutoML pipeline"""
        try:
            start_time = datetime.now()
            logger.info(f"Starting AutoML pipeline for {problem_type}")
            
            # Feature engineering
            logger.info("Step 1: Advanced feature engineering")
            X_enhanced = self.feature_engineer.create_advanced_features(X)
            
            # Preprocessing
            logger.info("Step 2: Preprocessing")
            X_processed, y_processed = self._preprocess_data(X_enhanced, y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=self.config.test_size,
                random_state=self.config.random_state, stratify=y_processed if problem_type == 'classification' else None
            )
            
            # Model training and optimization
            logger.info("Step 3: Model training and hyperparameter optimization")
            model_results = self._train_and_optimize_models(
                X_train, y_train, X_test, y_test, problem_type
            )
            
            # Ensemble creation
            logger.info("Step 4: Ensemble model creation")
            ensemble_results = self._create_ensembles(
                X_train, y_train, X_test, y_test, problem_type
            )
            
            # Model selection
            logger.info("Step 5: Best model selection")
            best_model_info = self._select_best_model(model_results + ensemble_results)
            
            # Feature importance analysis
            feature_importance = self._analyze_feature_importance(X_processed.columns)
            
            # Generate insights
            insights = self._generate_automl_insights(model_results, ensemble_results, feature_importance)
            
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            results = {
                "status": "completed",
                "total_time_seconds": total_time,
                "models_trained": len(model_results),
                "ensembles_created": len(ensemble_results),
                "best_model": best_model_info,
                "feature_importance": feature_importance,
                "model_results": model_results,
                "ensemble_results": ensemble_results,
                "insights": insights,
                "data_stats": {
                    "original_features": X.shape[1],
                    "engineered_features": X_enhanced.shape[1],
                    "final_features": X_processed.shape[1],
                    "training_samples": len(X_train),
                    "test_samples": len(X_test)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Save results
            self._save_automl_results(results)
            
            logger.info(f"AutoML completed in {total_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error in AutoML pipeline: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for modeling"""
        try:
            X_processed = X.copy()
            
            # Handle categorical variables
            categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            
            # Handle missing values
            X_processed = X_processed.fillna(X_processed.median())
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_processed)
            
            # Feature selection
            selector = SelectKBest(score_func=f_classif, k=min(50, X_scaled.shape[1]))
            X_selected = selector.fit_transform(X_scaled, y)
            
            return X_selected, y.values
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return X.values, y.values
    
    def _train_and_optimize_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray, y_test: np.ndarray,
                                  problem_type: str) -> List[Dict]:
        """Train and optimize multiple models"""
        results = []
        
        # Define base models
        if problem_type == 'classification':
            base_models = {
                'RandomForest': RandomForestClassifier(random_state=42),
                'GradientBoosting': GradientBoostingClassifier(random_state=42),
                'ExtraTrees': ExtraTreesClassifier(random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42),
                'SVM': SVC(probability=True, random_state=42),
                'MLP': MLPClassifier(random_state=42, max_iter=500)
            }
            scoring = 'roc_auc'
        else:
            base_models = {
                'RandomForest': RandomForestRegressor(random_state=42),
                'GradientBoosting': GradientBoostingRegressor(random_state=42),
                'ExtraTrees': ExtraTreesRegressor(random_state=42),
                'ElasticNet': ElasticNet(random_state=42),
                'SVR': SVR(),
                'MLP': MLPRegressor(random_state=42, max_iter=500)
            }
            scoring = 'r2'
        
        # Train and optimize each model
        for model_name, model in base_models.items():
            try:
                logger.info(f"Training {model_name}")
                start_time = datetime.now()
                
                # Get parameter grid
                param_grid = self.optimizer.param_grids.get(
                    model_name.replace('Classifier', '').replace('Regressor', ''), {}
                )
                
                # Optimize hyperparameters
                if param_grid:
                    optimized_model, best_params = self.optimizer.optimize_model(
                        model, param_grid, X_train, y_train, scoring=scoring
                    )
                else:
                    optimized_model = model
                    best_params = {}
                
                # Train final model
                optimized_model.fit(X_train, y_train)
                
                # Evaluate
                cv_score = cross_val_score(optimized_model, X_train, y_train, cv=5, scoring=scoring).mean()
                train_score = optimized_model.score(X_train, y_train)
                test_score = optimized_model.score(X_test, y_test)
                
                # Feature importance
                feature_importance = {}
                if hasattr(optimized_model, 'feature_importances_'):
                    feature_importance = {f"feature_{i}": imp for i, imp in enumerate(optimized_model.feature_importances_)}
                
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Prediction time test
                pred_start = datetime.now()
                _ = optimized_model.predict(X_test[:100])
                prediction_time = (datetime.now() - pred_start).total_seconds()
                
                result = {
                    'model_name': model_name,
                    'model_type': problem_type,
                    'cv_score': cv_score,
                    'train_score': train_score,
                    'test_score': test_score,
                    'feature_importance': feature_importance,
                    'hyperparameters': best_params,
                    'training_time': training_time,
                    'prediction_time': prediction_time,
                    'model_size': len(joblib.dumps(optimized_model)),
                    'model_object': optimized_model
                }
                
                results.append(result)
                
                # Save model
                model_path = f"data/live/models/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                os.makedirs("data/live/models", exist_ok=True)
                joblib.dump(optimized_model, model_path)
                
                logger.info(f"{model_name} - CV Score: {cv_score:.4f}, Test Score: {test_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return results
    
    def _create_ensembles(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         problem_type: str) -> List[Dict]:
        """Create ensemble models"""
        ensemble_results = []
        
        # Get best performing models
        top_models = sorted(self.best_models, key=lambda x: x['cv_score'], reverse=True)[:5]
        
        if len(top_models) < 2:
            return ensemble_results
        
        try:
            # Voting ensemble
            estimators = [(m['model_name'], m['model_object']) for m in top_models[:3]]
            
            if problem_type == 'classification':
                voting_ensemble = VotingClassifier(estimators=estimators, voting='soft')
                scoring = 'roc_auc'
            else:
                voting_ensemble = VotingRegressor(estimators=estimators)
                scoring = 'r2'
            
            voting_ensemble.fit(X_train, y_train)
            
            cv_score = cross_val_score(voting_ensemble, X_train, y_train, cv=5, scoring=scoring).mean()
            test_score = voting_ensemble.score(X_test, y_test)
            
            ensemble_results.append({
                'ensemble_type': 'voting',
                'base_models': [m['model_name'] for m in top_models[:3]],
                'cv_score': cv_score,
                'test_score': test_score,
                'weights': 'equal',
                'model_object': voting_ensemble
            })
            
            logger.info(f"Voting ensemble - CV Score: {cv_score:.4f}, Test Score: {test_score:.4f}")
            
        except Exception as e:
            logger.error(f"Error creating voting ensemble: {e}")
        
        return ensemble_results
    
    def _select_best_model(self, all_results: List[Dict]) -> Dict:
        """Select the best performing model"""
        if not all_results:
            return {}
        
        # Sort by CV score
        best_result = max(all_results, key=lambda x: x.get('cv_score', 0))
        
        return {
            'name': best_result.get('model_name', best_result.get('ensemble_type', 'unknown')),
            'type': best_result.get('model_type', 'ensemble'),
            'cv_score': best_result.get('cv_score', 0),
            'test_score': best_result.get('test_score', 0),
            'is_ensemble': 'ensemble_type' in best_result
        }
    
    def _analyze_feature_importance(self, feature_names: pd.Index) -> Dict[str, float]:
        """Analyze overall feature importance across models"""
        try:
            importance_sum = {}
            model_count = 0
            
            for result in self.best_models:
                if result.get('feature_importance'):
                    model_count += 1
                    for feature, importance in result['feature_importance'].items():
                        importance_sum[feature] = importance_sum.get(feature, 0) + importance
            
            if model_count > 0:
                avg_importance = {k: v / model_count for k, v in importance_sum.items()}
                # Sort by importance
                return dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True))
            
            return {}
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            return {}
    
    def _generate_automl_insights(self, model_results: List[Dict], 
                                 ensemble_results: List[Dict],
                                 feature_importance: Dict[str, float]) -> List[str]:
        """Generate insights from AutoML results"""
        insights = []
        
        try:
            if model_results:
                # Best model insight
                best_model = max(model_results, key=lambda x: x['cv_score'])
                insights.append(f"Best performing model: {best_model['model_name']} with CV score of {best_model['cv_score']:.4f}")
                
                # Model comparison insight
                scores = [m['cv_score'] for m in model_results]
                if len(scores) > 1:
                    score_std = np.std(scores)
                    if score_std < 0.02:
                        insights.append("Model performance is very similar across algorithms - ensemble methods recommended")
                    else:
                        insights.append(f"Significant performance difference between models (std: {score_std:.4f}) - model selection is critical")
                
                # Training time insight
                training_times = [m['training_time'] for m in model_results]
                fastest_model = min(model_results, key=lambda x: x['training_time'])
                insights.append(f"Fastest training model: {fastest_model['model_name']} ({fastest_model['training_time']:.2f}s)")
            
            if ensemble_results:
                best_ensemble = max(ensemble_results, key=lambda x: x['cv_score'])
                ensemble_improvement = best_ensemble['cv_score'] - (max(model_results, key=lambda x: x['cv_score'])['cv_score'] if model_results else 0)
                if ensemble_improvement > 0.01:
                    insights.append(f"Ensemble provides significant improvement: +{ensemble_improvement:.4f} over best single model")
                else:
                    insights.append("Ensemble provides minimal improvement - single model may be sufficient")
            
            if feature_importance:
                top_features = list(feature_importance.keys())[:3]
                insights.append(f"Top predictive features: {', '.join(top_features)}")
                
                # Feature distribution insight
                importance_values = list(feature_importance.values())
                if len(importance_values) > 0:
                    top_10_sum = sum(importance_values[:10])
                    if top_10_sum > 0.8:
                        insights.append("Model relies heavily on top 10 features - feature selection could improve efficiency")
                    else:
                        insights.append("Features contribute broadly to predictions - full feature set recommended")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights.append("Unable to generate detailed insights due to processing error")
        
        return insights
    
    def _save_automl_results(self, results: Dict[str, Any]):
        """Save AutoML results to database and files"""
        try:
            # Save to JSON
            with open("data/live/automl_results.json", "w") as f:
                # Remove model objects for JSON serialization
                results_copy = results.copy()
                if 'model_results' in results_copy:
                    for model in results_copy['model_results']:
                        model.pop('model_object', None)
                if 'ensemble_results' in results_copy:
                    for ensemble in results_copy['ensemble_results']:
                        ensemble.pop('model_object', None)
                        
                json.dump(results_copy, f, indent=2, default=str)
            
            # Save to database
            conn = sqlite3.connect(self.models_db)
            cursor = conn.cursor()
            
            # Save model results
            for model in results.get('model_results', []):
                cursor.execute("""
                    INSERT INTO model_results 
                    (model_name, model_type, cv_score, train_score, test_score, 
                     feature_importance, hyperparameters, training_time, 
                     prediction_time, model_size, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model['model_name'],
                    model['model_type'],
                    model['cv_score'],
                    model['train_score'],
                    model['test_score'],
                    json.dumps(model['feature_importance']),
                    json.dumps(model['hyperparameters']),
                    model['training_time'],
                    model['prediction_time'],
                    model['model_size'],
                    datetime.now().isoformat()
                ))
            
            # Save ensemble results
            for ensemble in results.get('ensemble_results', []):
                cursor.execute("""
                    INSERT INTO ensemble_results 
                    (ensemble_type, base_models, cv_score, test_score, weights, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    ensemble['ensemble_type'],
                    json.dumps(ensemble['base_models']),
                    ensemble['cv_score'],
                    ensemble['test_score'],
                    json.dumps(ensemble['weights']),
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving AutoML results: {e}")

def demo_automl():
    """Demo function for AutoML pipeline"""
    try:
        # Generate sample data
        np.random.seed(42)
        n_samples = 1000
        
        # Create synthetic subscription data
        data = {
            'monthly_charges': np.random.normal(50, 20, n_samples),
            'tenure': np.random.exponential(24, n_samples),
            'total_charges': np.random.normal(1200, 600, n_samples),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'payment_method': np.random.choice(['Electronic check', 'Credit card', 'Bank transfer'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'online_security': np.random.choice(['Yes', 'No'], n_samples),
            'tech_support': np.random.choice(['Yes', 'No'], n_samples),
            'streaming_tv': np.random.choice(['Yes', 'No'], n_samples),
            'paperless_billing': np.random.choice(['Yes', 'No'], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable (churn)
        # Higher monthly charges and shorter tenure increase churn probability
        churn_prob = (
            0.3 * (df['monthly_charges'] > 70).astype(int) +
            0.4 * (df['tenure'] < 12).astype(int) +
            0.2 * (df['contract_type'] == 'Month-to-month').astype(int) +
            0.1 * np.random.random(n_samples)
        )
        
        y = (churn_prob > 0.5).astype(int)
        
        # Run AutoML
        config = AutoMLConfig(max_time_mins=10, max_models=20)
        automl = AutoMLPipeline(config)
        
        results = automl.run_automl(df, pd.Series(y), problem_type='classification')
        
        print("AutoML Results:")
        print(f"Status: {results['status']}")
        print(f"Total time: {results['total_time_seconds']:.2f} seconds")
        print(f"Models trained: {results['models_trained']}")
        print(f"Best model: {results['best_model']['name']} (Score: {results['best_model']['cv_score']:.4f})")
        print("\nInsights:")
        for insight in results['insights']:
            print(f"- {insight}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in AutoML demo: {e}")
        return None

if __name__ == "__main__":
    demo_automl()