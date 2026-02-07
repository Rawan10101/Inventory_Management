"""
File: demand_forecasting.py
Description: Main demand forecasting model using XGBoost and ensemble methods
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class DemandForecaster:
    """
    Advanced demand forecasting using stacking ensemble [web:13][web:15][web:16]
    """
    
    def __init__(self, forecast_horizon: str = 'daily'):
        """
        Initialize forecaster
        
        Args:
            forecast_horizon: 'daily', 'weekly', or 'monthly'
        """
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.feature_columns = None
        self.feature_importance = None
        
    def _get_feature_columns(self) -> List[str]:
        """
        Define which columns to use as features
        """
        return [
            # Temporal features
            'day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter',
            'hour_of_day', 'is_weekend', 'is_holiday', 'days_to_holiday', 'is_ramadan',
            
            # Lag features
            'quantity_lag_1d', 'quantity_lag_7d', 'quantity_lag_14d', 'quantity_lag_28d',
            
            # Rolling statistics
            'quantity_rolling_mean_7d', 'quantity_rolling_std_7d',
            'quantity_rolling_mean_14d', 'quantity_rolling_std_14d',
            'quantity_rolling_mean_30d', 'quantity_rolling_std_30d',
            'quantity_ewm', 'dow_avg',
            
            # Price features
            'avg_price', 'price_change',
            
            # Context features (need to be encoded)
            'order_count'
        ]
    
    def _build_base_models(self) -> List[Tuple[str, object]]:
        """
        Build base models for stacking ensemble [web:16]
        """
        base_models = [
            ('xgb', xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.05,
                reg_lambda=1.0,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )),
            ('lgb', lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )),
            ('rf', RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ))
        ]
        
        return base_models
    
    def _build_stacking_model(self):
        """
        Build stacking ensemble model [web:16]
        """
        base_models = self._build_base_models()
        
        # Meta-learner (final estimator)
        meta_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Create stacking regressor
        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=TimeSeriesSplit(n_splits=5),
            n_jobs=-1
        )
        
        return stacking_model
    
    def train(self, df: pd.DataFrame, target_col: str = 'quantity_sold'):
        """
        Train the forecasting model [web:13][web:15]
        """
        print(f"\nTraining {self.forecast_horizon} demand forecasting model...")
        
        # Get feature columns
        self.feature_columns = self._get_feature_columns()
        
        # Prepare data
        X = df[self.feature_columns].copy()
        y = df[target_col].copy()
        
        # Remove any rows with NaN in target
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"Training samples: {len(X)}")
        print(f"Features: {len(self.feature_columns)}")
        
        # Build and train model
        self.model = self._build_stacking_model()
        
        print("Training stacking ensemble (this may take a few minutes)...")
        self.model.fit(X, y)
        
        # Calculate feature importance from XGBoost base model
        xgb_model = self.model.named_estimators_['xgb']
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(self.feature_importance.head(10))
        
        # Cross-validation
        print("\nPerforming time-series cross-validation...")
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=tscv, 
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=-1
        )
        
        mape = -cv_scores.mean() * 100
        print(f"Cross-Validation MAPE: {mape:.2f}%")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X = df[self.feature_columns].copy()
        predictions = self.model.predict(X)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def evaluate(self, df: pd.DataFrame, target_col: str = 'quantity_sold') -> Dict:
        """
        Evaluate model performance
        """
        X = df[self.feature_columns].copy()
        y_true = df[target_col].copy()
        
        # Remove NaN
        valid_idx = ~y_true.isna()
        X = X[valid_idx]
        y_true = y_true[valid_idx]
        
        y_pred = self.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        # Avoid division by zero in MAPE calculation
        mask = y_true > 0
        if mask.sum() > 0:
            mape_filtered = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
        else:
            mape_filtered = mape
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape_filtered
        }
        
        print("\nModel Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'forecast_horizon': self.forecast_horizon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.feature_importance = data['feature_importance']
        self.forecast_horizon = data['forecast_horizon']
        print(f"Model loaded from {filepath}")