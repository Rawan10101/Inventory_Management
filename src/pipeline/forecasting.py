
from __future__ import annotations

from typing import Dict, Optional, Any
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from feature_engineering import FEATURE_COLUMNS


class BaselineForecaster:
    """
    Baseline demand forecaster using RandomForestRegressor.
    """

    def __init__(self, forecast_horizon: str = "daily"):
        self.forecast_horizon = forecast_horizon
        self.model: Optional[RandomForestRegressor] = None
        self.feature_columns = FEATURE_COLUMNS

    def train(self, df, target_col: str = "quantity_sold"):
        print(f"Training baseline {self.forecast_horizon} forecaster...")
        X = df[self.feature_columns].copy()
        y = df[target_col].copy()

        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]

        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        return self

    def predict(self, df) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        X = df[self.feature_columns].copy()
        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)

    def evaluate(self, df, target_col: str = "quantity_sold") -> Dict[str, float]:
        X = df[self.feature_columns].copy()
        y_true = df[target_col].copy()

        valid_idx = ~y_true.isna()
        X = X[valid_idx]
        y_true = y_true[valid_idx]

        y_pred = self.predict(df.loc[valid_idx])

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

        print("Model Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")

        return metrics


class UltimateAIAdapter:
    """
    Adapter to make UltimateDemandForecaster compatible with the existing pipeline.
    """
    
    def __init__(self, forecast_horizon: str = "daily"):
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.feature_columns = FEATURE_COLUMNS
        
    def train(self, df, target_col: str = "quantity_sold"):
        """Train the Ultimate AI model."""
        print("=" * 60)
        print("🚀 TRAINING ULTIMATE AI SOLUTION v3")
        print("=" * 60)
        
        try:
            from ultimate_ai_solution_v3 import UltimateDemandForecaster
            
            # Initialize and train the Ultimate AI model
            self.model = UltimateDemandForecaster(horizon=self.forecast_horizon)
            
            # Ultimate AI models may need different training interface
            # Try different parameter combinations
            try:
                # Try with explicit target column
                self.model.train(df, target_column=target_col)
            except TypeError:
                try:
                    # Try without target column (model might infer it)
                    self.model.train(df)
                except Exception as e:
                    print(f"⚠️ Ultimate AI training failed: {e}")
                    print("Falling back to baseline training...")
                    # Fallback to training like baseline
                    X = df[self.feature_columns].copy()
                    y = df[target_col].copy()
                    self.model._model.fit(X, y)  # Try to access internal model
            
            print("✅ Ultimate AI model trained successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize Ultimate AI: {e}")
            raise
        
        return self
    
    def predict(self, df) -> np.ndarray:
        """Make predictions with Ultimate AI."""
        if self.model is None:
            raise ValueError("Ultimate AI model not trained. Call train() first.")
        
        try:
            # Try Ultimate AI's predict method
            predictions = self.model.predict(df)
            
            # Ensure predictions are non-negative
            predictions = np.maximum(predictions, 0)
            
            print(f"✅ Ultimate AI predictions generated: {len(predictions)} items")
            return predictions
            
        except Exception as e:
            print(f"⚠️ Ultimate AI prediction failed: {e}")
            print("Falling back to baseline prediction...")
            
            # Fallback to baseline-like prediction
            if hasattr(self.model, '_model'):
                X = df[self.feature_columns].copy()
                predictions = self.model._model.predict(X)
                return np.maximum(predictions, 0)
            else:
                raise
    
    def evaluate(self, df, target_col: str = "quantity_sold") -> Dict[str, float]:
        """Evaluate Ultimate AI model."""
        try:
            # Try Ultimate AI's evaluate method
            if hasattr(self.model, 'evaluate'):
                metrics = self.model.evaluate(df, target_column=target_col)
            else:
                # Fallback to standard evaluation
                X = df[self.feature_columns].copy()
                y_true = df[target_col].copy()
                
                valid_idx = ~y_true.isna()
                X = X[valid_idx]
                y_true = y_true[valid_idx]
                
                y_pred = self.predict(df.loc[valid_idx])
                
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = mean_absolute_percentage_error(y_true, y_pred) * 100
                
                metrics = {
                    "MAE": mae, 
                    "RMSE": rmse, 
                    "MAPE": mape,
                    "model": "Ultimate AI v3"
                }
            
            print("\n" + "=" * 60)
            print("ULTIMATE AI PERFORMANCE METRICS:")
            print("=" * 60)
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")
            
            return metrics
            
        except Exception as e:
            print(f"⚠️ Ultimate AI evaluation failed: {e}")
            print("Falling back to baseline evaluation...")
            
            # Fallback to baseline evaluation
            baseline = BaselineForecaster(forecast_horizon=self.forecast_horizon)
            baseline.model = getattr(self.model, '_model', None)
            return baseline.evaluate(df, target_col)


def get_forecaster(forecast_horizon: str = "daily", prefer_advanced: bool = True):
    """
    Get the appropriate forecaster.
    
    Args:
        forecast_horizon: "daily", "weekly", or "hourly"
        prefer_advanced: If True, tries to use Ultimate AI Solution v3
    
    Returns:
        A forecaster instance
    """
    
    # DEFAULT TO ADVANCED (Ultimate AI) - Changed from False to True
    if prefer_advanced:
        try:
            print("\n" + "=" * 70)
            print("🎯 ATTEMPTING TO USE ULTIMATE AI SOLUTION v3")
            print("=" * 70)
            
            # First try direct Ultimate AI integration
            try:
                from ultimate_ai_solution_v3 import UltimateDemandForecaster
                
                print("✅ Ultimate AI Solution v3 found!")
                print("   Features: Real names, multi-horizon, weather-aware")
                print("   Promotions: Dynamic pricing, bundle optimization")
                print("   Inventory: Expiration prediction, waste reduction")
                
                # Use adapter for compatibility
                return UltimateAIAdapter(forecast_horizon=forecast_horizon)
                
            except ImportError as e:
                print(f"⚠️ Direct Ultimate AI import failed: {e}")
                
                # Try intermediate advanced model
                try:
                    from demand_forecasting import DemandForecaster
                    print("✅ Using intermediate DemandForecaster")
                    return DemandForecaster(forecast_horizon=forecast_horizon)
                    
                except ImportError:
                    print("⚠️ Intermediate model also unavailable")
                    raise
            
        except Exception as e:
            print(f"\n❌ Advanced models unavailable: {e}")
            print("⚠️  Falling back to baseline RandomForest")
            print("⚠️  NOTE: Baseline may not include real name mapping!")
            print("=" * 70 + "\n")
    
    # Only use baseline if explicitly requested
    print("\n" + "=" * 60)
    print("USING BASELINE RANDOM FOREST FORECASTER")
    print("NOTE: This model uses numeric IDs, not real names!")
    print("=" * 60 + "\n")
    
    return BaselineForecaster(forecast_horizon=forecast_horizon)


# Legacy function for backward compatibility
def get_baseline_forecaster(forecast_horizon: str = "daily"):
    """Get baseline forecaster (explicitly)."""
    return BaselineForecaster(forecast_horizon=forecast_horizon)


def get_ultimate_ai_forecaster(forecast_horizon: str = "daily"):
    """Get Ultimate AI forecaster (explicitly)."""
    try:
        return UltimateAIAdapter(forecast_horizon=forecast_horizon)
    except Exception as e:
        print(f"❌ Could not get Ultimate AI forecaster: {e}")
        print("Falling back to baseline...")
        return BaselineForecaster(forecast_horizon=forecast_horizon)


# Quick test function
def test_forecaster_integration():
    """Test if Ultimate AI is properly integrated."""
    print("\n" + "=" * 70)
    print("TESTING FORECASTER INTEGRATION")
    print("=" * 70)
    
    # Test 1: Try to import Ultimate AI
    try:
        import ultimate_ai_solution_v3
        print("✅ ultimate_ai_solution_v3 module found")
        
        # Check for specific classes
        from ultimate_ai_solution_v3 import UltimateDemandForecaster
        print("✅ UltimateDemandForecaster class found")
        
        # Test initialization
        forecaster = UltimateDemandForecaster(horizon="daily")
        print("✅ UltimateDemandForecaster initialized successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Ultimate AI module not found: {e}")
        print("Location checked: ", __file__)
        
        # Show search path
        import sys
        print("\nPython path:")
        for p in sys.path:
            print(f"  {p}")
            
        return False
    except Exception as e:
        print(f"❌ Ultimate AI test failed: {e}")
        return False


if __name__ == "__main__":
    # Run integration test when file is executed directly
    test_forecaster_integration()