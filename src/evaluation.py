# evaluation.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from visualization import plot_predictions, plot_feature_importance


def evaluate_model(model, X_test, y_test, feature_names=None, model_name="Model"):
    """
    Evaluate a model on test data.
    
    Args:
        model: trained model (RandomForest, Prophet, LSTM, etc.)
        X_test: features (array or DataFrame)
        y_test: true values
        feature_names: list of feature names (for feature importance plots)
        model_name: str, name of the model

    Returns:
        dict of metrics
    """
    # Handle models differently
    if model_name.lower() == "prophet":
        # Prophet expects a DataFrame with "ds" column
        forecast = model.predict(X_test)
        y_pred = forecast["yhat"].values
    elif model_name.lower() == "lstm":
        # For LSTM, assume X_test is already shaped correctly
        y_pred = model.predict(X_test).flatten()
    else:
        # Default: sklearn-style models
        y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    print(f"\n🔎 {model_name} Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Plot predictions
    plt_pred = plot_predictions(y_test, y_pred, title=f"{model_name} Predictions")
    plt_pred.show()

    # Plot feature importance (if applicable)
    if feature_names is not None:
        try:
            plt_fi = plot_feature_importance(model, feature_names)
            plt_fi.show()
        except Exception as e:
            print(f"(Skipping feature importance) Reason: {e}")

    return metrics


def compare_models(results_dict):
    """
    Compare evaluation metrics across multiple models.
    
    Args:
        results_dict: dict like
            {
                "RandomForest": {"MAE": ..., "RMSE": ..., "R2": ...},
                "Prophet": {"MAE": ..., "RMSE": ..., "R2": ...}
            }
    """
    df = pd.DataFrame(results_dict).T
    print("\n📊 Model Comparison:\n", df)

    df.plot(kind="bar", figsize=(8,5), title="Model Performance Comparison")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()

    return df
