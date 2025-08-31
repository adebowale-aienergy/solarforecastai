from __future__ import annotations
from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Return MAE, RMSE, MAPE."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # Safe MAPE (avoid division by zero)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
