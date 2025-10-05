import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_predictions(y_true, y_pred, title="Model Predictions"):
    plt.figure(figsize=(8,5))
    sns.lineplot(x=range(len(y_true)), y=y_true, label="Actual", color="blue")
    sns.lineplot(x=range(len(y_pred)), y=y_pred, label="Predicted", color="red")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Solar Power")
    plt.legend()
    plt.tight_layout()
    return plt

def plot_feature_importance(model, feature_names):
    """
    Works for RandomForest or any model with `feature_importances_`
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        feat_imp = feat_imp.sort_values("Importance", ascending=False)

        plt.figure(figsize=(8,5))
        sns.barplot(data=feat_imp, x="Importance", y="Feature", palette="viridis")
        plt.title("Feature Importance")
        plt.tight_layout()
        return plt
    else:
        raise ValueError("Model has no feature_importances_ attribute")
 
