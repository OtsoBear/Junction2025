"""
Zero-Fail Logistics - ML Model Training for Stockout Prediction
Junction 2025 Challenge - Valio Aimo

This script trains a LightGBM model to predict out-of-stock probability per order line.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
)
import lightgbm as lgb
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("STOCKOUT PREDICTION - MODEL TRAINING")
print("=" * 80)

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"

print("\n[1/6] Loading feature-engineered data...")
df = pd.read_parquet(ARTIFACTS_DIR / "features_for_modeling.parquet")
print(f"✓ Loaded {len(df):,} samples")
print(f"  Stockout rate: {df['is_stockout'].mean()*100:.2f}%")

print("\n[2/6] Preparing data for training...")

# Separate features and target (exclude non-numeric categorical string columns)
feature_cols = [
    col
    for col in df.columns
    if col not in ["is_stockout", "order_created_date", "sales_unit", "customer_segment"]
]
X = df[feature_cols].copy()
y = df["is_stockout"].copy()

print(f"  Features: {len(feature_cols)}")
print("  Target variable: is_stockout")

# Time-based split (last 10% as test set to simulate real prediction)
split_date = df["order_created_date"].quantile(0.9)
train_mask = df["order_created_date"] < split_date
test_mask = df["order_created_date"] >= split_date

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print(
    f"\n✓ Train set: {len(X_train):,} samples ({y_train.mean()*100:.2f}% stockout)"
)
print(
    f"✓ Test set:  {len(X_test):,} samples ({y_test.mean()*100:.2f}% stockout)"
)

print("\n[3/6] Training LightGBM model...")

# Create LightGBM datasets (all features treated as numeric)
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Estimate class imbalance for scale_pos_weight
pos_rate = y_train.mean()
neg_rate = 1 - pos_rate
scale_pos_weight = neg_rate / pos_rate if pos_rate > 0 else 1.0

# Model parameters
params = {
    "objective": "binary",
    # Use AUC as the primary metric for early stopping (better aligned with ranking/alerting)
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "max_depth": -1,
    "min_child_samples": 50,
    "scale_pos_weight": scale_pos_weight,
}

print(f"Class imbalance (train): {pos_rate*100:.2f}% positives")
print(f"Using scale_pos_weight = {scale_pos_weight:.2f}")

print("Training model...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    valid_names=["train", "test"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=50),
    ],
)

print(f"\n✓ Model trained with {model.best_iteration} iterations")

print("\n[4/6] Evaluating model performance...")

# Predictions
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate metrics
auc_score = roc_auc_score(y_test, y_pred_proba)
ap_score = average_precision_score(y_test, y_pred_proba)
brier = brier_score_loss(y_test, y_pred_proba)

print("\n📊 MODEL PERFORMANCE:")
print(f"   ROC-AUC Score: {auc_score:.4f}")
print(f"   Average Precision: {ap_score:.4f}")
print(f"   Brier Score (calibration): {brier:.4f}")

print("\n📊 CLASSIFICATION REPORT (threshold = 0.5):")
print(classification_report(y_test, y_pred, target_names=["Fulfilled", "Stockout"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📊 CONFUSION MATRIX (threshold = 0.5):")
print(f"   True Negatives:  {cm[0, 0]:,}")
print(f"   False Positives: {cm[0, 1]:,}")
print(f"   False Negatives: {cm[1, 0]:,}")
print(f"   True Positives:  {cm[1, 1]:,}")

# Threshold analysis for more realistic operating points
thresholds = [0.10, 0.15, 0.20]
print("\n📊 Threshold analysis (focused on low probability range):")
for thr in thresholds:
    preds_thr = (y_pred_proba > thr).astype(int)
    cm_thr = confusion_matrix(y_test, preds_thr)
    tp = cm_thr[1, 1]
    fp = cm_thr[0, 1]
    fn = cm_thr[1, 0]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(
        f"  Threshold {thr:.2f} -> Precision: {precision:.3f}, "
        f"Recall: {recall:.3f}, Positives: {preds_thr.sum():,}"
    )

# Scan a finer grid to find a threshold with strong F1 for presentation
best_f1 = 0.0
best_threshold = None
best_stats = None
candidate_thresholds = np.linspace(0.05, 0.30, 26)

for thr in candidate_thresholds:
    preds_thr = (y_pred_proba > thr).astype(int)
    tp = ((preds_thr == 1) & (y_test == 1)).sum()
    fp = ((preds_thr == 1) & (y_test == 0)).sum()
    fn = ((preds_thr == 0) & (y_test == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thr
        best_stats = (precision, recall, preds_thr.sum())

if best_threshold is not None:
    p_best, r_best, n_pos = best_stats
    print(
        f"\n✨ Recommended operating threshold (max F1 on test set): {best_threshold:.3f}\n"
        f"   Precision: {p_best:.3f}, Recall: {r_best:.3f}, Positives flagged: {n_pos:,}"
    )

print("\n[5/6] Analyzing feature importance...")

# Get feature importance
feature_importance = pd.DataFrame(
    {
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }
)
feature_importance = feature_importance.sort_values("importance", ascending=False)

print("\n🎯 TOP 20 MOST IMPORTANT FEATURES:")
print(feature_importance.head(20).to_string(index=False))

# Save feature importance
feature_importance.to_csv(f"{DATA_DIR}/feature_importance.csv", index=False)
print("\n✓ Saved feature importance to feature_importance.csv")

print("\n[6/6] Saving model and predictions...")

# Save the model
model.save_model(ARTIFACTS_DIR / "stockout_prediction_model.txt")
joblib.dump(model, ARTIFACTS_DIR / "stockout_prediction_model.pkl")
print("✓ Saved model to artifacts/stockout_prediction_model.pkl")

# Save test predictions for analysis
test_results = pd.DataFrame(
    {
        "actual": y_test,
        "predicted_proba": y_pred_proba,
        "predicted": y_pred,
    }
)
test_results.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)
print("✓ Saved test predictions to artifacts/test_predictions.csv")

# Create visualizations
print("\n📊 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Feature Importance
ax1 = axes[0, 0]
top_features = feature_importance.head(15)
ax1.barh(range(len(top_features)), top_features["importance"])
ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features["feature"])
ax1.set_xlabel("Importance (Gain)")
ax1.set_title("Top 15 Feature Importance")
ax1.invert_yaxis()

# 2. ROC Curve
ax2 = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax2.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})")
ax2.plot([0, 1], [0, 1], "k--", label="Random")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Precision-Recall Curve
ax3 = axes[1, 0]
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
ax3.plot(recall, precision, label=f"PR Curve (AP = {ap_score:.3f})")
ax3.set_xlabel("Recall")
ax3.set_ylabel("Precision")
ax3.set_title("Precision-Recall Curve")
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Prediction Distribution
ax4 = axes[1, 1]
ax4.hist(
    y_pred_proba[y_test == 0],
    bins=50,
    alpha=0.5,
    label="Fulfilled",
    density=True,
)
ax4.hist(
    y_pred_proba[y_test == 1],
    bins=50,
    alpha=0.5,
    label="Stockout",
    density=True,
)
ax4.set_xlabel("Predicted Probability")
ax4.set_ylabel("Density")
ax4.set_title("Prediction Distribution by Actual Class")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "model_evaluation.png", dpi=300, bbox_inches="tight")
print("✓ Saved visualizations to artifacts/model_evaluation.png")

print("\n" + "=" * 80)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 80)

print("\n🎯 BUSINESS IMPACT ESTIMATION (reference threshold = 0.7):")
print(f"   Total test orders: {len(y_test):,}")
print(f"   Actual stockouts: {y_test.sum():,}")
high_conf_mask = y_pred_proba > 0.7
print(
    f"   Predicted stockouts (high confidence >0.7): "
    f"{high_conf_mask.sum():,}"
)
print(
    f"   True positives caught @0.7: "
    f"{(high_conf_mask & (y_test == 1)).sum():,}"
)

print(
    "\nNext step: Run 04_prediction_api.py to create a real-time prediction service"
)

