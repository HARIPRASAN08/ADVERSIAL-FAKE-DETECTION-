import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

os.makedirs("models", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)
sns.set_theme(style="whitegrid")

# 1. Load Processed Datasets from Step 2
train_df = pd.read_csv(os.path.join("data", "cleaned_train.csv"))
test_df = pd.read_csv(os.path.join("data", "cleaned_test.csv"))
adv_df = pd.read_csv(os.path.join("data", "adversarial_test.csv"))

feature_cols = [c for c in train_df.columns if c != "fake"]
X_train, y_train = train_df[feature_cols], train_df["fake"]
X_test, y_test = test_df[feature_cols], test_df["fake"]
X_adv, y_adv = adv_df[feature_cols], adv_df["fake"]

# Standardize features while preserving DataFrame column names
scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
X_adv_sc = pd.DataFrame(scaler.transform(X_adv), columns=feature_cols)

# 2. Generate Adversarial Training Augmentation (PDF Week 7: Adversarial Training)
rng = np.random.default_rng(42)
noise = rng.normal(0.0, 0.15, size=X_train_sc.shape)
X_train_noisy = pd.DataFrame(X_train_sc.values + noise, columns=feature_cols)
X_train_aug = pd.concat([X_train_sc, X_train_noisy], ignore_index=True)
y_train_aug = pd.concat([y_train, y_train], ignore_index=True)

# 3. Define Algorithms (HR Steps 4 & 5 + PDF Continual & Ensemble Learning)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=42),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
}

# Train standard models on clean training set
for name, model in models.items():
    model.fit(X_train_sc, y_train)

# Train Continual Learning model via mini-batch partial_fit (PDF Week 8)
sgd_continual = SGDClassifier(loss="log_loss", random_state=42)
batch_size = max(1, len(X_train_sc) // 4)
classes = np.array([0, 1])
for start in range(0, len(X_train_sc), batch_size):
    sgd_continual.partial_fit(
        X_train_sc.iloc[start : start + batch_size],
        y_train.iloc[start : start + batch_size],
        classes=classes,
    )
models["Continual Learner (SGD)"] = sgd_continual

# Train Adversarial-Resilient Soft-Voting Ensemble on clean + noisy augmented data (PDF Weeks 7 & 9)
adv_ensemble = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("xgb", XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=42)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)),
    ],
    voting="soft",
)
adv_ensemble.fit(X_train_aug, y_train_aug)
models["Adversarial Ensemble"] = adv_ensemble

# 4. Evaluate All Models on Clean & Adversarial Test Sets (HR Step 5)
results = []
for name, model in models.items():
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    y_adv_pred = model.predict(X_adv_sc)

    results.append({
        "Algorithm": name,
        "Clean Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Adversarial Accuracy": round(accuracy_score(y_adv, y_adv_pred), 4),
    })

results_df = pd.DataFrame(results).sort_values(by=["F1-Score", "Adversarial Accuracy"], ascending=False)
print("\n=== MODEL PERFORMANCE COMPARISON (CLEAN vs. ADVERSARIAL) ===")
print(results_df.to_string(index=False))

# 5. Plot Confusion Matrices for All Models (Completes HR Step 3)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for idx, (name, model) in enumerate(models.items()):
    cm = confusion_matrix(y_test, model.predict(X_test_sc))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Genuine", "Fake"])
    disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
    axes[idx].set_title(name)
    axes[idx].grid(False)

plt.suptitle("Confusion Matrices Across All Algorithms (HR Step 3 & Step 5)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("visualizations/7_confusion_matrices.png", bbox_inches="tight", dpi=150)
plt.close()

# 6. Plot Algorithm Comparison Chart (Clean F1 vs. Adversarial Accuracy)
melted_res = results_df.melt(
    id_vars="Algorithm",
    value_vars=["Clean Accuracy", "F1-Score", "Adversarial Accuracy"],
    var_name="Metric",
    value_name="Score",
)
plt.figure(figsize=(11, 5))
sns.barplot(data=melted_res, x="Algorithm", y="Score", hue="Metric", palette="Set2")
plt.ylim(0.5, 1.05)
plt.xticks(rotation=20, ha="right")
plt.title("Algorithm Benchmark: Clean Performance vs. Adversarial Robustness")
plt.tight_layout()
plt.savefig("visualizations/8_model_comparison.png", bbox_inches="tight", dpi=150)
plt.close()

# 7. Generate SHAP Explainable AI Summary Plot (PDF Week 10)
explainer = shap.TreeExplainer(models["XGBoost"])
shap_values = explainer.shap_values(X_test_sc)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_sc, show=False)
plt.title("SHAP Feature Impact on Fake Profile Detection (Explainable AI)")
plt.tight_layout()
plt.savefig("visualizations/9_shap_feature_importance.png", bbox_inches="tight", dpi=150)
plt.close()

# 8. Save Models & Metadata for Step 6 (Streamlit Dashboard)
joblib.dump(scaler, os.path.join("models", "scaler.joblib"))
joblib.dump(feature_cols, os.path.join("models", "feature_cols.joblib"))
for name, model in models.items():
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, os.path.join("models", f"{safe_name}.joblib"))

results_df.to_csv(os.path.join("models", "model_comparison.csv"), index=False)
print("\nSaved all trained models in 'models/' and 3 evaluation charts in 'visualizations/'!")
