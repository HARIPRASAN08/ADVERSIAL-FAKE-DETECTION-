import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

print("=== FULL PROJECT ACCURACY & PERFORMANCE AUDIT ===")

scaler = joblib.load(os.path.join("models", "scaler.joblib"))
feature_cols = joblib.load(os.path.join("models", "feature_cols.joblib"))
tfidf = joblib.load(os.path.join("models", "tfidf.joblib"))
svd = joblib.load(os.path.join("models", "svd.joblib"))
temporal_mlp = joblib.load(os.path.join("models", "temporal_mlp.joblib"))
graph_scaler = joblib.load(os.path.join("models", "graph_scaler.joblib"))
knn_graph = joblib.load(os.path.join("models", "knn_graph.joblib"))
train_ff_ratios = joblib.load(os.path.join("models", "train_ff_ratios.joblib"))

with open(os.path.join("models", "model_registry.json")) as f:
    registry = json.load(f)

def extract_full_features(df_in):
    df = df_in.copy()
    if "num_follows" not in df.columns:
        df["num_follows"] = df["num_followers"]
    if "follower_following_ratio" not in df.columns:
        df["follower_following_ratio"] = df["num_followers"] / (df["num_follows"] + 1.0)
    if "posts_follower_ratio" not in df.columns:
        df["posts_follower_ratio"] = df["num_posts"] / (df["num_followers"] + 1.0)
    if "profile_completeness" not in df.columns:
        df["profile_completeness"] = (
            df["profile_pic"].astype(int)
            + (df["description_length"] > 0).astype(int)
            + df["external_url"].astype(int)
        )

    texts = []
    for _, r in df.iterrows():
        num_tag = "high_digits" if r["nums/length_username"] > 0.25 else "clean_handle"
        name_tag = "same_name" if r["name==username"] == 1 else "distinct_name"
        bio_tag = f"bio_len_{int(r['description_length'] // 20)}"
        url_tag = "has_link" if r["external_url"] == 1 else "no_link"
        texts.append(f"{num_tag} {name_tag} {bio_tag} {url_tag} words_{int(r['fullname_words'])}")
    t_emb = svd.transform(tfidf.transform(texts))
    for i in range(4):
        df[f"text_emb_{i}"] = t_emb[:, i]

    df["vision_identity_score"] = (
        0.50 * df["profile_pic"]
        + 0.30 * (1.0 - df["name==username"])
        + 0.20 * (df["profile_completeness"] / 3.0)
    )

    hours = np.arange(24)
    human_curve = np.exp(-0.5 * ((hours - 10) / 3.0) ** 2) + 1.2 * np.exp(-0.5 * ((hours - 20) / 3.0) ** 2)
    human_curve /= human_curve.sum()
    seqs, entropies, bursts = [], [], []
    for _, r in df.iterrows():
        bf = min(1.0, r["posts_follower_ratio"] / (r["follower_following_ratio"] + 0.5))
        bot_curve = np.roll(human_curve, 12) * 0.4 + (np.ones(24) / 24.0) * 0.6
        seq = (1.0 - 0.5 * bf) * human_curve + (0.5 * bf) * bot_curve
        seq = np.clip(seq, 1e-6, 1.0)
        seq /= seq.sum()
        seqs.append(seq)
        entropies.append(-np.sum(seq * np.log2(seq)))
        bursts.append(bf)
    seqs = np.array(seqs)
    df["temporal_circadian_entropy"] = entropies
    df["temporal_burst_score"] = bursts
    df["temporal_risk_emb"] = temporal_mlp.predict_proba(seqs)[:, 1]

    graph_cols = ["num_posts", "num_followers", "num_follows", "follower_following_ratio", "posts_follower_ratio"]
    G_mat = graph_scaler.transform(df[graph_cols])
    dists, idxs = knn_graph.kneighbors(G_mat)
    idxs, dists = idxs[:, :6], dists[:, :6]
    weights = 1.0 / (dists + 1e-3)
    weights /= weights.sum(axis=1, keepdims=True)
    df["graph_neighbor_ff_ratio"] = (train_ff_ratios[idxs] * weights).sum(axis=1)
    df["graph_cluster_density"] = 1.0 / (dists.mean(axis=1) + 0.1)

    modal_gap = np.abs(df["vision_identity_score"] - (1.0 - df["temporal_risk_emb"]))
    fp_mismatch = np.abs(np.tanh(df["follower_following_ratio"]) - np.tanh(df["posts_follower_ratio"]))
    df["cross_modal_contradiction"] = 0.5 * modal_gap + 0.5 * fp_mismatch

    return df[feature_cols]

# Load evaluation splits
test_df = pd.read_csv(os.path.join("data", "cleaned_test.csv"))
adv_df = pd.read_csv(os.path.join("data", "adversarial_test.csv"))

X_test_sc = pd.DataFrame(scaler.transform(extract_full_features(test_df)), columns=feature_cols)
y_test = test_df["fake"].astype(int).values

X_adv_sc = pd.DataFrame(scaler.transform(extract_full_features(adv_df)), columns=feature_cols)
y_adv = adv_df["fake"].astype(int).values

X_total_sc = pd.concat([X_test_sc, X_adv_sc], ignore_index=True)
y_total = np.concatenate([y_test, y_adv])

models = {
    f"Adversarial Meta-Ensemble ({registry.get('version', 'v2.1.2')})": joblib.load(os.path.join("models", "adversarial_meta_ensemble_w9.joblib")),
    "XGBoost (Multi-Modal)": joblib.load(os.path.join("models", "xgboost_multi_modal.joblib")),
    "Random Forest (Multi-Modal)": joblib.load(os.path.join("models", "random_forest_multi_modal.joblib")),
    "Neural Network (MLP)": joblib.load(os.path.join("models", "neural_network_mlp.joblib")),
    "Logistic Regression": joblib.load(os.path.join("models", "logistic_regression.joblib")),
    "Continual Learner (SGD)": joblib.load(os.path.join("models", "continual_learner_sgd_w8.joblib")),
}

rows = []
for name, model in models.items():
    t0 = time.perf_counter()
    y_pred_clean = model.predict(X_test_sc)
    lat_ms = ((time.perf_counter() - t0) / len(X_test_sc)) * 1000.0
    y_prob_clean = model.predict_proba(X_test_sc)[:, 1]
    y_pred_adv = model.predict(X_adv_sc)
    y_pred_total = model.predict(X_total_sc)

    rows.append({
        "Algorithm": name,
        "Clean Accuracy": f"{accuracy_score(y_test, y_pred_clean)*100:.2f}%",
        "Adversarial Accuracy": f"{accuracy_score(y_adv, y_pred_adv)*100:.2f}%",
        "Total Combined Accuracy": f"{accuracy_score(y_total, y_pred_total)*100:.2f}%",
        "Precision": f"{precision_score(y_test, y_pred_clean)*100:.2f}%",
        "Recall": f"{recall_score(y_test, y_pred_clean)*100:.2f}%",
        "F1-Score": round(f1_score(y_test, y_pred_clean), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob_clean), 4),
        "Latency (ms)": round(lat_ms, 3),
    })

summary_df = pd.DataFrame(rows)
print("\n--- MODEL COMPARISON ACROSS ALL TEST SUITES ---")
print(summary_df.to_string(index=False))

# Detailed breakdown for the active Production Meta-Ensemble
prod_model = list(models.values())[0]
prod_preds = prod_model.predict(X_test_sc)
cm = confusion_matrix(y_test, prod_preds)

print(f"\n--- PRODUCTION ENSEMBLE ({registry.get('version', 'v2.1.2')}) DETAILED REPORT ---")
print(classification_report(y_test, prod_preds, target_names=["REAL (0)", "FAKE (1)"], digits=4))
print("Confusion Matrix (Clean Test Set):")
print(f"  True REAL predicted REAL : {cm[0][0]}")
print(f"  True REAL predicted FAKE : {cm[0][1]}  (False Positives)")
print(f"  True FAKE predicted REAL : {cm[1][0]}  (False Negatives)")
print(f"  True FAKE predicted FAKE : {cm[1][1]}")

# Sync updated metrics into models/model_comparison.csv for Tab 3
summary_df.to_csv(os.path.join("models", "model_comparison.csv"), index=False)
print("\nUpdated models/model_comparison.csv with latest metrics.")
