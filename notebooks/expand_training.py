import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("=== EXPANDING TRAINING DATA & ADAPTING PRODUCTION MODELS ===")

scaler = joblib.load(os.path.join("models", "scaler.joblib"))
feature_cols = joblib.load(os.path.join("models", "feature_cols.joblib"))
tfidf = joblib.load(os.path.join("models", "tfidf.joblib"))
svd = joblib.load(os.path.join("models", "svd.joblib"))
temporal_mlp = joblib.load(os.path.join("models", "temporal_mlp.joblib"))
graph_scaler = joblib.load(os.path.join("models", "graph_scaler.joblib"))
knn_graph = joblib.load(os.path.join("models", "knn_graph.joblib"))
train_ff_ratios = joblib.load(os.path.join("models", "train_ff_ratios.joblib"))

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

# 1. Generate realistic casual & personal real-user profiles so normal humans aren't flagged
rng = np.random.default_rng(101)
n_real = 90
real_records = []
for _ in range(n_real):
    flwrs = int(rng.integers(45, 850))
    flwng = int(rng.integers(80, 950))
    posts = int(rng.integers(0, 65))
    pic = int(rng.choice([1, 1, 1, 0], p=[0.85, 0.05, 0.05, 0.05]))
    desc = int(rng.integers(0, 95))
    priv = int(rng.choice([0, 1], p=[0.45, 0.55]))
    url = int(rng.choice([0, 1], p=[0.8, 0.2]))
    u_nums = float(rng.uniform(0.0, 0.28))  # e.g., birth years like _03 or 2004
    f_words = int(rng.choice([1, 2, 3], p=[0.25, 0.65, 0.10]))
    real_records.append({
        "profile_pic": pic,
        "nums/length_username": u_nums,
        "fullname_words": f_words,
        "nums/length_fullname": 0.0,
        "name==username": 0,
        "description_length": desc,
        "external_url": url,
        "private": priv,
        "num_posts": posts,
        "num_followers": flwrs,
        "num_follows": flwng,
        "fake": 0,
    })

real_aug_df = pd.DataFrame(real_records)
real_aug_df.to_csv(os.path.join("data", "realistic_human_augmentation.csv"), index=False)

base_train = pd.read_csv(os.path.join("data", "cleaned_train.csv"))
X_base = extract_full_features(base_train)
y_base = base_train["fake"].astype(int).values

X_real_aug = extract_full_features(real_aug_df)
y_real_aug = real_aug_df["fake"].astype(int).values

X_list = [X_base, X_real_aug]
y_list = [y_base, y_real_aug]

# 2. Load any Moderator Feedback Overrides (e.g., your own profile) and weight them 35x
fb_path = os.path.join("data", "moderator_feedback_buffer.csv")
if os.path.exists(fb_path):
    fb_df = pd.read_csv(fb_path)
    if len(fb_df) > 0:
        X_fb = fb_df[feature_cols] if all(c in fb_df.columns for c in feature_cols) else extract_full_features(fb_df)
        y_fb = fb_df["fake"].astype(int).values
        X_list.append(pd.concat([X_fb] * 35, ignore_index=True))
        y_list.append(np.tile(y_fb, 35))
        print(f"Loaded {len(fb_df)} moderator feedback sample(s) with 35x adaptation weight.")

X_all = pd.concat(X_list, ignore_index=True)
y_all = np.concatenate(y_list)

X_all_sc = pd.DataFrame(scaler.transform(X_all), columns=feature_cols)
X_noisy = pd.DataFrame(X_all_sc.values + rng.normal(0, 0.16, size=X_all_sc.shape), columns=feature_cols)
X_train_final = pd.concat([X_all_sc, X_noisy], ignore_index=True)
y_train_final = np.concatenate([y_all, y_all])

# Update Continual Learner (SGD)
sgd = joblib.load(os.path.join("models", "continual_learner_sgd_w8.joblib"))
sgd.partial_fit(X_all_sc, y_all, classes=np.array([0, 1]))
joblib.dump(sgd, os.path.join("models", "continual_learner_sgd_w8.joblib"))

# Retrain Production Adversarial Meta-Ensemble
new_ens = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=220, random_state=42)),
        ("xgb", XGBClassifier(n_estimators=220, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=42)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42)),
    ],
    voting="soft",
    weights=[1.0, 1.5, 2.0, 1.5],
)
new_ens.fit(X_train_final, y_train_final)
joblib.dump(new_ens, os.path.join("models", "adversarial_meta_ensemble_w9.joblib"))

# Evaluate on test set
test_df = pd.read_csv(os.path.join("data", "cleaned_test.csv"))
X_test_sc = pd.DataFrame(scaler.transform(extract_full_features(test_df)), columns=feature_cols)
y_test = test_df["fake"].values
preds = new_ens.predict(X_test_sc)
probs = new_ens.predict_proba(X_test_sc)[:, 1]

with open(os.path.join("models", "model_registry.json")) as f:
    reg = json.load(f)
reg["version"] = "v2.1.0"
with open(os.path.join("models", "model_registry.json"), "w") as f:
    json.dump(reg, f, indent=2)

print(f"Updated Production Ensemble ({reg['version']}) -> Test Accuracy: {accuracy_score(y_test, preds)*100:.2f}% | Precision: {precision_score(y_test, preds)*100:.2f}% | ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
