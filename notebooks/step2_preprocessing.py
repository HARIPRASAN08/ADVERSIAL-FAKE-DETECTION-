import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data_dir = "data"
csv_files = [
    f for f in os.listdir(data_dir)
    if f.endswith(".csv") and not f.startswith(("cleaned_", "adversarial_"))
]

if not csv_files:
    raise FileNotFoundError("No raw CSV files found in 'data/'. Please paste your Kaggle CSV files there.")

print("Scanning CSV files in data/:")
labeled_dfs = []
target_candidates = ["fake", "is_fake", "label", "target", "class", "is_bot", "bot"]

for f in sorted(csv_files):
    temp_df = pd.read_csv(os.path.join(data_dir, f))
    temp_df.columns = [c.strip().lower().replace(" ", "_").replace("#", "num_") for c in temp_df.columns]
    print(f" - {f}: shape={temp_df.shape}, columns={list(temp_df.columns)}")

    matched_target = next((col for col in target_candidates if col in temp_df.columns), None)
    if matched_target is not None and temp_df[matched_target].nunique() == 2:
        temp_df = temp_df.rename(columns={matched_target: "fake"})
        labeled_dfs.append((f, temp_df))

if not labeled_dfs:
    raise ValueError(
        "None of the CSV files in 'data/' have a binary label column ('fake' or 'is_fake'). "
        "Please make sure you copied 'train.csv' (not just the unlabeled 'test.csv') into 'data/'."
    )

# Prefer train.csv if available, otherwise pick the largest labeled dataset
labeled_dfs.sort(key=lambda item: (0 if "train" in item[0].lower() else 1, -len(item[1])))
primary_name, df = labeled_dfs[0]

# Combine if another labeled CSV has identical columns
for name, other_df in labeled_dfs[1:]:
    if list(other_df.columns) == list(df.columns):
        df = pd.concat([df, other_df], ignore_index=True)
        print(f"Combined {primary_name} + {name} (identical columns).")

print(f"\nSelected dataset: '{primary_name}' | Raw shape: {df.shape}")

# 1. Data Cleaning
df = df.drop_duplicates().reset_index(drop=True)

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna("")

# Ensure target 'fake' is binary 0/1 integer
if not pd.api.types.is_numeric_dtype(df["fake"]):
    df["fake"] = df["fake"].astype(str).str.lower().map({"fake": 1, "true": 1, "1": 1, "yes": 1}).fillna(0).astype(int)
else:
    df["fake"] = df["fake"].astype(int)

# 2. PDF-Inspired Feature Engineering (Behavioral & Graph Ratios)
follower_col = next((c for c in ["num_followers", "followers_count", "followers"] if c in df.columns), None)
following_col = next((c for c in ["num_follows", "following_count", "following", "follows"] if c in df.columns), None)
posts_col = next((c for c in ["num_posts", "posts_count", "posts"] if c in df.columns), None)

if follower_col and following_col:
    df["follower_following_ratio"] = df[follower_col] / (df[following_col] + 1.0)
if posts_col and follower_col:
    df["posts_follower_ratio"] = df[posts_col] / (df[follower_col] + 1.0)
if {"profile_pic", "description_length", "external_url"}.issubset(df.columns):
    df["profile_completeness"] = (
        df["profile_pic"]
        + (df["description_length"] > 0).astype(int)
        + df["external_url"]
    )

print(f"Cleaned dataset shape: {df.shape}")
print("Target 'fake' distribution (0 = Genuine, 1 = Fake):\n", df["fake"].value_counts())

# 3. Stratified Train / Clean-Test Split
X = df.drop(columns=["fake"])
y = df["fake"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Create Adversarial Evasion Test Set (Noise Injection)
rng = np.random.default_rng(42)
X_test_adv = X_test.copy()
numeric_cols = X_test_adv.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    if X_test_adv[col].nunique() > 2:
        noise = rng.normal(loc=0.0, scale=0.15 * (X_test_adv[col].std() + 1e-5), size=len(X_test_adv))
        X_test_adv[col] = np.maximum(0, X_test_adv[col] + noise)

# 5. Save Processed Splits
X_train.assign(fake=y_train).to_csv(os.path.join(data_dir, "cleaned_train.csv"), index=False)
X_test.assign(fake=y_test).to_csv(os.path.join(data_dir, "cleaned_test.csv"), index=False)
X_test_adv.assign(fake=y_test).to_csv(os.path.join(data_dir, "adversarial_test.csv"), index=False)

print("\nSuccess! Saved cleaned_train.csv, cleaned_test.csv, and adversarial_test.csv in data/.")