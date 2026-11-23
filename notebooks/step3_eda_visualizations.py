import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

os.makedirs("visualizations", exist_ok=True)
sns.set_theme(style="whitegrid")

df_train = pd.read_csv(os.path.join("data", "cleaned_train.csv"))
df_test = pd.read_csv(os.path.join("data", "cleaned_test.csv"))
df = pd.concat([df_train, df_test], ignore_index=True)

df["account_type"] = df["fake"].map({0: "Genuine", 1: "Fake"})

# 1. PIE CHART: Class Distribution
plt.figure(figsize=(6, 6))
counts = df["account_type"].value_counts()
plt.pie(
    counts,
    labels=counts.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=["#2ecc71", "#e74c3c"],
    explode=(0.03, 0.03),
)
plt.title("Dataset Class Distribution: Genuine vs. Fake Profiles")
plt.savefig("visualizations/1_pie_chart_distribution.png", bbox_inches="tight", dpi=150)
plt.close()

# 2. BAR GRAPH: Threat & Identity Signals by Account Type
signal_cols = ["profile_pic", "external_url", "private"]
melted = df.melt(id_vars="account_type", value_vars=signal_cols, var_name="Signal", value_name="Proportion")
plt.figure(figsize=(8, 5))
sns.barplot(data=melted, x="Signal", y="Proportion", hue="account_type", palette=["#2ecc71", "#e74c3c"])
plt.title("Identity & Profile Signals: Genuine vs. Fake Accounts")
plt.ylabel("Proportion of Accounts (0 to 1)")
plt.savefig("visualizations/2_bar_graph_signals.png", bbox_inches="tight", dpi=150)
plt.close()

# 3. CORRELATION HEATMAP
plt.figure(figsize=(12, 9))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig("visualizations/3_correlation_heatmap.png", bbox_inches="tight", dpi=150)
plt.close()

# 4. BOX PLOTS: Behavioral Ratios & Outliers
box_features = ["follower_following_ratio", "posts_follower_ratio", "description_length", "nums/length_username"]
fig, axes = plt.subplots(1, len(box_features), figsize=(16, 5))
for ax, feat in zip(axes, box_features):
    sns.boxplot(data=df, x="account_type", y=feat, ax=ax, hue="account_type", palette=["#2ecc71", "#e74c3c"], legend=False)
    ax.set_title(feat)
    ax.set_xlabel("")
    if "ratio" in feat:
        ax.set_yscale("symlog")
plt.suptitle("Behavioral Feature Distributions & Outliers (Box Plots)", y=1.02)
plt.tight_layout()
plt.savefig("visualizations/4_box_plots_outliers.png", bbox_inches="tight", dpi=150)
plt.close()

# 5. SCATTER PLOT: Followers vs. Following (Follow-Farm Detection)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="num_follows",
    y="num_followers",
    hue="account_type",
    palette={"Genuine": "#2ecc71", "Fake": "#e74c3c"},
    alpha=0.7,
)
plt.xscale("symlog")
plt.yscale("symlog")
plt.title("Followers vs. Following (Log Scale) - Coordinated Bot Pattern")
plt.xlabel("Following Count (symlog)")
plt.ylabel("Follower Count (symlog)")
plt.savefig("visualizations/5_scatter_followers_following.png", bbox_inches="tight", dpi=150)
plt.close()

# 6. CLUSTERING: PCA 2D Projection + K-Means Clusters
feature_cols = [c for c in numeric_df.columns if c != "fake"]
X_scaled = StandardScaler().fit_transform(df[feature_cols])

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["account_type"], palette={"Genuine": "#2ecc71", "Fake": "#e74c3c"}, ax=axes[0], alpha=0.7)
axes[0].set_title("PCA 2D Projection (Actual Labels)")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="viridis", ax=axes[1], alpha=0.7)
axes[1].set_title("K-Means Unsupervised Clusters (k=2)")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")

plt.tight_layout()
plt.savefig("visualizations/6_clustering_pca_kmeans.png", bbox_inches="tight", dpi=150)
plt.close()

print("All 6 EDA visualizations saved successfully inside 'visualizations/'!")
