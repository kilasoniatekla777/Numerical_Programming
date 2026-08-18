#  Import libraries
# ======================================
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ======================================
# Load your CSV
# ======================================
df = pd.read_csv('FOOD-DATA-GROUP1.csv')

# Select numeric columns only (drop text/non-numeric columns)
numeric_cols = df.select_dtypes(include=['number']).columns
X = df[numeric_cols]

# ======================================
#Scale numeric features
# ======================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================================
#K-Means Clustering
# ======================================
kmeans = KMeans(n_clusters=3, random_state=0)
df['cluster_kmeans'] = kmeans.fit_predict(X_scaled)

print("✅ K-Means Cluster assignments:")
print(df[['food', 'cluster_kmeans']].head())

# Show foods grouped by K-Means cluster
for cluster in sorted(df['cluster_kmeans'].unique()):
    print(f"\nK-Means Cluster {cluster}:")
    print(df[df['cluster_kmeans'] == cluster]['food'].to_list())

# ======================================
#  DBSCAN Clustering
# ======================================
dbscan = DBSCAN(eps=1.5, min_samples=5)
df['cluster_dbscan'] = dbscan.fit_predict(X_scaled)

print("\n✅ DBSCAN Cluster assignments:")
print(df[['food', 'cluster_dbscan']].head())

# Show foods grouped by DBSCAN cluster
for cluster in sorted(df['cluster_dbscan'].unique()):
    print(f"\nDBSCAN Cluster {cluster}:")
    print(df[df['cluster_dbscan'] == cluster]['food'].to_list())

# ======================================
#  2D PCA Visualization
# ======================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(14, 5))

# K-Means plot
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster_kmeans'], cmap='viridis', s=80)
plt.title('K-Means Clusters of Foods')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

# DBSCAN plot
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster_dbscan'], cmap='rainbow', s=80)
plt.title('DBSCAN Clusters of Foods')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

plt.tight_layout()
plt.show()