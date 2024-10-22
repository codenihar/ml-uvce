import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Step 1: Create dummy dataset (directly in Python)
# Two clusters: one around (1,2) and another around (8,8)
X = np.array([[1.0, 2.0],
              [1.5, 1.8],
              [5.0, 8.0],
              [8.0, 8.0],
              [1.0, 0.6],
              [9.0, 11.0],
              [8.0, 2.0],
              [10.0, 2.0],
              [9.0, 3.0]])

# Step 2: Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Step 3: Apply Gaussian Mixture Model (EM algorithm)
gmm = GaussianMixture(n_components=2, random_state=42)
gmm_labels = gmm.fit_predict(X)

# Step 4: Plotting the clustering results
plt.figure(figsize=(10, 5))

# Plot the KMeans clusters
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', marker='o', s=100)
plt.title('K-Means Clustering')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')

# Plot the GMM (EM) clusters
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis', marker='o', s=100)
plt.title('GMM (EM Algorithm) Clustering')
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], s=300, c='red', marker='X')

plt.show()

# Step 5: Compare the results
print("KMeans cluster centers:")
print(kmeans.cluster_centers_)

print("GMM Means (cluster centers):")
print(gmm.means_)
