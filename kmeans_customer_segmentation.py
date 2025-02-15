# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'Mall_Customers.csv'  # Update the path if necessary
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Dataset Head:")
print(data.head())

# Select relevant features for clustering
# Here, we use 'Annual Income (k$)' and 'Spending Score (1-100)'
X = data.iloc[:, [3, 4]].values

# Standardize the features (important for K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
wcss = []  # Within-Cluster-Sum-of-Squares
max_clusters = 10  # Maximum number of clusters to try

for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # Inertia is the WCSS

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(range(1, max_clusters + 1))
plt.grid()
plt.show()

# Based on the Elbow Method, choose the optimal number of clusters
# Look for the "elbow" point in the graph
optimal_clusters = int(input("Enter the optimal number of clusters based on the Elbow Method: "))

# Apply K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add the cluster labels to the original dataset
data['Cluster'] = clusters

# Save the clustered data to a new CSV file
data.to_csv('clustered_customers.csv', index=False)
print(f"Clustered data saved to 'clustered_customers.csv' with {optimal_clusters} clusters.")

# Visualize the clusters
plt.figure(figsize=(10, 6))
for i in range(optimal_clusters):
    plt.scatter(X_scaled[clusters == i, 0], X_scaled[clusters == i, 1], label=f'Cluster {i + 1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Customer Segments (Clusters)')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()
plt.show()
# Create the submission DataFrame
submission = pd.DataFrame({
    'CustomerID': data.iloc[:, 0],  # Assuming the first column is CustomerID
    'Cluster': clusters
})

# Save submission file as CSV
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")
