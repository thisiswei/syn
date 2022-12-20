
from sklearn.cluster import KMeans
import numpy as np

from .data import DataProcessor

# Create a KMeans model with 3 clusters
kmeans = KMeans(n_clusters=3)

dp = DataProcessor()
# Fit the model to the data
kmeans.fit(dp.train_data)

# Predict the cluster labels for each data point
labels = kmeans.predict(dp.train_data)

# Print the cluster labels
print(labels)
