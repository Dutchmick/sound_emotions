# Objective
''''
Use created embeddings to classify audio files into 4 clusters 
of emotions. After craeting the clusters, I will assing a label
to each of the created clusters by manuall inspecting a subset
of audio files.
'''

# Load libraries
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Folder locations
folder_interim = '../data/interim/'

# Load data
embeddings = np.load(folder_interim+'embeddings_array.npy')

# Normalize embeddings
scaler = StandardScaler()
normalised_embeddings = scaler.fit_transform(embeddings)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=10)
clusters = kmeans.fit_predict(normalised_embeddings)

# Visualize clusters
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(normalised_embeddings)
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis')
plt.title("Cluster Visualization")
plt.show()

# Manually label clusters based on listening to audio recordings
manual_cluster_labels = {
    0: "Happy",
    1: "Sad",
    2: "Angry",
    3: "Calm"
}

# Apply manual labels to clusters
emotion_labels = np.array([manual_cluster_labels[cluster] for cluster in clusters])

# Export data
np.save(folder_interim+'reduced_embeddings.npy', reduced_embeddings)
np.save(folder_interim+'clusters_array.npy', clusters)
np.save(folder_interim+'manual_labels.npy', emotion_labels)
