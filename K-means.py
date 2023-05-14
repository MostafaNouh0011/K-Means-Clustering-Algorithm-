import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


def init_centroids(X, k):
    m, n = X.shape 
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)

    for i in range(k):
        centroids[i, :] = X[idx[i], :]
    return centroids


def find_closest_centroid(X, centroids):
    m = X.shape[0]
    c = centroids.shape[0]
    idx = np.zeros(m)

    for i in range(m): # 0, 1, 2, 3,..., 299
        min_dist = 1000000
        for j in range(c):  # 0, 1, 2
            dist = np.sum((X[i, :] - centroids[j, :])**2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
        
    return idx


def compute_centroids(X, idx, k):
    m, n = X.shape
    centoids = np.zeros((k, n))

    for i in range(k):
        indices = np.where(idx == i)
        centoids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()

    return centoids


def run_k_means(X, initial_centroids, iters):
    m, n = X.shape
    c = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros(m)

    for i in range(iters):
        idx = find_closest_centroid(X, centroids)     # Selection
        centroids = compute_centroids(X, idx, c)      # Displacement

    return idx, centroids


def plot_clusters(idx, centoids):

    cluster1 = X[np.where(idx == 0)[0], :]
    cluster2 = X[np.where(idx == 1)[0], :]
    cluster3 = X[np.where(idx == 2)[0], :]

    fig, ax = plt.subplots(figsize= (9, 6))
    ax.scatter(cluster1[:, 0], cluster1[:, 1], s=30, color='r', label='Cluster 1')
    ax.scatter(centroids[0, 0], centroids[0, 1], s=300, color='r')

    ax.scatter(cluster2[:, 0], cluster2[:, 1], s=30, color='b', label='Cluster 2')
    ax.scatter(centroids[1, 0], centroids[1, 1], s=300, color='b')

    ax.scatter(cluster3[:, 0], cluster3[:, 1], s=30, color='g', label='Cluster 3')
    ax.scatter(centroids[2, 0], centroids[2, 1], s=300, color='g')

    ax.legend()
    plt.show()


# load MATLAB data
data = loadmat('ex7data2.mat')
X = data['X']
# print(X)
# print(X.shape)  # (300, 2)


# Assign random points or centroids
initial_centroids = init_centroids(X, 3)
print(initial_centroids)

# Select each data point to their closest centroid
idx = find_closest_centroid(X, initial_centroids)

# Calculate new centroids
updated_centroids = compute_centroids(X, idx, 3)
print(updated_centroids)


for i in range(6):
    # apply k-means
    idx, centroids = run_k_means(X, initial_centroids, i)

    print('idx \n', idx)
    print('centroids \n', centroids)

    plot_clusters(idx, centroids)
