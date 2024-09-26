import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, init='random', max_iter=300, manual_centroids=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None
        self.manual_centroids = manual_centroids  # To be provided by the user for manual init

    def initialize_centroids(self, X):
        if self.init == 'random':
            # Randomly pick k centroids from the data
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            return X[indices]
        elif self.init == 'farthest':
            return self.farthest_first_initialization(X)
        elif self.init == 'kmeans++':
            return self.kmeans_plus_plus_initialization(X)
        elif self.init == 'manual' and self.manual_centroids is not None:
            # Use the manually provided centroids
            return np.array(self.manual_centroids)
        else:
            raise ValueError("Unsupported initialization method or manual centroids not provided")

    def farthest_first_initialization(self, X):
        # Farthest first initialization
        centroids = [X[np.random.choice(X.shape[0])]]
        for _ in range(1, self.n_clusters):
            distances = np.min([np.linalg.norm(X - c, axis=1) for c in centroids], axis=0)
            next_centroid = X[np.argmax(distances)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def kmeans_plus_plus_initialization(self, X):
        # KMeans++ initialization
        centroids = [X[np.random.choice(X.shape[0])]]
        for _ in range(1, self.n_clusters):
            distances = np.min([np.linalg.norm(X - c, axis=1)**2 for c in centroids], axis=0)
            prob_dist = distances / np.sum(distances)
            next_centroid = X[np.random.choice(X.shape[0], p=prob_dist)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def assign_clusters(self, X):
        # Assign points to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels):
        # Recompute centroids as the mean of all points in a cluster
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

    def fit(self, X):
        # Step 1: Initialize centroids
        self.centroids = self.initialize_centroids(X)
        
        for _ in range(self.max_iter):
            # Step 2: Assign clusters
            labels = self.assign_clusters(X)
            # Step 3: Update centroids
            new_centroids = self.update_centroids(X, labels)
            
            # Check for convergence
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        
        return self.centroids, labels
