import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, init='random', max_iter=300, manual_centroids=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None
        self.manual_centroids = manual_centroids  # Used for manual initialization
        self.labels = None
        self.current_iter = 0  # Track current iteration number

    def initialize_centroids(self, X):
        if self.init == 'random':
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            return X[indices]
        elif self.init == 'farthest':
            return self.farthest_first_initialization(X)
        elif self.init == 'kmeans++':
            return self.kmeans_plus_plus_initialization(X)
        elif self.init == 'manual' and self.manual_centroids is not None:
            return np.array(self.manual_centroids)
        else:
            raise ValueError("Unsupported initialization method or manual centroids not provided")

    def farthest_first_initialization(self, X):
        centroids = [X[np.random.choice(X.shape[0])]]
        for _ in range(1, self.n_clusters):
            distances = np.min([np.linalg.norm(X - c, axis=1) for c in centroids], axis=0)
            next_centroid = X[np.argmax(distances)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def kmeans_plus_plus_initialization(self, X):
        centroids = [X[np.random.choice(X.shape[0])]]
        for _ in range(1, self.n_clusters):
            distances = np.min([np.linalg.norm(X - c, axis=1)**2 for c in centroids], axis=0)
            prob_dist = distances / np.sum(distances)
            next_centroid = X[np.random.choice(X.shape[0], p=prob_dist)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels):
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

    def partial_fit(self, X):
        if self.centroids is None:
            self.centroids = self.initialize_centroids(X)

        # Step: Assign clusters
        self.labels = self.assign_clusters(X)

        # Step: Update centroids
        new_centroids = self.update_centroids(X, self.labels)

        if np.allclose(self.centroids, new_centroids):  # Use allclose for floating-point comparison
            return self.centroids, self.labels, True  # Converged

        self.centroids = new_centroids
        self.current_iter += 1  # Track the step/iteration number
        return self.centroids, self.labels, False  # Not converged yet

    def fit(self, X):
        """Runs KMeans until convergence or max iterations are reached."""
        if self.centroids is None:
            self.centroids = self.initialize_centroids(X)

        for _ in range(self.max_iter):
            # Assign clusters and update centroids in each iteration
            self.labels = self.assign_clusters(X)
            new_centroids = self.update_centroids(X, self.labels)

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):  # Use allclose for numerical stability
                return self.centroids, self.labels  # Return once converged

            self.centroids = new_centroids

        return self.centroids, self.labels  # Return after max_iter
