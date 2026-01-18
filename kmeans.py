import numpy as np


class KMeans:
    """
    Runs K-means clustering logic
    """

    def __init__(self, k: int, n_iterations: int = 100, tol: float = 1e-8):
        """
        Creates a KMeans clustering instance

        Parameters:
            k (int): Number of clusters
            n_iterations (int): Number of iterations to run EM for
            tol (float): Stopping criterion tolerance
        """
        self.k = k
        self.n_iterations = n_iterations
        self.tol = tol

        self._n_iter = 0

        self.old_centroids = None
        self.centroids = None

    def fit(self, X: np.ndarray) -> None:
        """
        Runs clustering algorithm for `n_iterations`

        Parameters:
            x (np.ndarray): Feature matrix of shape (N, D)
        """
        self.N, self.D = X.shape

        # Matrix of shape (k, D)
        self.centroids = self._initialize_centroids()

        while not self.should_stop:
            self.old_centroids = self.centroids.copy()

            # E: Calculate distances and re-cluster

            # Broadcast from (N, _, D) and (_, k, D) -> (N, k)
            distances = np.linalg.norm(
                X[:, None, ...] - self.centroids[None, ...], axis=2
            )
            labels = np.argmin(distances, axis=1)

            # M: Compute new centroids and update
            for i in range(self.k):
                cluster = X[labels == i]  # Cluster i data points
                if cluster.shape[0] > 0:
                    # Only update centroid when there are datapoints for that cluster
                    self.centroids[i] = np.mean(cluster, axis=0)

            # Update iteration counter
            self._n_iter += 1

        print(f"Clustering algorithm finished after {self._n_iter} iterations")

    def predict(self, X: np.ndarray) -> int:
        """
        Runs prediction after being fit
        """
        distances = np.linalg.norm(X[:, None, ...] - self.centroids[None, ...], axis=2)

        return np.argmin(distances, axis=1)

    @property
    def should_stop(self) -> bool:
        """
        Returns whether the fit should stop
        """
        too_many_steps = self._n_iter >= self.n_iterations
        if self.old_centroids is None:
            return too_many_steps
        return too_many_steps or np.allclose(
            self.centroids, self.old_centroids, atol=self.tol
        )

    def _initialize_centroids(self) -> None:
        """
        Initializes the starting cluster centroids
        """
        return np.random.randn(self.k, self.D)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k})"
