import numpy as np


class kNNClassifier:
    """
    k-Nearest Neighbors classifier
    """

    def __init__(self, k: int):
        """
        Creates an instance of the `kNNClassifier` class
        """
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the kNN classifier model

        Parameters:
            X (np.ndarray): Design matrix of shape (N, D), where D = feature dimensionality
            y (np.ndarray): Class labels of shape (N,)
        """
        self.num_samples, self.feature_dim = X.shape
        self.X, self.y = X, y

    def predict(self, newX: np.ndarray) -> np.ndarray:
        """
        Predicts class for each new data point

        Parameters:
            newX (np.ndarray): New matrix of shape (M, D)

        Returns:
            np.ndarray: Class labels of shape (M,)
        """
        M, _ = newX.shape

        # L2 distance of (_, N, D) - (M, _, D) -> broadcasting to (M, N, D)
        batch_differences = self.X[None, ...] - newX[:, None, ...]

        distances = np.linalg.norm(
            batch_differences, axis=2
        )  # L2 norm, distances of shape (M, N)
        topk_indices = np.argsort(distances, axis=1)[:, : self.k]  # (M, k)

        predicted_labels = np.empty(M, dtype=self.y.dtype)

        for i in range(M):
            predicted_labels[i] = np.argmax(np.bincount(self.y[topk_indices[i]]))

        return predicted_labels
