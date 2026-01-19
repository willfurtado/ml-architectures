import numpy as np


class NaiveSoftmax:
    """
    Naive version of softmax -> WILL OVERFLOW/UNDERFLOW
    """

    def forward(self, logits: np.ndarray) -> np.ndarray:
        """
        Runs softmax function, normalizing logits

        Parameters:
            logits (np.ndarray): Input batch of shape (B, n_features)

        Returns:
            np.ndarray: Output batch of shape (B, n_features)
        """
        numerator = np.exp(logits)  # (B, n_features)
        denominator = np.sum(np.exp(logits), axis=1, keepdims=True)  # (B, 1)

        return numerator / denominator

    def __call__(self, logits: np.ndarray) -> np.ndarray:
        return self.forward(logits=logits)


class StableSoftmax(NaiveSoftmax):
    """
    Stable version of softmax making use of LogSumExp trick
    """

    def forward(self, logits: np.ndarray) -> np.ndarray:
        """
        Runs softmax function, normalizing logits
        """
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)

        return super().forward(logits=shifted_logits)


if __name__ == "__main__":
    x = np.random.randn(8, 10) * 1e3  # Will {under,over}flow

    ns = NaiveSoftmax()
    ss = StableSoftmax()
    ns_out = ns(x)
    ss_out = ss(x)
    ns_out, ss_out
