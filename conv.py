import numpy as np


class Conv2d:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int],
        stride: int = 1,
        padding: int = 0,
    ):
        """
        Creates a 2d conv layer
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Weight matrix of shape: (C_out, C_in, K_h, K_w)
        self.W = np.random.randn(
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )

        # Bias vector of shape (C_out,)
        self.bias = np.zeros(shape=(self.out_channels,))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for Conv2d

        {H',W'} = floor[ ({H,W} - K_{h,w} + 2P) / S ] + 1

        Parameters:
            x (np.ndarray): Batch of shape (B, C_in, H, W)

        Returns:
            np.ndarray: Output batch of shape (B, C_out, H', W')
        """
        B, _, H, W = x.shape
        K_h, K_w = self.kernel_size

        # Output height
        H_prime = int((H - K_h + 2 * self.padding) / self.stride) + 1
        W_prime = int((W - K_w + 2 * self.padding) / self.stride) + 1

        out = np.zeros(shape=(B, self.out_channels, H_prime, W_prime))

        # Pad x with appropriate padding
        x = np.pad(
            x,
            pad_width=(
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
        )

        for b in range(B):
            for c in range(self.out_channels):

                curr_kernel = self.W[
                    c
                ]  # Grab filter for current channel: (C_in, K_h, K_w)
                curr_bias = self.bias[c]  # Grab current bias for channel: (1,)

                for h in range(H_prime):
                    for w in range(W_prime):
                        h_start, w_start = h * self.stride, w * self.stride
                        input_slice = x[
                            b, :, h_start : h_start + K_h, w_start : w_start + K_w
                        ]  # Of shape (C_in, K_h, K_w)

                        hadamard_prod = np.sum(curr_kernel * input_slice) + curr_bias

                        out[b, c, h, w] = hadamard_prod
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Implict forward pass
        """
        return self.forward(x=x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels})"


class MaxPool2d:
    """
    Max pooling layer
    """

    def __init__(self, kernel_size: tuple[int, int], stride: int = 1):
        """
        Creates an instance of the `MaxPool2d` class
        """
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for the max pooling layer

        Parameters:
            x (np.ndarray): Input batch of shape (B, C, H, W)

        Returns:
            np.ndarray: Downsampled batch of shape (B, C, H', W')
        """
        B, C, H, W = x.shape
        K_h, K_w = self.kernel_size

        H_prime = int((H - K_h) / self.stride) + 1
        W_prime = int((W - K_w) / self.stride) + 1

        out = np.empty(shape=(B, C, H_prime, W_prime))

        for b in range(B):
            for c in range(C):
                for h in range(H_prime):
                    for w in range(W_prime):
                        h_start, w_start = h * self.stride, w * self.stride
                        input_slice = x[
                            b, c, h_start : h_start + K_h, w_start : w_start + K_w
                        ]  # Of shape (K_h, K_w)

                        out[b, c, h, w] = np.max(input_slice)
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x=x)


if __name__ == "__main__":
    layer = Conv2d(
        in_channels=3, out_channels=8, kernel_size=(3, 3), stride=2, padding=1
    )
    x = np.ones((2, 3, 32, 32))
    out = layer.forward(x)
    out.shape
