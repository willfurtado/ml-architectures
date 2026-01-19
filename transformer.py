import torch
import torch.nn as nn


class ViT(nn.Module):
    """
    Vision transformer module
    """

    def __init__(
        self,
        patch_size: int,
        img_dims: tuple[int, int, int],
        depth: int,
        n_heads: int,
        input_dim: int,
        d_k: int,
        d_v: int,
        mlp_dim: int,
        num_classes: int,
    ):
        """
        Creates an instance of the `ViT` class
        """
        super().__init__()

        self.patch_size = patch_size
        self.C, self.img_h, self.img_w = img_dims
        self.depth = depth
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.d_k = d_k
        self.d_v = d_v
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes

        if (self.img_h % self.patch_size) != 0 or (self.img_w % self.patch_size) != 0:
            raise ValueError(
                f"Image dimensions: {img_dims} must be divisible by patch size: {self.patch_size}"
            )

        # Calculate number of patches (i.e. sequence length)
        num_patches = (self.img_h // self.patch_size) * (self.img_w // self.patch_size)

        # Create patches and linearly project using Conv2d layer
        self.patch_proj = nn.Conv2d(
            in_channels=self.C,
            out_channels=self.input_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Define positional embeddings and cls token embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, self.input_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.input_dim))

        # Define `depth` transformer blocks
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    n_heads=self.n_heads,
                    input_dim=self.input_dim,
                    d_k=self.d_k,
                    d_v=self.d_v,
                    mlp_dim=self.mlp_dim,
                )
                for _ in range(self.depth)
            ]
        )

        # Define last projection layer
        self.head = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ViT module/

        Parameters:
            x (torch.Tensor): Batch of images of shape (B, C, H, W)

        Returns:
            torch.Tensor: Model logits of shape (B, num_classes)
        """
        B, *_ = x.shape

        img_patches = self.patch_proj(x)  # (B, C, H, W) -> (B, input_dim, H / P, W / P)

        # (B, input_dim, H / P, W / P) -> (B, F = HW/P^2, input_dim)
        img_patches = img_patches.reshape(B, self.input_dim, -1).transpose(1, 2)

        _, F, _ = img_patches.shape

        # Repeat cls token for each batch: (B, 1, input_dim)
        cls_tokens = self.cls_token.repeat(B, 1, 1)
        img_patches = torch.cat(
            [cls_tokens, img_patches], dim=1
        )  # (B, 1 + F, input_dim)
        img_patches += self.pos_embedding[:, : (F + 1)]

        # Run transformer blocks: (B, 1 + F, input_dim)
        for layer in self.transformer:
            img_patches = layer(img_patches)

        # Extract cls token and pass through linear model head
        cls_token_embedding = img_patches[:, 0]

        logits = self.head(cls_token_embedding)  # (B, input_dim) -> (B, num_classes)

        return logits


class TransformerBlock(nn.Module):
    """
    Module for transformer block
    """

    def __init__(
        self,
        n_heads: int,
        input_dim: int,
        d_k: int,
        d_v: int,
        mlp_dim: int,
    ):
        """
        Creates an instance of the `TransformerBlock` class
        """
        super().__init__()

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.d_k = d_k
        self.d_v = d_v
        self.mlp_dim = mlp_dim

        self.mha = MultiHeadAttention(
            n_heads=self.n_heads, input_dim=self.input_dim, d_k=self.d_k, d_v=self.d_v
        )

        self.ln1 = nn.LayerNorm(self.input_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.input_dim),
        )

        self.ln2 = nn.LayerNorm(self.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transformer block
        """

        x = self.ln1(x + self.mha(x))
        x = self.ln2(x + self.mlp(x))

        return x


class AttentionHead(nn.Module):
    """
    Attention module for scaled dot product attention
    """

    def __init__(self, input_dim: int, d_k: int, d_v: int):
        """
        Creates an instance of the `ScaledDotProductAttention` class
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_k = d_k
        self.d_v = d_v

        # Construct project layers from input dimensionality
        self.Q_proj = nn.Linear(self.input_dim, self.d_k)
        self.K_proj = nn.Linear(self.input_dim, self.d_k)
        self.V_proj = nn.Linear(self.input_dim, self.d_v)

        # Softmax layer over the keys of the attention matrix
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass for attention mechanism

        Parameters:
            x (torch.Tensor): Input sequence of shape (B, F, input_dim)

        Returns:
            torch.Tensor: Output sequence of shape (B, F, output_dim)
        """
        Q = self.Q_proj(x)  # (B, F, input_dim) -> (B, F, d_k)
        K = self.K_proj(x)  # (B, F, input_dim) -> (B, F, d_k)
        V = self.V_proj(x)  # (B, F, input_dim) -> (B, F, d_v)

        scale = 1 / (self.d_k**0.5)  # Scaling factor for stability
        dot_product = Q @ K.transpose(1, 2)  # (B, F, d_k) x (B, d_k, F) -> (B, F, F)
        normalized_scores = self.softmax(dot_product * scale)  # Remains (B, F, F)

        return normalized_scores @ V  # (B, F, F) x (B, F, d_v) -> (B, F, d_v)


class MultiHeadAttention(nn.Module):
    """
    Multi-headed attention module
    """

    def __init__(self, n_heads: int, input_dim: int, d_k: int, d_v: int):
        """
        Creates an instance of the `MultiHeadAttention` class
        """
        super().__init__()

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.d_k = d_k
        self.d_v = d_v

        # Construct all attention heads as list
        self.attention_heads = nn.ModuleList(
            [
                AttentionHead(input_dim=self.input_dim, d_k=self.d_k, d_v=self.d_v)
                for _ in range(self.n_heads)
            ]
        )

        # Final projection layer
        self.linear_proj = nn.Linear(self.n_heads * self.d_v, self.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass for multi-headed attention block

        Parameters:
            x (torch.Tensor): Input sequence of shape (B, F, input_dim)

        Returns:
            torch.Tensor: Output sequence of shape (B, F, input_dim)
        """
        concat_outputs = torch.cat([head(x) for head in self.attention_heads], dim=2)

        mixed_outputs = self.linear_proj(
            concat_outputs
        )  # (B, F, n_heads * d_v) -> (B, F, input_dim)

        return mixed_outputs


if __name__ == "__main__":
    input_dim, n_heads, d_k, d_v, mlp_dim = 32, 8, 64, 96, 48
    patch_size, img_dims, depth = 8, (3, 32, 32), 4
    num_classes = 10
    B, F = 2, 5

    head = AttentionHead(input_dim=input_dim, d_k=d_k, d_v=d_v)
    mha = MultiHeadAttention(n_heads=n_heads, input_dim=input_dim, d_k=d_k, d_v=d_v)
    block = TransformerBlock(
        n_heads=n_heads, input_dim=input_dim, d_k=d_k, d_v=d_v, mlp_dim=mlp_dim
    )

    # Construct final vision transformer model
    vit = ViT(
        patch_size=patch_size,
        img_dims=img_dims,
        depth=depth,
        n_heads=n_heads,
        input_dim=input_dim,
        d_k=d_k,
        d_v=d_v,
        mlp_dim=mlp_dim,
        num_classes=num_classes,
    )

    x = torch.ones(B, F, input_dim)
    img = torch.randn(B, *img_dims)

    # Check forward pass
    head_out = head(x)
    mha_out = mha(x)
    block_out = block(x)
    vit_out = vit(img)

    print(f"{head_out.shape = }")
    print(f"{mha_out.shape = }")
    print(f"{block_out.shape = }")
    print(f"{vit_out.shape = }")


class UnfoldLinearPatchify(nn.Module):
    """
    Unfold + Linear version of image patching
    """

    def __init__(self, img_channels: int, patch_size: int, feature_dim: int):
        """
        Creates an instance of the `UnfoldLinearPatchify` class
        """
        super().__init__()

        self.img_channels = img_channels
        self.patch_size = patch_size
        self.feature_dim = feature_dim

        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.proj = nn.Linear(self.img_channels * self.patch_size**2, self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for image patching
        """

        unfolded_patches = self.unfold(x)  # (B, C, H, W) -> (B, C * P^2, HW / P ^2)
        patches_reshaped = unfolded_patches.transpose(1, 2)  # (B, HW / P ^2, C * P^2)
        patch_embeddings = self.proj(patches_reshaped)

        return patch_embeddings


class ConvPatchify(nn.Module):
    """
    Conv-based version of image patching
    """

    def __init__(self, img_channels: int, patch_size: int, feature_dim: int):
        """
        Creates an instance of the `ConvPatchify` class
        """
        super().__init__()

        self.img_channels = img_channels
        self.patch_size = patch_size
        self.feature_dim = feature_dim

        self.conv = nn.Conv2d(
            in_channels=self.img_channels,
            out_channels=self.feature_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the `ConvPatchify` layer
        """
        B, *_ = x.shape

        conv_out = self.conv(x)  # (B, C_in, H, W) -> (B, feature_dim, H/P, W/P)
        patch_embeddings = conv_out.reshape(B, self.feature_dim, -1).transpose(1, 2)

        return patch_embeddings
