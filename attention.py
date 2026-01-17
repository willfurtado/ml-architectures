import torch
import torch.nn as nn

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
        Q = self.Q_proj(x) # (B, F, input_dim) -> (B, F, d_k)
        K = self.K_proj(x) # (B, F, input_dim) -> (B, F, d_k)
        V = self.V_proj(x) # (B, F, input_dim) -> (B, F, d_v)

        scale = 1 / (self.d_k ** 0.5) # Scaling factor for stability
        dot_product = Q @ K.transpose(1, 2) # (B, F, d_k) x (B, d_k, F) -> (B, F, F)
        normalized_scores = self.softmax(dot_product * scale) # Remains (B, F, F)

        return normalized_scores @ V # (B, F, F) x (B, F, d_v) -> (B, F, d_v)


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
        self.attention_heads = nn.ModuleList([
            AttentionHead(input_dim=self.input_dim, d_k=self.d_k, d_v=self.d_v)
            for _ in range(self.n_heads)
        ])

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

        mixed_outputs = self.linear_proj(concat_outputs) # (B, F, n_heads * d_v) -> (B, F, d_v)

        return mixed_outputs
        
