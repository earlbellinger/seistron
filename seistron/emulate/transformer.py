from ._layers import TransformerBlock
import jax.numpy as jnp
import flax.linen as nn
# Type hints
from jax.typing import ArrayLike
from jax import Array
Module = nn.Module

class Transformer(nn.Module):
    num_layers: int
    model_dim: int
    num_heads: int
    ff_dim: int
    output_dim: int
    activation_fn: Module

    @nn.compact
    def __call__(self, x: ArrayLike) -> Array:
        """Args: x (batch_size, input_dim). Returns: (batch_size, output_dim)."""
        x = nn.Dense(self.model_dim)(x)
        for _ in range(self.num_layers):
            x = TransformerBlock(self.model_dim,
                                 self.num_heads,
                                 self.ff_dim,
                                 self.activation_fn)(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class EmbeddingTransformer(nn.Module):
    num_layers: int
    model_dim: int
    num_heads: int
    ff_dim: int
    activation_fn: Module
    sequence_length: int = 100

    def setup(self):
        self.pos_encoding = self.param('pos_encoding',
                                       nn.initializers.normal(stddev=0.02),
                                       (1, self.sequence_length,
                                        self.model_dim))
        self.feature_proj = nn.Dense(self.model_dim)
        self.output_proj = nn.Dense(1)

    @nn.compact
    def __call__(self, x: ArrayLike) -> Array:
        """Args: x (batch_size, input_dim). Returns: (batch_size, output_dim)."""
        x = self.feature_proj(x)
        # Expand to sequence length and add positional encoding
        x = jnp.repeat(x[:, jnp.newaxis, :], self.sequence_length, axis=1)
        x = x + self.pos_encoding
        for _ in range(self.num_layers):
            x = TransformerBlock(self.model_dim,
                                 self.num_heads,
                                 self.ff_dim,
                                 self.activation_fn)(x)
        return self.output_proj(x).squeeze(-1)
