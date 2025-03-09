import flax.linen as nn
import jax.numpy as jnp


# TODO: add type hints
class TransformerBlock(nn.Module):
    model_dim: int
    num_heads: int
    feed_forward_dim: int
    activation_fn: nn.Module

    @nn.compact
    def __call__(self, x):
        x_norm = nn.LayerNorm()(x)

        attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.model_dim)(x_norm)
        x = x + attn

        x = x + nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.feed_forward_dim),
            self.activation_fn,
            nn.Dense(self.model_dim)
        ])(x)

        return x


class FiLMGenerator(nn.Module):
    """
    Feature wise Linear Modulation
    Reference: https://arxiv.org/abs/1709.07871
    """
    model_dim: int

    @nn.compact
    def __call__(self, x):
        gamma = nn.Dense(self.model_dim)(x)
        beta = nn.Dense(self.model_dim)(x)
        return gamma, beta


class EmbeddingTransformerBlock(nn.Module):
    model_dim: int
    num_heads: int
    feed_forward_dim: int
    activation_fn: nn.Module

    @nn.compact
    def __call__(self, x, phase_embed):
        x = jnp.concatenate([x, phase_embed], axis=-1)
        x = nn.Dense(self.model_dim)(x)

        attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.model_dim)(x)
        x = x + attn

        x = x + nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.feed_forward_dim),
            self.activation_fn,
            nn.Dense(self.model_dim)
        ])(x)

        return x
