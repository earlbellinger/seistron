import flax.linen as nn
import jax.numpy as jnp

# Typing
from jax import Array
from jax.typing import ArrayLike
Module = nn.Module


class FiLMGenerator(Module):
    """
    Feature wise Linear Modulation
    Reference: https://arxiv.org/abs/1709.07871
    """
    model_dim: int

    @nn.compact
    def __call__(self, x: ArrayLike) -> Array:
        gamma = nn.Dense(self.model_dim)(x)
        beta = nn.Dense(self.model_dim)(x)
        return gamma, beta


class TransformerBlock(Module):
    model_dim: int
    num_heads: int
    feed_forward_dim: int
    activation_fn: Module

    @nn.compact
    def __call__(self, x: ArrayLike) -> Array:
        x_norm = nn.LayerNorm()(x)
        attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads,
                                               qkv_features=self.model_dim)(x_norm)
        x = x + attn
        x = x + nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.feed_forward_dim),
            self.activation_fn,
            nn.Dense(self.model_dim)
        ])(x)

        return x


class PhaseAwareTransformerBlock(Module):
    model_dim: int
    num_heads: int
    ff_dim: int

    @nn.compact
    def __call__(self, x: ArrayLike, phase_embed: ArrayLike) -> Array:
        """Concatenates the phase and feature embeddings before attention block"""
        x = jnp.concatenate([x, phase_embed], axis=-1)
        x = nn.Dense(self.model_dim)(x)
        attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.model_dim)(x)
        x = x + attn
        x = x + nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.ff_dim),
            nn.gelu,
            nn.Dense(self.model_dim)
        ])(x)

        return x


class CrossAttentionPhaseAwareTransformerBlock(Module):
    model_dim: int
    num_heads: int
    feed_forward_dim: int
    activation_fn: Module

    @nn.compact
    def __call__(self, x: ArrayLike, phase_embed: ArrayLike) -> Array:
        """No concatenation, uses cross attention between phase and feature embeddings."""
        x = nn.Dense(self.model_dim)(x)
        # cross attention between the phase and features
        attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.model_dim)(inputs_q=x, inputs_kv=phase_embed)
        x = x + attn
        x = x + nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.feed_forward_dim),
            self.activation_fn,
            nn.Dense(self.model_dim)])(x)

        return x
