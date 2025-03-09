"""
Base module for positional encoders.
TODO: Actually use these in the transformer definitions.
"""
from abc import ABC, abstractmethod
from flax import linen as nn
from jax import Array


class BasePosEncoder(nn.Module, ABC):
    """Positional encoder interface."""
    @abstractmethod
    def __call__(self, x: Array) -> Array:
        raise NotImplementedError


class LearnedPositionalEncoder(BaseEncoder):
    """Learned positional encoding for sequences."""
    sequence_length: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, **kwargs) -> jnp.ndarray:
        pos_enc = self.param('pos_encoding',
            nn.initializers.normal(stddev=0.02),
            (1, self.sequence_length, self.model_dim)
        )
        return inputs + pos_enc


class SinusoidalPositionalEncoder(BaseEncoder):
    """Static sinusoidal positional encoding"""
    sequence_length: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, **kwargs) -> jnp.ndarray:
        position = jnp.arange(self.sequence_length)[jnp.newaxis, :]
        div_term = jnp.exp(jnp.arange(0, self.model_dim, 2) * (-jnp.log(10000.0) / self.model_dim))
        pos_enc = jnp.zeros((1, self.sequence_length, self.model_dim))
        pos_enc = pos_enc.at[:, :, 0::2].set(jnp.sin(position * div_term))
        pos_enc = pos_enc.at[:, :, 1::2].set(jnp.cos(position * div_term))
        return inputs + pos_enc
