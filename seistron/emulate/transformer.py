from ._layers import TransformerBlock, FiLMGenerator, PhaseAwareTransformerBlock
import jax.numpy as jnp
import flax.linen as nn
# Type hints
from jax.typing import ArrayLike
from jax import Array
Module = nn.Module


class Transformer(Module):
    num_layers: int
    model_dim: int
    num_heads: int
    ff_dim: int
    output_dim: int
    activation_fn: Module

    @nn.compact
    def __call__(self, x: ArrayLike) -> Array:
        """
        Args:
            x (batch_size, input_dim)
        Returns:
            (batch_size, output_dim)
        """

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
        """
        Args:
            x (batch_size, input_dim)
        Returns:
            (batch_size, output_dim)
        """
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


# TODO: abstract away model details
class FiLMPhaseAwareTransformer(Module):
    num_layers: int = 4
    model_dim: int = 128
    num_heads: int = 8
    ff_dim: int = 256
    sequence_length: int = 100

    def setup(self):
        self.static_encoder = nn.Sequential([nn.Dense(256),
                                             nn.gelu,
                                             nn.Dense(self.model_dim)
                                             ])
        self.phase_encoder = nn.Dense(self.model_dim)

        self.film_generators = [FiLMGenerator(self.model_dim) for _ in range(self.num_layers)]
        self.blocks = [PhaseAwareTransformerBlock(self.model_dim, self.num_heads, self.ff_dim) for _ in range(self.num_layers)]
        self.output_proj = nn.Dense(1)

    def __call__(self, static_inputs):
        """
        Args:
            x (batch_size, input_dim)
        Returns:
            (batch_size, output_dim)
        """

        batch_size = static_inputs.shape[0]

        static_embed = self.static_encoder(static_inputs)

        phases = jnp.linspace(0, 1, self.sequence_length)
        phase_embed = self.phase_encoder(phases[None, :, None])
        phase_embed = jnp.repeat(phase_embed, batch_size, axis=0)

        x = jnp.zeros((batch_size, self.sequence_length, self.model_dim))

        for i in range(self.num_layers):
            gamma, beta = self.film_generators[i](static_embed)
            x = self.blocks[i](x, phase_embed)
            x = gamma[:, None, :] * x + beta[:, None, :]  # FiLM modulation

        return self.output_proj(x).squeeze(-1)
