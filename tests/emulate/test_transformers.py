import pytest
import jax
import jax.numpy as jnp
import flax.linen as nn
from seistron.emulate.transformer import Transformer, EmbeddingTransformer

@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)

def test_transformer_output_shape(rng):
    model = Transformer(
        num_layers=2,
        model_dim=64,
        num_heads=4,
        ff_dim=128,
        output_dim=10,
        activation_fn=nn.relu
    )
    x = jnp.ones((5, 32))
    variables = model.init(rng, x)
    output = model.apply(variables, x)
    assert output.shape == (5, 10)

def test_transformer_different_batch_sizes(rng):
    model = Transformer(
        num_layers=2,
        model_dim=64,
        num_heads=4,
        ff_dim=128,
        output_dim=10,
        activation_fn=nn.relu
    )
    variables = model.init(rng, jnp.ones((1, 32)))

    output = model.apply(variables, jnp.ones((1, 32)))
    assert output.shape == (1, 10)

    output = model.apply(variables, jnp.ones((10, 32)))
    assert output.shape == (10, 10)

def test_transformer_zero_layers(rng):
    model = Transformer(
        num_layers=0,
        model_dim=64,
        num_heads=4,
        ff_dim=128,
        output_dim=10,
        activation_fn=nn.relu
    )
    x = jnp.ones((5, 32))
    variables = model.init(rng, x)
    output = model.apply(variables, x)
    assert output.shape == (5, 10)

def test_embedding_transformer_output_shape(rng):
    model = EmbeddingTransformer(
        num_layers=2,
        model_dim=64,
        num_heads=4,
        ff_dim=128,
        activation_fn=nn.relu,
        sequence_length=100
    )
    x = jnp.ones((5, 32))
    variables = model.init(rng, x)
    output = model.apply(variables, x)
    assert output.shape == (5, 100)

def test_embedding_transformer_pos_encoding_shape(rng):
    model = EmbeddingTransformer(
        num_layers=2,
        model_dim=64,
        num_heads=4,
        ff_dim=128,
        activation_fn=nn.relu,
        sequence_length=100
    )
    x = jnp.ones((5, 32))
    variables = model.init(rng, x)
    pos_encoding = variables["params"]["pos_encoding"]
    assert pos_encoding.shape == (1, 100, 64)
