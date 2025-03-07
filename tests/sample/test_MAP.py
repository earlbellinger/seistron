from seistron.sample.MAP import estimate_maximum_a_posteriori
import jax.numpy as jnp
import pytest


def test_simple_quadratic_model():
    # Test with model f(x) = x, observation 5.0
    obs = jnp.array([5.0])
    x_init = jnp.array([0.0])
    model = lambda x: x
    model_kwargs = {}
    optimizer_kwargs = {'tol': 1e-6}

    result = estimate_maximum_a_posteriori(
        obs=obs,
        x_init=x_init,
        model=model,
        model_kwargs=model_kwargs,
        optimizer_kwargs=optimizer_kwargs
    )
    assert jnp.allclose(result, 5.0, atol=1e-5)

def test_model_with_keyword_arguments():
    # Test with model f(x, offset) = x + offset, offset=3.0, observation 5.0
    obs = jnp.array([5.0])
    x_init = jnp.array([0.0])
    model = lambda x, offset: x + offset
    model_kwargs = {'offset': 3.0}
    optimizer_kwargs = {'tol': 1e-6}

    result = estimate_maximum_a_posteriori(
        obs=obs,
        x_init=x_init,
        model=model,
        model_kwargs=model_kwargs,
        optimizer_kwargs=optimizer_kwargs
    )
    assert jnp.allclose(result, 2.0, atol=1e-5)

def test_multidimensional_input():
    # Test with model f(x) = x, observation [2.0, 3.0]
    obs = jnp.array([2.0, 3.0])
    x_init = jnp.array([0.0, 0.0])
    model = lambda x: x
    model_kwargs = {}
    optimizer_kwargs = {'tol': 1e-6}

    result = estimate_maximum_a_posteriori(
        obs=obs,
        x_init=x_init,
        model=model,
        model_kwargs=model_kwargs,
        optimizer_kwargs=optimizer_kwargs
    )
    assert jnp.allclose(result, jnp.array([2.0, 3.0]), atol=1e-5)
