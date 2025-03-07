import jax.scipy as jcp
from jax import jit
from jax.typing import Array
from typing import Callable
from functools import partial


def estimate_maximum_a_posteriori(obs: Array, model: Callable, priors: Array, *args, **kwargs) -> Array:
    """
    Estimate the MAP argument for some model given observations and bounds
    obs: The observations.
    model: The forward function for computing predictions.
    priors: Bounds to constrain the parameter space for sampling the model.
    *args, **kwargs: arguments and keyword arguments for calling the model.
    """
    body_fn = jit(partial(model, *args, **kwargs))
    # pred = body_fn(obs)
    raise NotImplementedError
