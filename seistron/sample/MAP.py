from jax.scipy.optimize import minimize
from jax.scipy.optimize import OptimizeResults
from jax import jit, Array
from jax.typing import ArrayLike
from typing import Callable
from functools import partial
import jax.numpy as jnp

# TODO: Include other algorithms
def estimate_maximum_a_posteriori(obs: ArrayLike, x_init: ArrayLike, model: Callable,
                                  model_kwargs: dict, optimizer_kwargs: dict,
                                  method='BFGS') -> OptimizeResults:
    """
    A wrapper for jax.scipy.optimize.minimize. Gives a MAP estimate for a forward model given observations
    obs: The observations.
    model: The forward function for computing predictions.
    model_kwargs: key word arguments for calling the model
    optimizer_kwargs: key word arguments for calling jax.scipt.optimize.minimize
    """
    # If the forward model needs arguments, pass them in
    body_fn = jit(partial(model, **model_kwargs))
    # Minimize the mean squared error between forward model output and obs
    def obj_fn(X):
        return jnp.mean(jnp.square(body_fn(X) - obs))
    res = minimize(obj_fn, x0=x_init, method=method, **optimizer_kwargs)
    return res
