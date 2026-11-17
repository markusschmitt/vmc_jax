"""PRNG keys and seeds."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

def format_key(key):
    """
    Raw PRNG key, a ``uint32`` array of shape ``(2,)``, from a seed or a key.

    Args:
        * ``key``: Integer seed, raw key (``jax.random.PRNGKey()``), typed key \
          (``jax.random.key()``), or ``None`` for a random seed from ``generate_seed()``.

    Returns:
        ``jax.random.PRNGKey(key)`` for a seed, the data of a typed key, a raw key unchanged.
    """
    if isinstance(key, jax.Array):
        if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
            return jax.random.key_data(key)
        return key
    elif key is None:
        return jax.random.PRNGKey(generate_seed())
    else:
        return jax.random.PRNGKey(int(key))
    
def generate_seed():
    """Random seed in ``[0, 2**32)``, from the entropy of the operating system."""
    seed = np.random.SeedSequence().entropy % int(2 ** 32)
    return int(multihost_utils.broadcast_one_to_all(np.uint32(seed)))