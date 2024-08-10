import functools
from typing import Optional, Callable

import numpy as np
import jax
from jax import lax, random, numpy as jnp

import flax
from flax import struct, traverse_util, linen as nn
from flax.core import freeze, unfreeze
from flax.training import train_state, checkpoints

import optax # Optax for common losses and optimizers.

from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils

device_mesh = mesh_utils.create_device_mesh((2, 4))
#print(device_mesh)

mesh = Mesh(devices=device_mesh, axis_names=('data', 'model'))
#print(mesh)

x_sharding = NamedSharding(mesh, PartitionSpec('data'))

@functools.partial(jax.jit, in_shardings=(None),out_shardings=None)
def test(rng):
    new_rng = jax.random.fold_in(rng,jax.process_index())
    jax.debug.print("{x}",x=jax.random.key_data(new_rng))
    jax.debug.print("{y}",y=jax.random.normal(new_rng,(6)))
    return new_rng

key = jax.random.PRNGKey(0)
test(key)