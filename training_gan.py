from typing import Callable, Tuple
from absl import logging

import os
import time
import datetime
from PIL import Image
import numpy as np
import jax
import jax.numpy as jnp
from jax.random import PRNGKey as jkey
from chex import Scalar, Array, PRNGKey, Shape
import flax
from flax import linen as nn
from flax.training.train_state import TrainState as RawTrainState
from flax.training.checkpoints import save_checkpoint
import optax
import matplotlib.pyplot as plt

from architectures import *


def load_ds(ds_path: str = "datasets/galaxies", plot: bool = False):

    files = os.listdir(ds_path)
    load_galaxy = lambda file: jnp.array(Image.open(os.path.join(ds_path, file)).convert('RGB'))
    images = jnp.array(list(map(load_galaxy, files))) / 255

    if plot:
        plot_samples(images, subplots_shape=(5, 5))
    
    return images


@jax.jit
def binary_cross_entropy(logits: Array, labels: Array):

    return -jnp.mean(labels * jnp.log(logits) + (1 - labels) * jnp.log(1 - logits))


@jax.jit
def apply_dys_model(state: TrainState, batch: Array, labels: Array):
    """Computes dyscriminators gradients, loss and accuracy for a single batch."""

    def loss_fn(params, batch_stats):

        logits, mutated_vars = state.apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            batch,
            training=True,
            mutable=['batch_stats'],
            rngs={'dropout': jax.random.PRNGKey(42)}
        )
        logits = jnp.squeeze(logits)
        loss = binary_cross_entropy(logits=logits, labels=labels)

        return loss, (mutated_vars, logits)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (mutated_vars, logits)), grads = grad_fn(state.params, state.batch_stats)

    new_state = TrainState(
        step=state.step,
        apply_fn=state.apply_fn,
        params=state.params,
        tx=state.tx,
        opt_state=state.opt_state,
        batch_stats=mutated_vars['batch_stats']
    )
    
    return grads, new_state, loss, logits


@jax.jit
def update_dyscriminator(state: TrainState, grads: nn.FrozenDict):

    return state.apply_gradients(grads=grads)


def dyscriminator_train_step(
    key: PRNGKey,
    state_dys: TrainState,
    state_gen: RawTrainState,
    ds: Array,
    batch_size: int,
    perm: Array
) -> Tuple[TrainState, Scalar, Array]:

    key, batch_key, uniform_key_1, uniform_key_2 = jax.random.split(key, 4)
    batch_seed_vector = jax.random.normal(batch_key, shape=(batch_size, 128))

    batch_authentic = ds[perm, ...]
    batch_syntetic = state_gen.apply_fn(
        {'params': state_gen.params["params"]},
        batch_seed_vector
    )
    batch_merged = jnp.concatenate((batch_authentic, batch_syntetic))
    labels = jnp.concatenate((jnp.ones(batch_size), jnp.zeros(batch_size)))
    noise = jnp.concatenate((
        jax.random.uniform(uniform_key_1, (batch_size,), minval=-0.05, maxval=0.0),
        jax.random.uniform(uniform_key_2, (batch_size,), minval=0.0, maxval=0.05)
    ))
    labels += noise

    grads, state_dys, loss, logits = apply_dys_model(state_dys, batch_merged, labels)
    state_dys = update_dyscriminator(state_dys, grads)

    return state_dys, loss, logits


def plot_samples(batch: Array, subplots_shape: Shape = (3, 5), seed: int = 42):

    rows = subplots_shape[0]
    cols = subplots_shape[1]
    num = rows * cols

    indeces = jax.random.choice(jax.random.PRNGKey(seed), batch.shape[0], (num,), replace=False)

    images = batch[indeces]

    fig, axes = plt.subplots(rows, cols)
    fig.set_size_inches(rows * 2, cols * 2)
    for i in range(num):
        ax = axes[i // cols, i % cols]
        ax.imshow(images[i])
        ax.axis('off')
    fig.tight_layout()
    plt.show()
