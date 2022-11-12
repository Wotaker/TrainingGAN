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


BATCH_SIZE = 8
GENERATOR_LABELS = jnp.ones((BATCH_SIZE,))
IMG_WHITE = jnp.ones(shape=(64, 64, 3))


def load_ds(ds_path: str = "datasets/galaxies", plot: bool = False):

    files = os.listdir(ds_path)
    load_galaxy = lambda file: jnp.array(Image.open(os.path.join(ds_path, file)).convert('RGB'))
    images = jnp.array(list(map(load_galaxy, files))) / 255

    if plot:
        plot_samples(images, subplots_shape=(5, 5))
    
    return images


@jax.jit
def compute_dis_grads(state: TrainState, batch: Array, labels: Array):
    """Computes discriminators gradients, loss and accuracy for a single batch."""

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
def discriminator_train_step(
    key: PRNGKey,
    state_dis: TrainState,
    state_gen: RawTrainState,
    batch_authentic: Array
) -> Tuple[TrainState, Scalar, Array]:

    batch_key, uniform_key_1, uniform_key_2 = jax.random.split(key, 3)
    batch_seed_vector = jax.random.normal(batch_key, shape=(BATCH_SIZE, 128))

    batch_syntetic = state_gen.apply_fn(
        {'params': state_gen.params},
        batch_seed_vector
    )
    batch_merged = jnp.concatenate((batch_authentic, batch_syntetic))
    labels = jnp.concatenate((jnp.ones(BATCH_SIZE), jnp.zeros(BATCH_SIZE)))
    noise = jnp.concatenate((
        jax.random.uniform(uniform_key_1, (BATCH_SIZE,), minval=-0.05, maxval=0.0),
        jax.random.uniform(uniform_key_2, (BATCH_SIZE,), minval=0.0, maxval=0.05)
    ))
    labels += noise

    grads, state_dis, loss, logits = compute_dis_grads(state_dis, batch_merged, labels)
    state_dis = state_dis.apply_gradients(grads=grads)

    return state_dis, loss, logits


@jax.jit
def compute_gen_grads(state_dis: TrainState, state_gen: RawTrainState, seed_vector: Array):

    def loss_fn(gen_params: FrozenDict):

        batch_syntetic = state_gen.apply_fn(
            {'params': gen_params},
            seed_vector
        )
        logits = discriminate(state_dis, batch_syntetic)
        loss = binary_cross_entropy(logits, GENERATOR_LABELS)
        # loss = jnp.mean(IMG_WHITE - batch_syntetic)

        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state_gen.params)

    new_state_gen = RawTrainState(
        step=state_gen.step,
        apply_fn=state_gen.apply_fn,
        params=state_gen.params,
        tx=state_gen.tx,
        opt_state=state_gen.opt_state,
    )

    return grads, new_state_gen, loss


@jax.jit
def generator_train_step(state_dis: TrainState, state_gen: RawTrainState, seed_vector: Array):

    grads, state_gen, loss = compute_gen_grads(state_dis, state_gen, seed_vector)
    state_gen = state_gen.apply_gradients(grads=grads)

    return state_gen, loss


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
