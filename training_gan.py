from typing import Callable, Tuple

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


BATCH_SIZE          = 8
LR_DISCRIMINATOR    = 0.00001
LR_GENERATOR        = 0.00001

MONITOR_VECTORS     = jax.random.normal(jkey(43), shape=(6, 128))
GENERATOR_LABELS    = jnp.ones((BATCH_SIZE,))
IMG_WHITE           = jnp.ones(shape=(64, 64, 3))


class Metrices:

    def __init__(self, epochs: Array):
        
        n_epochs = epochs.shape[0]
        self.idx = 0
        self.epochs: Array = epochs
        self.loss_dis_trace: Array = jnp.zeros(n_epochs)
        self.loss_gen_trace: Array = jnp.zeros(n_epochs)
	
    def update(self, loss_dis, loss_gen):

        self.loss_dis_trace = self.loss_dis_trace.at[self.idx].set(loss_dis)
        self.loss_gen_trace = self.loss_gen_trace.at[self.idx].set(loss_gen)
        self.idx += 1


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


@jax.jit
def train_epoch(
    epoch_key: PRNGKey,
    state_dis: TrainState,
    state_gen: RawTrainState,
    dataset: Array,
) -> Tuple[PRNGKey, TrainState, RawTrainState, Scalar, Scalar]:

    key, perm_key, vectors_key = jax.random.split(epoch_key, 3)

    n_samples = dataset.shape[0]
    steps_per_epoch = n_samples // BATCH_SIZE
    perms = jax.random.permutation(perm_key, n_samples)[:steps_per_epoch * BATCH_SIZE]
    perms = jnp.reshape(perms, (steps_per_epoch, BATCH_SIZE))
    seed_vectors = jax.random.normal(vectors_key, shape=(steps_per_epoch, BATCH_SIZE, 128))

    def scan_fun(carry, perm):

        state_dis, state_gen, step, key = carry

        key, step_key = jax.random.split(key)

        batch_authentic = dataset[perm, ...]
        state_dis, loss_dis, _ = discriminator_train_step(step_key, state_dis, state_gen, batch_authentic)
        state_gen, loss_gen = generator_train_step(state_dis, state_gen, seed_vectors[step])

        return (state_dis, state_gen, step + 1, key), (loss_dis, loss_gen)

    scan_init = (state_dis, state_gen, 0, key)
    (state_dis, state_gen, _, key), (loss_dis_acc, loss_gen_acc) = jax.lax.scan(scan_fun, scan_init, perms)

    return key, state_dis, state_gen, jnp.mean(loss_dis_acc), jnp.mean(loss_gen_acc)


def train(
    seed: int,
    state_dis: TrainState,
    state_gen: RawTrainState,
    dataset: Array,
    epoch_count: int,
    epoch_start: int = 1,
    log_every: int = 0,
    checkpoint_dir: str = ""
) -> Tuple[TrainState, RawTrainState, Metrices, float]:

    key, epoch_key = jax.random.split(jkey(seed))

    # Create structures to accumulate metrices
    epochs = jnp.arange(epoch_start, epoch_start + epoch_count)
    metrices = Metrices(epochs)

    # Iterate through the dataset for epochs number of times
    t_start = time.time()
    for epoch in epochs:

        epoch_key, state_dis, state_gen, loss_dis, loss_gen = train_epoch(
            epoch_key,
            state_dis,
            state_gen,
            dataset
        )
        metrices.update(loss_dis=loss_dis, loss_gen=loss_gen)

        if log_every and (epoch % log_every == 0 or epoch in {epoch_start, epoch_start + epoch_count - 1}):
            checkpoint(checkpoint_dir, state_dis, state_gen, metrices, epoch)
            print(
                'epoch:% 3d, discriminator_loss: %.4f, generator_loss: %.4f'
                % (epoch, loss_dis, loss_gen)
            )
    
    return state_dis, state_gen, metrices, time.time() - t_start


def checkpoint(
    checkpoint_dir: str,
    state_dis: TrainState,
    state_gen: RawTrainState,
    metrices: Metrices,
    epoch: int,
) -> None:

    def save_generated():
        
        # Create directory tree if necessary
        if not ("generated" in os.listdir(checkpoint_dir)):
            os.mkdir(os.path.join(checkpoint_dir, "generated"))

        # Collect checkpoints steps to a list
        checkpoints = os.listdir(os.path.join(checkpoint_dir, "generated"))
        checkpoints = list(map(lambda cpt: int(cpt.split('_')[-1]), checkpoints))

        # Remove later checkpoints (if any)
        to_removal = list(filter(lambda cpt: cpt >= epoch, checkpoints))
        for cpt in to_removal:
            cpt_directory = os.path.join(checkpoint_dir, "generated", f"checkpoint_{cpt}")
            os.system(f'rm -rf {cpt_directory}')

        # Create directory to store generated monitors
        os.mkdir(os.path.join(checkpoint_dir, "generated", f"checkpoint_{epoch}"))

        # Generate monitor images and save in appropriate checkpoint directory
        monitors_imgs = generate(state_gen, MONITOR_VECTORS)
        for idx, img in enumerate(monitors_imgs):
            plt.imsave(
                os.path.join(checkpoint_dir, "generated", f"checkpoint_{epoch}", f"monitor_{idx}.png"),
                img,
                cmap="Greys"
            )
    
    try:
        save_checkpoint(
            checkpoint_dir,
            state_dis,
            epoch,
            prefix="checkpoint-discriminator_",
            overwrite=True,
            keep_every_n_steps=1
        )
        save_checkpoint(
            checkpoint_dir,
            state_gen,
            epoch,
            prefix="checkpoint-generator_",
            overwrite=True,
            keep_every_n_steps=1
        )
    except flax.errors.InvalidCheckpointError:
        print(f'[Warning] Could not save state after epoch {epoch}!')
        return
    
    try:
        save_generated()
    except:
        print(f'[Warning] Could not save generated monitor images after epoch {epoch}!')
    
    try:
        plot_metrices(checkpoint_dir, metrices)
    except:
        print(f'[Warning] Could not save loss plot after epoch {epoch}!')
    
    return


def plot_metrices(checkpoint_dir: str, metrices: Metrices):

    phi = (1 + jnp.sqrt(5)) / 2
    height = 5
    epochs = metrices.epochs

    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(phi * height, height)
    
    axes.plot(epochs, metrices.loss_dis_trace, label="discriminator")
    axes.plot(epochs, metrices.loss_gen_trace, label="generator")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss Value")
    axes.set_title("Loss Curves")
    axes.legend()

    # Create directory tree if necessary
    if not ("loss" in os.listdir(checkpoint_dir)):
        os.mkdir(os.path.join(checkpoint_dir, "loss"))

    plt.savefig(os.path.join(checkpoint_dir, "loss", f"loss.png"))
    plt.close()


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


if __name__ == "__main__":

    ds_galaxies = load_ds()
    ds_galaxies.shape

    state_dis = create_Discriminator(jkey(42))
    state_gen = create_Generator(jkey(42))

    state_dis, state_gen, metrices, elapsed_time = train(
        seed=42,
        state_dis=state_dis,
        state_gen=state_gen,
        dataset=ds_galaxies,
        epoch_count=25,
        epoch_start=5,
        log_every=1,
        checkpoint_dir="/home/students/wciezobka/agh/TrainingGAN/checkpoints/test_run"
    )

    print(f'\nTraining time: {elapsed_time:.4f}')
