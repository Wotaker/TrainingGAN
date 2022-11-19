from typing import Tuple

import os
import time
from PIL import Image
import jax
import jax.numpy as jnp
from jax.random import PRNGKey as jkey
from chex import Scalar, Array, PRNGKey, Shape
import flax.errors
from flax.training.train_state import TrainState as RawTrainState
from flax.training.checkpoints import save_checkpoint, latest_checkpoint, restore_checkpoint
import matplotlib.pyplot as plt

from architectures import *


# ======= Hyperparameters =======
BATCH_SIZE          = 8
EPOCHS              = 1500
LR_DISCRIMINATOR    = 0.00001
LR_GENERATOR        = 0.00001
B1_DISCRIMINATOR    = 0.9
B1_GENERATOR        = 0.9
B2_DISCRIMINATOR    = 0.999
B2_GENERATOR        = 0.999
RESET_DIS           = False     # Only considered when loading a saved state
RESET_GEN           = False     # Only considered when loading a saved state

# ======= Variables =============
SEED = 22236
EPOCH_START = 0
CKPT_EVERY = 100
LOAD_CKPT_DIR = "/home/students/wciezobka/agh/TrainingGAN/checkpoints/amanita/seed_22236/phase_1/retrived"
SAVE_CKPT_DIR = "/home/students/wciezobka/agh/TrainingGAN/checkpoints/amanita/seed_22236/phase_2"
DATASET_PATH = "agh/TrainingGAN/datasets/anime_small"

# ======= Constants ===================================================
SQRT_MONITOR        = 5
MONITOR_VECTORS     = jax.random.normal(jkey(666), shape=(SQRT_MONITOR * SQRT_MONITOR, 128))
GENERATOR_LABELS    = jnp.ones((BATCH_SIZE,))
IMG_WHITE           = jnp.ones(shape=(64, 64, 3))


class Metrices:

    def __init__(self, epochs: Array):
        
        n_epochs = epochs.shape[0]
        self.epochs: Array = epochs

        self.loss_dis_trace: Array = jnp.zeros(n_epochs)
        self.loss_gen_trace: Array = jnp.zeros(n_epochs)

        self.acc_real_trace: Array = jnp.zeros(n_epochs)
        self.acc_fake_trace: Array = jnp.zeros(n_epochs)
        self.acc_gen_trace: Array = jnp.zeros(n_epochs)

        self.idx = 0
	
    def update(self, loss_dis, loss_gen, acc_real, acc_fake, acc_gen):

        self.loss_dis_trace = self.loss_dis_trace.at[self.idx].set(loss_dis)
        self.loss_gen_trace = self.loss_gen_trace.at[self.idx].set(loss_gen)

        self.acc_real_trace = self.acc_real_trace.at[self.idx].set(acc_real)
        self.acc_fake_trace = self.acc_fake_trace.at[self.idx].set(acc_fake)
        self.acc_gen_trace = self.acc_gen_trace.at[self.idx].set(acc_gen)

        self.idx += 1


def main():

    dataset = load_ds(
        jkey(SEED),
        ds_path=DATASET_PATH
    )

    epoch_start, state_dis, state_gen = initialize_GAN(
        epoch_start=EPOCH_START,
        checkpoint_dir=LOAD_CKPT_DIR,
        reset_dis=RESET_DIS,
        reset_gen=RESET_GEN
    )

    state_dis, state_gen, _, elapsed_time = train(
        seed=SEED,
        state_dis=state_dis,
        state_gen=state_gen,
        dataset=dataset,
        epoch_count=EPOCHS,
        epoch_start=epoch_start,
        checkpoint_every=CKPT_EVERY,
        checkpoint_dir=SAVE_CKPT_DIR
    )

    print(f'\nTraining time: {elapsed_time:.4f}')


def load_ds(key: PRNGKey, ds_path: str, ds_size: int = -1, plot: bool = False):

    key, plot_key = jax.random.split(key)

    files = os.listdir(ds_path)
    ds_size = ds_size if ds_size > 0 else len(files)
    indeces = jax.random.choice(key, jnp.arange(len(files)), (ds_size,), replace=False)
    load_images = lambda file_id: jnp.array(Image.open(os.path.join(ds_path, files[file_id])).resize((64, 64)).convert('RGB')) / 255
    images = jnp.array(list(map(load_images, indeces)))

    if plot:
        plot_samples(plot_key, images, subplots_shape=(5, 5))
    
    return images


def load_state(
    checkpoint_dir: str,
    epoch: int = None
) -> Tuple[int, TrainState, TrainState]:

    dummy_state_dis = create_Discriminator(
        seed=SEED,
        lr=LR_DISCRIMINATOR,
        b1=B1_DISCRIMINATOR,
        b2=B2_DISCRIMINATOR
    )
    dummy_state_gen = create_GeneratorV2(
        seed=SEED,
        lr=LR_GENERATOR,
        b1=B1_GENERATOR,
        b2=B2_GENERATOR
    )

    if not epoch:
        latest_path = latest_checkpoint(checkpoint_dir, "checkpoint-discriminator_")
        epoch = latest_path.split("_")[-1]

    # Load trainning states from checkpoints
    state_restored_dis: TrainState = restore_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=dummy_state_dis,
        step=epoch,
        prefix="checkpoint-discriminator_"
    )

    state_restored_gen: TrainState = restore_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=dummy_state_gen,
        step=epoch,
        prefix="checkpoint-generator_"
    )

    # Update optimizers
    state_dis = TrainState(
        step=state_restored_dis.step,
        apply_fn=state_restored_dis.apply_fn,
        params=state_restored_dis.params,
        tx=optax.adam(
            learning_rate=LR_DISCRIMINATOR,
            b1=B1_DISCRIMINATOR,
            b2=B2_DISCRIMINATOR
        ),
        opt_state=state_restored_dis.tx.init(state_restored_dis.params),
        batch_stats=state_restored_dis.batch_stats
    )

    state_gen = TrainState(
        step=state_restored_gen.step,
        apply_fn=state_restored_gen.apply_fn,
        params=state_restored_gen.params,
        tx=optax.adam(
            learning_rate=LR_GENERATOR,
            b1=B1_GENERATOR,
            b2=B2_GENERATOR
        ),
        opt_state=state_restored_gen.tx.init(state_restored_gen.params),
        batch_stats=state_restored_gen.batch_stats
    )

    print(f"[Info] loaded state from epoch {epoch} checkpoint")

    return epoch + 1, state_dis, state_gen


def initialize_GAN(
    epoch_start: int = None,
    checkpoint_dir: str = "",
    reset_dis: bool = False,
    reset_gen: bool = False
) -> Tuple[int, TrainState, TrainState]:

    assert not epoch_start or (epoch_start and checkpoint_dir), \
        f"[Error] You need to provide path to checkpoint dir as well if starting training from epoch {epoch_start}!"
    
    # Load states
    if epoch_start:
        epoch, state_dis, state_gen = load_state(checkpoint_dir, epoch_start)
    else:
        epoch = 1
        state_dis = create_Discriminator(
            seed=SEED,
            lr=LR_DISCRIMINATOR,
            b1=B1_DISCRIMINATOR,
            b2=B2_DISCRIMINATOR
        )
        state_gen = create_GeneratorV2(
            seed=SEED,
            lr=LR_GENERATOR,
            b1=B1_GENERATOR,
            b2=B2_GENERATOR
        )

        return epoch, state_dis, state_gen
    
    # Reset states if necessary
    if reset_dis:
        state_dis = create_Discriminator(
            seed=SEED,
            lr=LR_DISCRIMINATOR,
            b1=B1_DISCRIMINATOR,
            b2=B2_DISCRIMINATOR
        )
    if reset_gen:
        state_gen = create_GeneratorV2(
            seed=SEED,
            lr=LR_GENERATOR,
            b1=B1_GENERATOR,
            b2=B2_GENERATOR
        )

    return epoch, state_dis, state_gen


@jax.jit
def compute_dis_grads(state_dis: TrainState, batch: Array, labels: Array):
    """Computes discriminators gradients and loss of a single batch"""

    def loss_fn(params, batch_stats):

        logits, mutated_vars = state_dis.apply_fn(
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
    (loss, (mutated_vars, logits)), grads = grad_fn(state_dis.params, state_dis.batch_stats)

    new_state = TrainState(
        step=state_dis.step,
        apply_fn=state_dis.apply_fn,
        params=state_dis.params,
        tx=state_dis.tx,
        opt_state=state_dis.opt_state,
        batch_stats=mutated_vars['batch_stats']
    )
    
    return grads, new_state, loss, logits


@jax.jit
def discriminator_train_step(
    key: PRNGKey,
    state_dis: TrainState,
    state_gen: TrainState,
    batch_authentic: Array
) -> Tuple[TrainState, Scalar, Scalar, Scalar]:

    batch_key, uniform_key_1, uniform_key_2 = jax.random.split(key, 3)
    batch_seed_vector = jax.random.normal(batch_key, shape=(BATCH_SIZE, 128))

    batch_syntetic = generateV2(state_gen, batch_seed_vector)
    batch_merged = jnp.concatenate((batch_authentic, batch_syntetic))
    labels = jnp.concatenate((jnp.ones(BATCH_SIZE), jnp.zeros(BATCH_SIZE)))
    noise = jnp.concatenate((
        jax.random.uniform(uniform_key_1, (BATCH_SIZE,), minval=-0.05, maxval=0.0),
        jax.random.uniform(uniform_key_2, (BATCH_SIZE,), minval=0.0, maxval=0.05)
    ))
    labels += noise

    grads, state_dis, loss, logits = compute_dis_grads(state_dis, batch_merged, labels)
    state_dis = state_dis.apply_gradients(grads=grads)

    # Compute discriminator accuracy (acc_real, acc_fake)
    acc_real = jnp.sum(logits[:BATCH_SIZE]) / BATCH_SIZE
    acc_fake = jnp.sum(1 - logits[BATCH_SIZE:]) / BATCH_SIZE

    return state_dis, loss, acc_real, acc_fake


@jax.jit
def compute_gen_grads(state_dis: TrainState, state_gen: TrainState, seed_vector: Array):

    def loss_fn(params, batch_stats):

        batch_syntetic, mutated_vars = state_gen.apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            seed_vector,
            training=True,
            mutable=['batch_stats']
        )
        logits = discriminate(state_dis, batch_syntetic)
        loss = binary_cross_entropy(logits, GENERATOR_LABELS)

        return loss, (mutated_vars, logits)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (mutated_vars, logits)), grads = grad_fn(state_gen.params, state_gen.batch_stats)

    new_state_gen = TrainState(
        step=state_gen.step,
        apply_fn=state_gen.apply_fn,
        params=state_gen.params,
        tx=state_gen.tx,
        opt_state=state_gen.opt_state,
        batch_stats=mutated_vars['batch_stats']
    )

    return grads, new_state_gen, loss, logits


@jax.jit
def generator_train_step(state_dis: TrainState, state_gen: TrainState, seed_vector: Array):

    grads, state_gen, loss, logits = compute_gen_grads(state_dis, state_gen, seed_vector)
    state_gen = state_gen.apply_gradients(grads=grads)

    # Compute generator accuracy (acc_gen)
    acc_gen = jnp.sum(logits) / BATCH_SIZE

    return state_gen, loss, acc_gen


@jax.jit
def train_epoch(
    epoch_key: PRNGKey,
    state_dis: TrainState,
    state_gen: TrainState,
    dataset: Array,
) -> Tuple[PRNGKey, TrainState, TrainState, Scalar, Scalar, Scalar, Scalar, Scalar]:

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
        state_dis, loss_dis, acc_real, acc_fake = \
            discriminator_train_step(step_key, state_dis, state_gen, batch_authentic)
        state_gen, loss_gen, acc_gen = \
            generator_train_step(state_dis, state_gen, seed_vectors[step])

        return (state_dis, state_gen, step + 1, key), (loss_dis, loss_gen, acc_real, acc_fake, acc_gen)

    scan_init = (state_dis, state_gen, 0, key)
    (state_dis, state_gen, _, key), (loss_dis_acc, loss_gen_acc, acc_real_acc, acc_fake_acc, acc_gen_acc) = \
        jax.lax.scan(scan_fun, scan_init, perms)
    
    # Calculate metrices
    loss_dis = jnp.mean(loss_dis_acc)
    loss_gen = jnp.mean(loss_gen_acc)
    acc_real = jnp.mean(acc_real_acc) / steps_per_epoch
    acc_fake = jnp.mean(acc_fake_acc) / steps_per_epoch
    acc_gen = jnp.mean(acc_gen_acc) / steps_per_epoch

    return key, state_dis, state_gen, loss_dis, loss_gen, acc_real, acc_fake, acc_gen


def train(
    seed: int,
    state_dis: TrainState,
    state_gen: TrainState,
    dataset: Array,
    epoch_count: int,
    epoch_start: int = 1,
    checkpoint_every: int = 0,
    checkpoint_dir: str = ""
) -> Tuple[TrainState, TrainState, Metrices, float]:

    key, epoch_key = jax.random.split(jkey(seed))

    # Create structures to accumulate metrices
    epochs = jnp.arange(epoch_start, epoch_start + epoch_count)
    metrices = Metrices(epochs)

    # Iterate through the dataset for epochs number of times
    print(f'[Info] Training started at {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}')
    t_start = time.time()
    for epoch in epochs:

        epoch = int(epoch)

        # Train one epoch
        epoch_key, state_dis, state_gen, loss_dis, loss_gen, acc_real, acc_fake, acc_gen = train_epoch(
            epoch_key,
            state_dis,
            state_gen,
            dataset
        )
        metrices.update(
            loss_dis=loss_dis,
            loss_gen=loss_gen,
            acc_real=acc_real,
            acc_fake=acc_fake,
            acc_gen=acc_gen
        )

        # Update loss plot
        print(
            '[Info] epoch:% 3d, discriminator_loss: %.4f, generator_loss: %.4f, acc_real: %.4f, acc_fake: %.4f, acc_gen: %.4f' \
                % (epoch, loss_dis, loss_gen, acc_real, acc_fake, acc_gen)
        )
        try:
            plot_metrices(checkpoint_dir, metrices)
        except:
            print(f'[Warning] Could not save loss plot after epoch {epoch}!')

        # Checkpoint every "log_every"
        if checkpoint_every and (epoch % checkpoint_every == 0 or epoch in {int(epochs[0]), int(epochs[-1])}):
            checkpoint(checkpoint_dir, state_dis, state_gen, epoch)
    
    return state_dis, state_gen, metrices, time.time() - t_start


def checkpoint(
    checkpoint_dir: str,
    state_dis: TrainState,
    state_gen: TrainState,
    epoch: int,
) -> None:

    # TODO verify saving directories
    def save_generated():
        
        # Create directory tree if necessary
        if not ("generated" in os.listdir(checkpoint_dir)):
            os.mkdir(os.path.join(checkpoint_dir, "generated"))

        # Collect checkpoints steps to a list
        checkpoints = os.listdir(os.path.join(checkpoint_dir, "generated"))
        checkpoints = list(map(lambda cpt: int(cpt.split('_')[-1].split('.')[0]), checkpoints))

        # Remove later checkpoints (if any)
        to_removal = list(filter(lambda cpt: cpt >= epoch, checkpoints))
        for cpt in to_removal:
            cpt_directory = os.path.join(checkpoint_dir, "generated", f"checkpoint_{cpt}.png")
            os.system(f'rm -rf {cpt_directory}')

        # Generate monitor images and save in appropriate checkpoint directory
        n = SQRT_MONITOR
        monitors_imgs = generateV2(state_gen, MONITOR_VECTORS * 25)
        fig, axes = plt.subplots(n, n)
        fig.set_size_inches(n * 2, n * 2)
        for i in range(n * n):
            ax = axes[i // n, i % n]
            ax.imshow(monitors_imgs[i])
            ax.axis('off')
        fig.tight_layout()
        plt.savefig(os.path.join(checkpoint_dir, "generated", f"checkpoint_{epoch}.png"))
        plt.clf()
    
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
    
    return


def plot_metrices(checkpoint_dir: str, metrices: Metrices):

    phi = (1 + jnp.sqrt(5)) / 2
    height = 3
    epochs = metrices.epochs

    fig, axes = plt.subplots(2, 1)
    fig.set_size_inches(phi * height, 2 * height)
    
    # Plot loss
    axes[0].plot(epochs, metrices.loss_dis_trace, label="discriminator")
    axes[0].plot(epochs, metrices.loss_gen_trace, label="generator")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss Value")
    axes[0].legend()

    # Plot accuracy
    axes[1].plot(epochs, 100 * metrices.acc_real_trace, label="disc. real")
    axes[1].plot(epochs, 100 * metrices.acc_fake_trace, label="disc. fake")
    axes[1].plot(epochs, 100 * metrices.acc_gen_trace, label="generator")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy [%]")
    axes[1].legend()

    # Create directory tree if necessary
    if not ("metrices" in os.listdir(checkpoint_dir)):
        os.mkdir(os.path.join(checkpoint_dir, "metrices"))

    plt.savefig(os.path.join(checkpoint_dir, "metrices", f"metrices.png"))
    plt.close()


def plot_samples(key: PRNGKey, batch: Array, subplots_shape: Shape = (5, 5)):

    rows = subplots_shape[0]
    cols = subplots_shape[1]
    num = rows * cols

    indeces = jax.random.choice(key, batch.shape[0], (num,), replace=False)

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
    main()
