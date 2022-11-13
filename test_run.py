import time
import jax
import jax.numpy as jnp

from jax.random import PRNGKey as jkey
from chex import Array, Shape, PRNGKey

from architectures import *
from training_gan import *


def main():

    ds_galaxies = load_ds()
    ds_galaxies.shape

    state_dis = create_Discriminator(jkey(42))
    state_gen = create_Generator(jkey(42))

    key = jkey(42)

    key, perm_key = jax.random.split(key, 2)
    n_samples = ds_galaxies.shape[0]
    steps_per_epoch = n_samples // BATCH_SIZE

    perms = jax.random.permutation(perm_key, n_samples)[:steps_per_epoch * BATCH_SIZE]
    perms = jnp.reshape(perms, (steps_per_epoch, BATCH_SIZE))
    seed_vectors = jax.random.normal(key, shape=(steps_per_epoch, BATCH_SIZE, 128))

    def scan_fun(carry, perm):

        state_dis, state_gen, step, key = carry

        key, step_key = jax.random.split(key)

        batch_authentic = ds_galaxies[perm, ...]
        state_dis, loss_dis, _ = discriminator_train_step(step_key, state_dis, state_gen, batch_authentic)
        state_gen, loss_gen = generator_train_step(state_dis, state_gen, seed_vectors[step])

        return (state_dis, state_gen, step + 1, key), (loss_dis, loss_gen)

    scan_init = (state_dis, state_gen, 0, key)

    t_start = time.time()
    (state_dis, state_gen, _, key), losses = jax.lax.scan(scan_fun, scan_init, perms)
    elapsed_time = time.time() - t_start

    loss_dis_acc, loss_gen_acc = losses
    loss_dis_acc, loss_gen_acc = losses

    print(f"Discriminator loss:\n{loss_dis_acc}\n\n")
    print(f"Discriminator loss:\n{loss_gen_acc}\n\n")
    print(f"Total training time: {elapsed_time:.3f}")


if __name__ == "__main__":

    ds_galaxies = load_ds()
    ds_galaxies.shape

    state_dis = create_Discriminator(jkey(42))
    state_gen = create_Generator(jkey(42))

    state_dis, state_gen, metrices, elapsed_time = train(
        42,
        state_dis,
        state_gen,
        ds_galaxies,
        epochs=5,
        log_every=1,
    )

    print(f'\nTraining time: {elapsed_time:.4f}')
