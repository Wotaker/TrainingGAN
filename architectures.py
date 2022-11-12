import jax.numpy as jnp
import jax
import optax
from flax import linen as nn
from flax.training.train_state import TrainState as RawTrainState
from flax.training.checkpoints import restore_checkpoint
from flax.core import FrozenDict, unfreeze
from chex import Array, Scalar, PRNGKey


class TrainState(RawTrainState):
    batch_stats: FrozenDict


@jax.jit
def binary_cross_entropy(logits: Array, labels: Array):

    return -jnp.mean(labels * jnp.log(logits) + (1 - labels) * jnp.log(1 - logits))


def zero_grads():
    def init_fn(_): 
        return ()
    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()
    return optax.GradientTransformation(init_fn, update_fn)


def conv_block(x: Array, features: int, training: bool) -> Array:

    x = nn.Conv(features=features, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
    x = nn.BatchNorm(use_running_average=not training, momentum=0.9)(x)
    x = nn.leaky_relu(x, negative_slope=0.2)

    return x


def trans_conv_block(x: Array, features: int) -> Array:

    x = nn.ConvTranspose(features=features, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
    x = nn.leaky_relu(x, negative_slope=0.2)

    return x


class Discriminator(nn.Module):
    
    @nn.compact
    def __call__(self, batch: Array, training: bool):
        
        batch_size = batch.shape[0]
        x = batch / 255
        x = conv_block(x, 64, training)
        x = conv_block(x, 128, training)
        x = conv_block(x, 128, training)
        x = jnp.reshape(x, (batch_size, -1))
        x = nn.Dropout(rate=0.2, deterministic=not training)(x)
        x = nn.Dense(features=1)(x)
        x = nn.sigmoid(x)
        
        return x


class Generator(nn.Module):

    @nn.compact
    def __call__(self, batch: Array):

        batch_size = batch.shape[0]
        x = batch
        x = nn.Dense(features=8192)(x)
        x = jnp.reshape(x, (batch_size, 8, 8, 128))
        x = trans_conv_block(x, 128)
        x = trans_conv_block(x, 256)
        x = trans_conv_block(x, 512)
        x = nn.Conv(features=3, kernel_size=(5, 5), strides=(1, 1), padding='SAME')(x)
        x = nn.sigmoid(x)

        return x


class SimpleModel(nn.Module):

    @nn.compact
    def __call__(self, batch: Array):

        batch_size = batch.shape[0]
        x = batch
        x = nn.Dense(features=8)(x)
        x = nn.relu(x)
        x = nn.Dense(features=3)(x)
        x = nn.relu(x)

        return x


def create_Discriminator(
    dummy_batch: Array,
    init_key: PRNGKey,
    lr: Scalar = 0.00001
) -> TrainState:

    model = Discriminator()
    variables = model.init(init_key, dummy_batch, training=False)

    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=lr),
        batch_stats=variables['batch_stats'])


@jax.jit
def discriminate(state_dys: TrainState, batch: Array) -> Scalar:
    
    logits = state_dys.apply_fn(
        {'params': state_dys.params, 'batch_stats': state_dys.batch_stats},
        batch,
        training=False,
        rngs={'dropout': jax.random.PRNGKey(42)}
    )

    return jnp.squeeze(logits)


def create_Generator(
    dummy_batch: Array,
    init_key: PRNGKey,
    lr: Scalar = 0.001,
) -> RawTrainState:

    model = Generator()
    params = model.init(init_key, dummy_batch)['params']

    return RawTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(learning_rate=lr)
    )


def create_SimpleModel(
    dummy_batch: Array,
    init_key: PRNGKey,
    lr: Scalar = 0.001,
    momentum: Scalar = 0.9
) -> RawTrainState:

    model = SimpleModel()
    params = model.init(init_key, dummy_batch)['params']
    opt = optax.multi_transform(
        {"sgd": optax.sgd(learning_rate=lr, momentum=momentum), "zero": zero_grads()},
        {"Dense_0": "sgd", "Dense_1": "zero"}
    )
    # opt = optax.sgd(learning_rate=lr, momentum=momentum)

    return RawTrainState.create(apply_fn=model.apply, params=unfreeze(params), tx=opt)

