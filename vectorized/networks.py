import flax.linen as nn
import jax.numpy as jnp


class JaxActor(nn.Module):
    act_size: int

    @nn.compact
    def __call__(self, o):
        x = nn.relu(nn.Dense(256)(o))
        x = nn.relu(nn.Dense(256)(x))
        x = nn.relu(nn.Dense(512)(x))
        x = nn.relu(nn.Dense(512)(x))
        x = nn.relu(nn.Dense(256)(x))
        x = nn.tanh(nn.Dense(self.act_size)(x))

        return x


class JaxCritic(nn.Module):
    @nn.compact
    def __call__(self, o, a):
        x = nn.relu(nn.Dense(512)(o))
        x = nn.relu(nn.Dense(512)(x))
        x = nn.relu(nn.Dense(1024)(x))
        x = jnp.concatenate([x, a], axis=-1)
        x = nn.relu(nn.Dense(512)(x))
        x = nn.relu(nn.Dense(512)(x))
        x = nn.Dense(1)(x)

        return x
