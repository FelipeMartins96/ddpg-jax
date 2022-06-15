from networks import JaxActor, JaxCritic
import jax
import optax
import jax.numpy as jnp


def get_sample_action(act_model):
    def sample(k, params, state, sigma, theta, obs):
        key, k = jax.random.split(k)
        act = act_model.apply(params, obs)
        noise = jax.random.normal(key=k, shape=act.shape) * sigma
        act = jnp.clip(act + noise, -1, 1)
        return key, noise, act

    return jax.jit(sample)


def get_sample_action_ou(act_model):
    def sample(k, params, state, sigma, theta, obs):
        key, k = jax.random.split(k)
        act = act_model.apply(params, obs)
        noise = jax.random.normal(key=k, shape=act.shape)
        noise = state + theta * - state + sigma * noise
        noise = jnp.clip(noise, -2, 2)
        act = jnp.clip(act + noise, -1, 1)
        return key, noise, act

    return jax.jit(sample)

def get_update(act_model, crt_model, optimizer_a, optimizer_c, gamma, tau):
    def critic_loss(
        crt_params, tgt_act_params, tgt_crt_params, obs, acts, rws, dones, next_obs
    ):
        q_value = crt_model.apply(crt_params, obs, acts)  # expected q
        a_next = act_model.apply(
            tgt_act_params, next_obs
        )  # Get an Bootstrap Action for next_obs
        q_next = crt_model.apply(tgt_crt_params, next_obs, a_next)  # bootstrap q next
        q_next = q_next * (1.0 - dones)  # No bootstrap if transition is terminal
        q_ref = rws + q_next * gamma
        loss = jnp.mean((q_value - q_ref) ** 2)
        return loss

    def actor_loss(act_params, crt_params, obs):
        a_cur = act_model.apply(act_params, obs)
        loss = -crt_model.apply(crt_params, obs, a_cur)
        loss = jnp.mean(loss)
        return loss

    crt_loss_grad = jax.value_and_grad(critic_loss)
    act_loss_grad = jax.value_and_grad(actor_loss)

    def update(
        act_params,
        crt_params,
        tgt_act_params,
        tgt_crt_params,
        act_opt_params,
        crt_opt_params,
        obs,
        acts,
        rws,
        dones,
        next_obs,
    ):
        # update actor
        act_loss, act_grad = act_loss_grad(act_params, crt_params, obs)
        updates, new_act_opt_params = optimizer_a.update(act_grad, act_opt_params)
        new_act_params = optax.apply_updates(act_params, updates)

        # update critic
        crt_loss, critic_grad = crt_loss_grad(
            crt_params, tgt_act_params, tgt_crt_params, obs, acts, rws, dones, next_obs
        )
        updates, new_crt_opt_params = optimizer_c.update(critic_grad, crt_opt_params)
        new_crt_params = optax.apply_updates(crt_params, updates)

        # sync networks
        new_tgt_act_params = jax.tree_multimap(
            lambda p, tp: p * tau + tp * (1 - tau), act_params, tgt_act_params
        )
        new_tgt_crt_params = jax.tree_multimap(
            lambda p, tp: p * tau + tp * (1 - tau), crt_params, tgt_crt_params
        )

        return (
            new_act_params,
            new_crt_params,
            new_tgt_act_params,
            new_tgt_crt_params,
            new_act_opt_params,
            new_crt_opt_params,
        ), (act_loss, crt_loss)

    return jax.jit(update)


class DDPG:
    def __init__(self, obs_space, act_space, lr_c, lr_a, gamma, seed, sigma, theta=0.15, ou=False):
        self.key = jax.random.PRNGKey(seed)
        self.key, k0, k1 = jax.random.split(self.key, 3)
        act_size = act_space.shape[0]
        self.actor = JaxActor(act_size)
        self.critic = JaxCritic()
        self.actor_params = self.actor.init(k0, obs_space.sample())
        self.critic_params = self.critic.init(
            k1, obs_space.sample(), act_space.sample()
        )
        self.tgt_actor_params = self.actor_params
        self.tgt_critic_params = self.critic_params
        self.sigma = jnp.array(sigma)
        self.theta = jnp.array(theta)
        self.noise_state = jnp.zeros_like(act_space.sample())

        # Optimizers
        optimizer_c = optax.chain(
            optax.scale_by_adam(),
            optax.scale(-lr_c),
        )
        optimizer_a = optax.chain(
            optax.scale_by_adam(),
            optax.scale(-lr_a),
        )
        self.act_opt_params = optimizer_c.init(self.actor_params)
        self.crt_opt_params = optimizer_a.init(self.critic_params)

        self._sample_action = get_sample_action(self.actor) if not ou else get_sample_action_ou(self.actor)
        self._update = get_update(self.actor, self.critic, optimizer_c, optimizer_a, gamma, 0.005)
        self._get_action = jax.jit(self.actor.apply)

    def sample_action(self, obs):
        self.key, self.state, action = self._sample_action(self.key, self.actor_params, self.noise_state, self.sigma, self.theta, obs)
        return action

    def get_action(self, obs):
        return self._get_action(self.actor_params, obs)

    def update(self, batch):
        obs, acts, rws, dones, next_obs = batch

        (
            self.actor_params,
            self.critic_params,
            self.tgt_actor_params,
            self.tgt_critic_params,
            self.act_opt_params,
            self.crt_opt_params,
        ), (act_loss, crt_loss) = self._update(
            self.actor_params,
            self.critic_params,
            self.tgt_actor_params,
            self.tgt_critic_params,
            self.act_opt_params,
            self.crt_opt_params,
            obs,
            acts,
            rws,
            dones,
            next_obs,
        )

        return act_loss, crt_loss
