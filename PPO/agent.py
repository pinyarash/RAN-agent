from queue import Empty
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from memory import PPOMemory
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64,
                 n_epochs=10, chkpt_dir='models/'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir
        self.noise = 0.1
        self.max_action = 1.0
        self.min_action = 0.0
        self.n_actions = n_actions
        self.actor = ActorNetwork(n_actions)
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic = CriticNetwork()
        self.critic.compile(optimizer=Adam(learning_rate=alpha))
        self.memory = PPOMemory(batch_size)
        self.noise_t = 0.0
        self.epsilon = 0.3

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self, name):
        print('... saving models ...')
        self.actor.save(self.chkpt_dir + name +'_actor')
        self.critic.save(self.chkpt_dir + name + '_critic')

    def load_models(self, name):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.chkpt_dir + name +'_actor')
        self.critic = keras.models.load_model(self.chkpt_dir + name +'_critic')

    def choose_action_continous(self, observation, evaluate=True):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        action_mean = self.actor(state)
        # if not evaluate:
        #     actions += tf.random.normal(shape=[self.n_actions],
        #                                 mean=0.0, stddev=self.noise)
        # note that if the env has an action > 1, we have to multiply by
        # max action at some point
        # self.noise_t = max(self.epsilon, 0) * OU.function(action_mean, 0, 0.15, 0.2)

        dist = tfp.distributions.Normal(action_mean, self.noise)

        if not evaluate:
            action = action_mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        actions = tf.clip_by_value(action, self.min_action, self.max_action)
        actions = actions.numpy()[0]

        value = self.critic(state)
        value = value.numpy()[0]

        return actions, log_prob, value


    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])

        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs)
        # dist = tfp.distributions.Categorical(probs=probs, dtype=tf.float32)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # print("probs :",probs.numpy()[0], " action :" ,action.numpy()[0], " log_prob :" ,log_prob.numpy()[0])
        value = self.critic(state)

        action = action.numpy()[0]
        value = value.numpy()[0]
        log_prob = log_prob.numpy()[0]

        return action, log_prob, value

    # def choose_action(self,state):
    #     actor_net = self.build_net(state.shape)

    #     (a_mu, a_sigma) = actor_net(state)
    #     pi_a = tfp.distributions.Normal(a_mu,a_sigma)
    #     action = tf.squeeze(pi_a.sample(1), 0)
    #     # pi_action = [tf.random.normal([1],mean = amu[i], stddev = a_sigma[i])[0].numpy() for i in range(self.n_actions)]
    #     action = tf.clip_by_value(action, self.action_bound[0], self.action_bound[1])
    #     action = tf.squeeze(action ,axis = 0)

    #     return action.numpy(), tf.squeeze(pi_a.prob(action.numpy()) , 0).numpy() #shape:

    def learn(self):
        actor_loss_list = []
        critic_loss_list = []
        avg_actor_loss, avg_critics_loss = 0,0
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (
                        1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch], dtype= tf.float32)

                    action_mean = self.actor(states)
                    # probs = self.actor(states)
                    # dist = tfp.distributions.Categorical(probs)
                    dist = tfp.distributions.Normal(action_mean, self.noise)
                    entropy = dist.entropy()
                    new_probs = dist.log_prob(actions)

                    critic_value = self.critic(states)

                    critic_value = tf.squeeze(critic_value, 1)

                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio,
                                                     1-self.policy_clip,
                                                     1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs,
                                                  weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]
                    # critic_loss = tf.math.reduce_mean(tf.math.pow(
                    #                                  returns-critic_value, 2))
                    critic_loss = keras.losses.MSE(critic_value, returns)
                    # print("actor_loss: ", actor_loss, "critic_loss: ", critic_loss)

                actor_params = self.actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.actor.optimizer.apply_gradients(
                        zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(
                        zip(critic_grads, critic_params))
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)
                a_loss, c_loss = actor_loss, critic_loss
                avg_critics_loss = np.mean(critic_loss_list)
                avg_actor_loss = np.mean(actor_loss_list)
        # print("actor_loss_list: ", actor_loss_list, "critic_loss_list: ", critic_loss_list)
        # print("actor_loss: ", a_loss, "critic_loss: ", c_loss)

        self.memory.clear_memory()

        return avg_actor_loss, avg_critics_loss
