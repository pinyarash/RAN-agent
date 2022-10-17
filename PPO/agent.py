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
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, noise=0.1,
                 n_epochs=10, chkpt_dir='models/'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir
        self.noise = noise
        self.max_action = 1.0
        self.min_action = 0.0
        self.n_actions = n_actions
        self.actor = ActorNetwork(n_actions)
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic = CriticNetwork()
        self.critic.compile(optimizer=Adam(learning_rate=alpha))
        self.memory = PPOMemory(batch_size)
        self.noise_t = 0.0
        self.epsilon = 0.2
        self.action_clip = "flip"

    def store_transition(self, state, action, probs, vals, delay,noise, reward, done):
        self.memory.store_memory(state, action, probs, vals, delay,noise, reward, done)

    def save_models(self, name):
        print('... saving models ...')
        self.actor.save(self.chkpt_dir + name +'_actor')
        self.critic.save(self.chkpt_dir + name + '_critic')

    def load_models(self, name):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.chkpt_dir + name +'_actor')
        self.critic = keras.models.load_model(self.chkpt_dir + name +'_critic')
        self.actor.summary()
        self.critic.summary()

    def choose_action_continous(self, observation, delay_list, ue_noise, evaluate=True):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        action_mean = self.actor(state)
        ue_state = observation[-2:]

        dist = tfp.distributions.Normal(action_mean, self.noise)
        entropy = dist.entropy()
        if not evaluate:
            # dist = tfp.distributions.Normal(action_mean, 0.01)
            test_action = action_mean
            action = action_mean

        else:
            action = dist.sample()

        action = tf.clip_by_value(action, self.min_action, self.max_action)
        d1,d2 = delay_list
        delay_diff = abs(d1 - d2)
        print("action_mean :",action_mean.numpy()[0], " sample action :" ,action.numpy()[0], " d1 :",d1, " d2 :",d2, "diff :",delay_diff)

        action = action.numpy()[0]
        action_mean = action_mean.numpy()[0]
        guided_reward = 0
        # Guided action mean and flip
        if self.action_clip == "mean":

            if delay_diff > 1:
                if d1 < d2:
                    print("Increasing traffic to Link1")


                    if action_mean >= action:
                        print("Wrong direction")
                        min_val = max(action_mean,action)
                        action = tf.clip_by_value(action, min_val, action_mean)
                        print(action.numpy())

                    else:
                        print("Right direction")
                        print(action)

                elif d1 > d2:


                    print("Increasing traffic to Link2")
                    if action_mean <= action:
                        print("Wrong direction")
                        max_val = min(action_mean,action)
                        action = tf.clip_by_value(action, action_mean, max_val)
                        print(action.numpy())

                    else :
                        print("Right direction")
                        print(action)

        elif self.action_clip == "flip":

            
            if delay_diff >= 0:
                if d1 < d2:
                    print("Increasing traffic to Link1")

                    diff = action-action_mean
                    if action_mean < 0.5:
                        action_mean = 0.5
                    offset = action_mean + abs(diff)
                    action = tf.clip_by_value(offset, action_mean, offset)
                    bonus_reward = 0
                    print(action.numpy())
                    # if action_mean >= action:
                    #     print("Wrong direction")
                    #     diff = action-action_mean
                    #     if action_mean < 0.5:
                    #         action_mean = 0.5
                    #     offset = action_mean + abs(diff)
                    #     action = tf.clip_by_value(offset, action_mean, offset)
                    #     bonus_reward = 0
                    #     print(action.numpy())

                    # else:
                    #     print("Right direction")
                    #     # action = tf.clip_by_value(action, 0.5, max(0.5,action))
                    #     print(action)
                    #     bonus_reward = 2

                elif d1 > d2:

                    print("Increasing traffic to Link2")
                    diff = action-action_mean
                    if action_mean > 0.5:
                        action_mean = 0.5
                    offset = action_mean - abs(diff)
                    action = tf.clip_by_value(offset, offset, action_mean)
                    print(action.numpy())
                    bonus_reward = 0


                    # if action_mean <= action:
                    #     print("Wrong direction")
                    #     diff = action-action_mean
                    #     if action_mean > 0.5:
                    #         action_mean = 0.5
                    #     offset = action_mean - abs(diff)
                    #     action = tf.clip_by_value(offset, offset, action_mean)
                    #     print(action.numpy())
                    #     bonus_reward = 0

                    # else :
                    #     print("Right direction")
                    #     # action = tf.clip_by_value(action, min(action,0.5), 0.5)

                    #     print(action)
                    #     bonus_reward = 2

        print("================================================================================")
        log_prob = dist.log_prob(action)

        value = self.critic(state)
        value = value.numpy()[0]
        guided_reward = 2
        if not evaluate :
            action = test_action.numpy()[0]
        return action, log_prob, value, guided_reward, entropy


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
        action = np.clip(action, self.min_action, self.max_action)
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

    def learn(self,delay_list):
        actor_loss_list = []
        critic_loss_list = []
        avg_actor_loss, avg_critics_loss = 0,0
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, delay_arr, ue_noise_arr,\
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
                    # Evaluating old actions and values
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch], dtype= tf.float32)
                    # delay_list = tf.convert_to_tensor(delay_arr[batch])
                    ue_noise = tf.convert_to_tensor(ue_noise_arr[batch])
                    #Getting new prob
                    action_mean = self.actor(states)
                    # print(delay_arr[batch])
                    # print(delay_arr[batch].shape)
                    # _, new_probs, _, _ , entropy = self.choose_action_continous(state_arr[batch],delay_arr[batch][0],ue_noise_arr[batch], False)
                    # add guided exploration here


                    dist = tfp.distributions.Normal(action_mean, self.noise)
                    entropy = dist.entropy()
                    new_probs = dist.log_prob(actions)
                    # print(delay_list)
                    # d1,d2 = delay_list
                    # delay_diff = abs(d1 - d2)
                    # print("action_mean :",action_mean.numpy()[0], " d1 :",d1, " d2 :",d2, "diff :",delay_diff)

                    critic_value = self.critic(states)

                    critic_value = tf.squeeze(critic_value, 1)

                    # Finding the ratio (pi_theta / pi_theta__old)
                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    # print(prob_ratio)
                    # Finding Surrogate Loss sur1 = weighted_probs and sur2 = weighted_clipped_probs
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio,
                                                     1-self.policy_clip,
                                                     1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs * advantage[batch]

                    # final loss of clipped objective PPO
                    actor_loss = -tf.math.minimum(weighted_probs,
                                                  weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss) - 0.01*entropy

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

        # self.memory.clear_memory()

        return avg_actor_loss, avg_critics_loss
