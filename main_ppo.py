#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import argparse
from ns3gym import ns3env

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import numpy as np
import gym
import datetime as dt
from ppo import *
from itertools import permutations
import wandb
wandb.init(project="NS3-gym", entity="mai_intel")


__author__ = "Pinyarash Pinyoanuntapong"
__copyright__ = "Copyright (c) 2022, Intel"
__version__ = "0.1.0"
__email__ = "pinyarash pinyoanuntapong"


parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start',
                    type=int,
                    default=1,
                    help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations',
                    type=int,
                    default=1,
                    help='Number of iterations, Default: 1')
args = parser.parse_args()
startSim = bool(args.start)
iterationNum = int(args.iterations)


STORE_PATH = 'logs'
CRITIC_LOSS_WEIGHT = 0.5
ENTROPY_LOSS_WEIGHT = 0.01
ENT_DISCOUNT_RATE = 0.995
BATCH_SIZE = 4
GAMMA = 0.99
CLIP_VALUE = 0.2
LR = 0.001

NUM_TRAIN_EPOCHS = 10

# env = gym.make("CartPole-v0")


ent_discount_val = ENTROPY_LOSS_WEIGHT

port = 5555
simTime = 20 # seconds
stepTime = 0.5  # seconds
seed = 0
simArgs = {"--simTime": simTime,
           "--testArg": 123}
debug = False




env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)

env.reset()


def findPairs(lst, K, N): 
    return [pair for pair in permutations(lst, N) if sum(pair) == K] 

lst = list(range(0, 5))
K = 4
N = 2
# action_list = findPairs(lst,K,N)
# action_list.append((2,2))
# action_list.sort()

action_list = [(0, 4), (1, 3), (2, 2),(3,1),(4,0)]

ob_space = env.observation_space
ac_space = action_list
s_size = env.observation_space.shape[0]
num_actions = len(action_list)

# print("state size space: ", state_size)

# print("Action size: ", env.action_space.n)


# print("Observation space: ", state_size,  ob_space.dtype)
# print("Action space: ", num_actions, ac_space.dtype)

def scale_action(action_space, action):
    """
    Rescale the action from [low, high] to [-1, 1]
    (no need for symmetric action space)

    :param action_space: (gym.spaces.box.Box)
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    low, high = action_space.low, action_space.high
    return 2.0 * ((action - low) / (high - low)) - 1.0


def unscale_action(action_space, scaled_action):
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)

    :param action_space: (gym.spaces.box.Box)
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    low, high = action_space.low, action_space.high
    return low + (0.5 * (scaled_action + 1.0) * (high - low))

stepIdx = 0
currIt = 0

model = PPO_Agent(num_actions)
optimizer = keras.optimizers.Adam(learning_rate=LR)

train_writer = tf.summary.create_file_writer(STORE_PATH + f"/PPO_{dt.datetime.now().strftime('%d%m%Y%H%M')}")

num_steps = 100

# state = env.observation_space
state = np.array(env.reset())/100.0
# print(np.array(state)/100)

episode = 0
total_loss = None
for step in range(num_steps):
    episode_reward_sum = []

    rewards = []
    actions = []
    values = []
    states = []
    dones = []
    probs = []
    for _ in range(BATCH_SIZE):
        _, policy_logits = model(np.reshape(state, [1, s_size]))

        action, value = model.action_value(np.reshape(state,[1, s_size]))
        # gma_action = unscale_action(action)
        new_state, reward, done, _ = env.step(action.numpy()[0])
        print("step: {}, state: {}, action: {}, rew: {},"
                .format(step,state, action, reward))
        actions.append(action)
        values.append(value[0])
        states.append(state)
        dones.append(done)
        probs.append(policy_logits)
        # episode_reward_sum += reward
        episode_reward_sum.append(reward)


        state = np.array(new_state)/100.0

        if total_loss is not None:

            print(f"Episode: {episode}, avg episode reward: {np.mean(episode_reward_sum)}, "
                    f"total loss: {np.mean(total_loss)}, critic loss: {np.mean(c_loss)}, "
                    f"actor loss: {np.mean(act_loss)}, entropy loss {np.mean(ent_loss)}")

        if done:
            rewards.append(0.0)
            state = env.reset()
            state = state
            print(state)

            if total_loss is not None:
                print(f"Episode: {episode}, avg episode reward: {np.mean(episode_reward_sum)}, "
                      f"total loss: {np.mean(total_loss)}, critic loss: {np.mean(c_loss)}, "
                      f"actor loss: {np.mean(act_loss)}, entropy loss {np.mean(ent_loss)}")
            with train_writer.as_default():
                tf.summary.scalar('rewards', np.mean(episode_reward_sum), episode)
            episode_reward_sum = []

            episode += 1
        else:
            rewards.append(reward)

    _, next_value = model.action_value(np.reshape(state,[1, s_size]))
    discounted_rewards, advantages = get_advantages(rewards, dones, values, next_value[0])

    actions = tf.squeeze(tf.stack(actions))
    probs = tf.nn.softmax(tf.squeeze(tf.stack(probs)))
    action_inds = tf.stack([tf.range(0, actions.shape[0]), tf.cast(actions, tf.int32)], axis=1)

    total_loss = np.zeros((NUM_TRAIN_EPOCHS))
    act_loss = np.zeros((NUM_TRAIN_EPOCHS))
    c_loss = np.zeros(((NUM_TRAIN_EPOCHS)))
    ent_loss = np.zeros((NUM_TRAIN_EPOCHS))

    for epoch in range(NUM_TRAIN_EPOCHS):
        loss_tuple = train_model(model, action_inds, tf.gather_nd(probs, action_inds),
                                 states, advantages, discounted_rewards, optimizer,
                                 ent_discount_val)
        total_loss[epoch] = loss_tuple[0]
        c_loss[epoch] = loss_tuple[1]
        act_loss[epoch] = loss_tuple[2]
        ent_loss[epoch] = loss_tuple[3]
    ent_discount_val *= ENT_DISCOUNT_RATE

    with train_writer.as_default():
        tf.summary.scalar('tot_loss', np.mean(total_loss), step)
        tf.summary.scalar('critic_loss', np.mean(c_loss), step)
        tf.summary.scalar('actor_loss', np.mean(act_loss), step)
        tf.summary.scalar('entropy_loss', np.mean(ent_loss), step)
        tf.summary.scalar('rewards', np.mean(episode_reward_sum), step)

    wandb.log({"Step": step, "reward": np.mean(episode_reward_sum),"critic_loss": np.mean(c_loss), "actor_loss": np.mean(act_loss)})



# # try:
# #     while True:
# #         print("Start iteration: ", currIt)
# #         obs = env.reset()
# #         print("Step: ", stepIdx)
# #         print("---obs:", obs)

# #         while True:
# #             stepIdx += 1
# #             action = env.action_space.sample()
# #             print("---action: ", action)

# #             print("Step: ", stepIdx)
# #             obs, reward, done, info = env.step(action)
# #             print("---obs, reward, done, info: ", obs, reward, done, info)
# #             if done:
# #                 stepIdx = 0
# #                 if currIt + 1 < iterationNum:
# #                     env.reset()
# #                 break

# #         currIt += 1
# #         if currIt == iterationNum:
# #             break

# except KeyboardInterrupt:
#     print("Ctrl-C -> Exit")
# finally:
#     env.close()
#     print("Done")