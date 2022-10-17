#!/usr/bin/env python3

from email import policy
from queue import Empty
from xmlrpc.client import boolean
import gym
import numpy as np
from ns3gym import ns3env
import argparse
import pandas as pd

from agent import Agent
import wandb

np.set_printoptions(precision=4)

num_users = 4

wifi_peakrate_List = np.array([0,6.5, 13, 19.5, 26, 39, 52, 58.5, 65])
# cqiList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def delay_to_scale(data):
    """
    Rescale the action from [low, high] to [-1, 1]
    (no need for symmetric action space)

    :param action_space: (gym.spaces.box.Box)
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    # low, high = 0,220
    # return -10*(2.0 * ((data - low) / (high - low)) - 1.0)
    low, high = 0,20
    norm = np.clip(data, 0.1, 200)

    norm = -10*(1.0 * ((data - low) / (high - low)) - 1.0)
    # norm = (-1*np.log(norm) + 3) * 2.5
    norm = np.clip(norm, -10, 20)

    return norm

def ratio_to_num(data):
    '''
    Convert ratio [0,1] to [0,32]
    '''
    d_max = 32
    d_min = 0
    pkt_split = data * (d_max - d_min) + d_min
    return np.rint(pkt_split).astype(np.uint8)

def delay_to_state(data):
    '''
    Convert int [0,120] to [0,1]
    '''

    d_max = 200
    d_min = 0
    data = np.clip(data, d_min, d_max)
 
    norm = (data - d_min) / (d_max - d_min)
    return norm

def cqi_to_state(data):
    '''
    Convert int [0,15] to [0,1]
    '''
    d_max = 15
    d_min = 1
    return (data - d_min) / (d_max - d_min)

def mcs_to_state(data):
    '''
    Convert int [0,15] to [0,1]
    '''
    d_max = 7
    d_min = 1
    return (data - d_min) / (d_max - d_min)

def wifi_rate_to_state(data):
    '''
    Convert int [0,15] to [0,1]
    '''
    d_max = 65
    d_min = 0
    return (data - d_min) / (d_max - d_min)

def lte_rate_to_state(data):
    '''
    Convert int [0,15] to [0,1]
    '''
    d_max = 75
    d_min = 0
    return (data - d_min) / (d_max - d_min)

def num_to_ratio(data):
    '''
    Convert int [0,32] to [0,1]
    '''
    d_max = 32
    d_min = 0    
    return (data - d_min) / (d_max - d_min)

def traiffc_to_level(data):
    '''
    Convert int [0,32] to [0,1]
    '''
    d_max = 60
    d_min = 0    
    return (data - d_min) / (d_max - d_min)

def ratio_to_split_ratio(action):
    # convert ratio for two links
    return np.array([action[0],1 - action[0]])

# def get_state(obseravation):
#     print(obseravation)
#     state = delay_to_state(np.array(obseravation)[:2])
#     num_split = num_to_ratio(np.array(obseravation)[2:])
#     # print(state,num_split)
#     return np.concatenate((state, num_split))

def find_traffic_load(df,ns3_time,traffic_interval):
    traffic_sche = df.index.values*traffic_interval
    # print(traffic_sche)
    # int(float(ns3_time))
    less_arr = traffic_sche <= int(float(ns3_time))
    low_array = traffic_sche[less_arr]
    index = np.where(traffic_sche == low_array[-1])[0][0]
    cur_traffic_load = df.load[index]
    return cur_traffic_load
    # print(cur_traffic_load)

def get_state(obseravation, traffic_load):
    # observation = [delay, mcs, cqi, traffic_load]
    """
    :param prm:
    :param env:
    :param obseravation:
        obseravation[:8]  # WiFi(odd)/LTE(even) link delay (ms)
        obseravation[8:12]  # WiFi peak rate in Mbps
        obseravation[12:16]  # LTE peak rate in Mbps
        obseravation[16:]  # avg traffic arrival rate in Mbps
    :return:
    states = [UE1_delay, UE1_mcs, UE1_cqi, UE1_traffic_load, UE2_delay, UE2_mcs, UE2_cqi, UE2_traffic_load]
    """


    delay_index = num_users * 2 # Find max index of delay links for all users
    RAN_measurements_index = num_users # Find max index of RAN measurements for all users
    print(obseravation, traffic_load)
    #Parse observation to get delay and RAN measurements for all users
    delay_state = delay_to_state(np.array(obseravation)[:8])
    # mcs_state = mcs_to_state(np.array(obseravation)[delay_index:-RAN_measurements_index])
    wifi_rate_state = wifi_rate_to_state(wifi_peakrate_List[np.array(obseravation)[8:12]])
    lte_rate_state = lte_rate_to_state(np.array(obseravation)[12:])
    # print(wifi_rate_state)
    # print(lte_rate_state)
    traffic_load = traiffc_to_level(np.full((1, 4), traffic_load))
    #group link delay for each UE
    d1 = delay_state[::2]
    d2 = delay_state[1::2]
    #reposition of each featers for each UE
    # Feature Matrix for each UE (features x num of UEs)
    # group = np.vstack((traffic_load,wifi_rate_state,lte_rate_state,d1,d2)).T
    group = np.vstack((traffic_load,wifi_rate_state,lte_rate_state)).T
    # group = np.vstack((wifi_rate_state,lte_rate_state,d1,d2)).T

    # group = np.vstack((wifi_rate_state,lte_rate_state)).T

    #convert 2d array to 1d array
    states = group.flatten()
    # states = [UE1_delay, UE1_mcs, UE1_cqi, UE1_traffic_load, UE2_delay, UE2_mcs, UE2_cqi, UE2_traffic_load, ...]

    # return np.append(delay_state, cqi_state)
    return states
    # return delay_state

def extract_link_delay(obseravation):
    """
    Parse observation to get Wifi/LTE Link delay for all users
    :param obseravation: (obseravation)
    :return: (np.ndarray)
    """
    delay_list = np.array(obseravation)[:8]
    return delay_list.reshape(4,2)

def shiftList(states,n):
    return states[n:] + states[:n]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start simulation script on/off')
    parser.add_argument('--start',
                        type=int,
                        default=1,
                        help='Start ns-3 simulation script 0/1, Default: 1')
    parser.add_argument('--iterations',
                        type=int,
                        default=1,
                        help='Number of iterations, Default: 1')
    parser.add_argument('--name',
                        type=str,
                        default="Test",
                        help='Name of Experiment')
    parser.add_argument('--train',
                        type=int,
                        default=1,
                        help='Train or Test 1/0, Default: 1')
    parser.add_argument('--env',
                        type=int,
                        default=1,
                        help='Deployment case Default: 1')
    parser.add_argument('--load',
                        type=str,
                        default="",
                        help='Load pre-trained model: ')
    parser.add_argument('--reward',
                    type=str,
                    default="",
                    help='Load pre-trained model: ')
    parser.add_argument('--gma',
                    type=bool,
                    default=False,
                    help='GMA policy, only monitor reward: ')
    parser.add_argument('--shift',
                    type=bool,
                    default=False,
                    help='shift input location: ')
    parser.add_argument('--trace_time',
                        type=int,
                        default=10,
                        help='Deployment case Default: 1')
    args = parser.parse_args()
    
    startSim = bool(0)
    iterationNum = int(1)

    port = 5555
    simTime = 100 # seconds
    stepTime = 0.1  # seconds
    seed = 0
    simArgs = {"--simTime": simTime,
            "--testArg": 123}
    debug = False
    shift = args.shift
    env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)

    # env = gym.make('CartPole-v0')
    N_Learn = 4
    batch_size = 16
    n_epochs = 5
    alpha = 1e-5
    beta = 1e-5
    noise = 0.05
    policy_clip = 0.8

    num_users = 4
    # state_size = 2
    action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    action_dim = 1
    features = 3
    state_dim =features*num_users + features

    print("Action size: ", action_dim)
    print("state size space: ", state_dim)

    agent = Agent(n_actions=action_dim,
                  batch_size=batch_size,
                  alpha=alpha,
                  n_epochs=n_epochs,
                  input_dims=state_dim,
                  noise=noise,
                  policy_clip= policy_clip
                  )
    n_games = 10

    best_score = 0
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    max_iter = simTime/ stepTime
    # max_iter = 10


    traffic_load = 0
    traffic_interval = args.trace_time

    train = bool(args.train)
    config_params = {"Train":train, "env_type":args.env  ,"lr": alpha, "batch_size": batch_size, "n_epochs":n_epochs, "N_Learn":N_Learn, "noise":noise,
                     "policy_clip":policy_clip, "max_iter":max_iter, "num_users":num_users, "features":features, "simTime":simTime, "stepTime":stepTime}
    wandb.init(project="NS3-gym", entity="mai_intel",name= args.name, config=config_params)
 
    # wandb.config.update({"Train":train,"lr": alpha, "batch_size": batch_size, "n_epochs":n_epochs, "N_Learn":N_Learn })
    save_model_name = "env" + str(args.env)+"_" + args.reward
    load_model_name = args.load

    # Read network trace for particular environment (env1, env2, env3)
    env_path = "/home/pinyarash/NS3/applications.simulators.mengleiz-gma-sim/traffic_trace/loads_"

    if args.env == 1:
        env_file = "env1.csv"
    elif args.env == 2:
        env_file = "env2.csv"
    elif args.env == 3:
        env_file = "env3.csv"
    elif args.env == 4:
        env_file = "env2.csv"
    elif args.env == 5:
        env_file = "env5.csv"
    elif args.env == 0:
        env_file = "all.csv"
    traffic_trace = pd.read_csv(env_path+env_file,index_col=None) 

    print(traffic_trace)
    # if test load the pre-trained model and disable exploration
    if not train:
        agent.load_models(load_model_name)
        eval = False
        print("Testing the model")
        # traffic_interval = 3
    if train and args.load != "":
        agent.load_models(load_model_name)
        eval = True
        print("Training the model with pre-trained model")
        # traffic_interval=3
    # for i in range(n_games):
    observation = get_state(env.reset(),traffic_load)
    delay_list = extract_link_delay(observation)
    # print(observation)
    done = False
    score = 0
    prev_reward = 0
    # for n_steps in range()):
    while not done:
        action_list = []
        probs_list = []
        val_list = []
        split_ratio_list = []
        ue_noise_list = []
        guided_reward = []
        # pkt_split_list = []
        # action, probs, val = agent.choose_action(observation)
        bonus_reward = 0
        for ueid in range(num_users):
            diff = abs(delay_list[ueid][0] - delay_list[ueid][1])
            ue_noise = agent.noise
            if diff < 2:
                ue_noise = 0.01
            if shift:
                # obs = np.roll(observation, -1*features*ueid)
                local_obs = observation[ueid*features:ueid*features+features]
                obs = np.array([])
                obs = np.append(observation,local_obs )
                print("shift_states: ", obs)

            action, probs, val, bonus_reward, _ = agent.choose_action_continous(obs,delay_list[ueid],ue_noise, eval)
            action_list.extend(action)
            probs_list.extend(probs)
            val_list.extend(val)
            ue_noise_list.append(ue_noise)
            guided_reward.append(bonus_reward)
            split_ratio_list.extend(ratio_to_split_ratio(action))
            bonus_reward += bonus_reward
        # print(split_ratio_list)
        pkt_split_list = ratio_to_num(np.array(split_ratio_list))
        print(pkt_split_list)


        state_ , avg_delay, done, info = env.step(pkt_split_list)
        delay_list_ = extract_link_delay(state_)
        traffic_load = find_traffic_load(traffic_trace, info, traffic_interval)

        observation_ = get_state(state_, traffic_load)
        reward = delay_to_scale(avg_delay) + bonus_reward

        n_steps += 1

        print("step: {}, NS3_Time: {},state: {},reward: {}, avg_owd:{}"
            .format(n_steps,info,observation_, reward, avg_delay))
        cur_time = int(float(info))

        # if reward > 10.0:
        #     if agent.noise > 0.01:
        #         agent.noise -= 0.005
        # elif 0.1 < reward < 8.9:
        #     # if agent.noise > 0.04:
        #         agent.noise = 0.05
        # elif reward < 0.1:
        #     if agent.noise < 0.05:
        #         agent.noise += 0.01
        if avg_delay <= 2.0:
            if agent.noise > 0.01:
                agent.noise -= 0.005
        elif 2.0 < avg_delay < 20:
            if agent.noise > 0.01:
                agent.noise = 0.03                
        elif 20 < avg_delay < 40:
            # if agent.noise > 0.04:
                agent.noise = 0.05
        elif avg_delay > 40:
            if agent.noise < 0.08:
                agent.noise += 0.01

        if args.gma:
            wandb.log({"step": n_steps,"episode": learn_iters, "reward":reward, "avg_delay":avg_delay, "time": cur_time })
        else:
            wandb.log({"step": n_steps,"episode": learn_iters, "reward":reward, "avg_delay":avg_delay, "time": cur_time,
            "UE1_link1": pkt_split_list[0], "UE1_link2": pkt_split_list[1], "UE2_link1": pkt_split_list[2], "UE2_link2": pkt_split_list[3],
            "UE3_link1": pkt_split_list[4], "UE3_link2": pkt_split_list[5], "UE4_link1": pkt_split_list[6], "UE4_link2": pkt_split_list[7] })

        # Single UE
        # agent.store_transition(observation, action,probs, val, reward, done)

        for ueid in range(num_users):
            if shift:
                # obs = np.roll(observation_, -1*features*ueid)
                local_obs = observation[ueid*features:ueid*features+features]
                obs = np.array([])
                obs = np.append(observation,local_obs )
                diff = abs(delay_list_[ueid][0] - delay_list_[ueid][1])
                ue_reward = delay_to_scale(diff)
            agent.store_transition(obs, action_list[ueid],probs_list[ueid], val_list[ueid],delay_list[ueid],ue_noise_list[ueid] ,reward, done)
            # agent.store_transition(obs, action_list[ueid],probs_list[ueid], val_list[ueid],delay_list[ueid],ue_noise_list[ueid] ,ue_reward + guided_reward[ueid], done)

        score_history.append(reward)        

        if n_steps % N_Learn == 0:

            if train:
                a_loss, c_loss = agent.learn(delay_list_)
                # agent.save_models(save_model_name)
                agent.memory.clear_memory()

            learn_iters += 1
            avg_score = np.mean(score_history)
            if avg_score > best_score:
                best_score = avg_score
                if train:
                    agent.save_models(save_model_name+"_best")
            print('episode', learn_iters, 'avg score %.1f' % avg_score,
        'time_steps', n_steps, 'learning_steps', learn_iters )
            if args.gma or not train:
                wandb.log({"step": n_steps,"episode": learn_iters, "time": cur_time, "avg_reward":avg_score, "traffic_load": traffic_load})
            else:
                wandb.log({"step": n_steps,"episode": learn_iters, "time": cur_time, "avg_reward":avg_score, "traffic_load": traffic_load, "noise": agent.noise, "a_loss": a_loss, "c_loss": c_loss})
            score_history = []
                

        observation = observation_
        delay_list = delay_list_
        if train and n_steps == 240:
            agent.save_models(save_model_name)


        if n_steps > max_iter:
            agent.save_models(save_model_name)
            done = True
            env.close()
            print("Done")
            break
