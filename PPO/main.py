#!/usr/bin/env python3

from queue import Empty
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
    low, high = 0,40
    
    norm = -10*(1.0 * ((data - low) / (high - low)) - 1.0)
    
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
    d_max = 220
    d_min = 0
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

def find_traffic_load(df,ns3_time):
    traffic_sche = df.index.values*20
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
    group = np.vstack((traffic_load,wifi_rate_state,lte_rate_state,d1,d2)).T
    #convert 2d array to 1d array
    states = group.flatten()
    # states = [UE1_delay, UE1_mcs, UE1_cqi, UE1_traffic_load, UE2_delay, UE2_mcs, UE2_cqi, UE2_traffic_load, ...]

    # return np.append(delay_state, cqi_state)
    return states
    # return delay_state


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
    parser.add_argument('--pretrained',
                        type=str,
                        default="",
                        help='Load pre-trained model: ')

    args = parser.parse_args()
    
    startSim = bool(0)
    iterationNum = int(1)

    port = 5555
    simTime = 200 # seconds
    stepTime = 0.1  # seconds
    seed = 0
    simArgs = {"--simTime": simTime,
            "--testArg": 123}
    debug = False

    env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)

    # env = gym.make('CartPole-v0')
    N_Learn = 4
    batch_size = 4
    n_epochs = 4
    alpha = 0.001
    # alpha = 0.01


    num_users = 4
    # state_size = 2
    action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    action_dim = 1
    features = 5
    state_dim =features*num_users
    print("Action size: ", action_dim)
    print("state size space: ", state_dim)

    agent = Agent(n_actions=action_dim, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=state_dim)
    n_games = 10

    best_score = 0
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    # max_iter = simTime/ stepTime
    max_iter = 10


    traffic_load = 0
    train = bool(args.train)
    config_params = {"Train":train, "env_type":args.env  ,"lr": alpha, "batch_size": batch_size, "n_epochs":n_epochs, "N_Learn":N_Learn }
    wandb.init(project="NS3-gym", entity="mai_intel",name= args.name, config=config_params)
 
    # wandb.config.update({"Train":train,"lr": alpha, "batch_size": batch_size, "n_epochs":n_epochs, "N_Learn":N_Learn })
    save_model_name = "env" + str(args.env)
    load_model_name = args.pretrained

    # Read network trace for particular environment (env1, env2, env3)
    traffic_trace = pd.read_csv('/home/pinyarash/NS3/applications.simulators.mengleiz-gma-sim/traffic_trace/loads_env1.csv',index_col=None) 
    print(traffic_trace)
    # if test load the pre-trained model and disable exploration
    if not train:
        agent.load_models(load_model_name)
        eval = False
        print("Testing the model")

    if train and args.pretrained != "":
        agent.load_models(load_model_name)
        eval = True
        print("Training the model with pre-trained model")

    # for i in range(n_games):
    observation = get_state(env.reset(),traffic_load)
    # print(observation)
    done = False
    score = 0

    # for n_steps in range()):
    while not done:
        action_list = []
        probs_list = []
        val_list = []
        split_ratio_list = []
        # pkt_split_list = []
        # action, probs, val = agent.choose_action(observation)
        for ueid in range(num_users):
            action, probs, val = agent.choose_action_continous(observation, eval)
            action_list.extend(action)
            probs_list.extend(probs)
            val_list.extend(val)
            split_ratio_list.extend(ratio_to_split_ratio(action))
        
        # print(split_ratio_list)
        pkt_split_list = ratio_to_num(np.array(split_ratio_list))
        print(pkt_split_list)

        # split_ratio = ratio_to_split_ratio(action)
        # pkt_split = ratio_to_num(split_ratio)

        state_ , avg_delay, done, info = env.step(pkt_split_list)
        traffic_load = find_traffic_load(traffic_trace, info)
        observation_ = get_state(state_, traffic_load)
        reward = delay_to_scale(avg_delay)
        n_steps += 1

        print("step: {}, NS3_Time: {},state: {},reward: {}, avg_owd:{}"
            .format(n_steps,info,observation_, reward, avg_delay))
        cur_time = int(float(info))

        # print("split_ratio: {}, pkt_split: {}"
        #     .format( split_ratio, pkt_split))

        # score += reward

        wandb.log({"step": n_steps,"episode": learn_iters, "reward":reward, "avg_delay":avg_delay,
        "delay1_1" : observation_[0],"delay1_2" : observation_[1], "delay2_1" : observation_[2],"delay2_2" : observation_[3],
        "delay3_1" : observation_[4],"delay3_2" : observation_[5], "delay4_1" : observation_[6],"delay4_2" : observation_[7],
        "reward" : reward, "avgDelay": avg_delay, "time": cur_time })

        # Single UE
        # agent.store_transition(observation, action,probs, val, reward, done)

        for ueid in range(num_users):
            agent.store_transition(observation_, action_list[ueid],probs_list[ueid], val_list[ueid], reward, done)

        score_history.append(reward)
        # wandb.log({"episode": i, "reward":avg_score})


        if n_steps % N_Learn == 0:

            if train:
                a_loss, c_loss = agent.learn()
                agent.save_models(save_model_name)
            learn_iters += 1
            avg_score = np.mean(score_history)
            print('episode', learn_iters, 'avg score %.1f' % avg_score,
        'time_steps', n_steps, 'learning_steps', learn_iters )
            wandb.log({"step": n_steps,"episode": learn_iters, "avg_reward":avg_score, "traffic_load": traffic_load})
            score_history = []
                

        observation = observation_
        if n_steps > max_iter:
            done = True
            env.close()
            print("Done")
            break

            # if learn_iters == 49:
            #     agent.save_models()
            # if learn_iters >3:
            #     if avg_score > best_score:
                    
            #         best_score = avg_score
            #         agent.save_models()


    # x = [i+1 for i in range(len(score_history))]
    # plot_learning_curve(x, score_history, figure_file)