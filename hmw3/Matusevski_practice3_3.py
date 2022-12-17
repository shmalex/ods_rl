import colorama

import tqdm
import matplotlib.pyplot as plt
import graphviz
import numpy as np
import Frozen_Lake as fl


def init_policy(env):
    actions_set = set()
    row_max = 0
    col_max = 0
    for state in env.get_all_states():
        actions_set.update(env.get_possible_actions(state))
        row_max = max(row_max, state[0])
        col_max = max(col_max, state[1])

    actions = {i:action for i,action in enumerate(sorted(actions_set))}

    policy = np.zeros((row_max+1, col_max+1, len(actions_set)))

    for y,x in env.get_all_states():
        possible_actions = env.get_possible_actions((y,x))
        if len(possible_actions)!=0:
            uniform_prob = 1/len(possible_actions)
            policy[y][x] = uniform_prob
    return policy, list(sorted(actions_set))

def init_values(policy):
     return np.zeros((policy.shape[0]*policy.shape[1],))

def value_iteration(value, gamma):
    new_values = np.zeros(policy.shape[0]*policy.shape[1])
    for state in env.get_all_states():
        state_y, state_x = state
        idx_state_values = state_y*policy.shape[0] + state_x
        actions = []
        for i, action in enumerate(env.get_possible_actions(state)):
            state_action = 0
            policy_prob = policy[state_y][state_x][actions_dict[action]]
            for next_state in env.get_next_states(state, action):
                next_state_y, next_state_x = next_state
                idx_next_state_values = next_state_y*policy.shape[0] + next_state_x
                # reward
                reward = env.get_reward(state, action, next_state)
                # value
                trans_prob = env.get_transition_prob(state, action, next_state)
                next_value = values[idx_next_state_values]
                state_action += policy_prob * trans_prob * (reward + gamma * next_value)
            actions.append(state_action)
        if len(actions)!=0:
            new_values[idx_state_values] = max(actions)
    return new_values

def Q(policy, values, actions_dict):
    new_q = np.zeros(policy.shape)

    for state in env.get_all_states():
        state_y, state_x = state
        for action in env.get_possible_actions(state):
            for next_state in env.get_next_states(state, action):
                next_state_y, next_state_x = next_state

                reward = env.get_reward(state, action, next_state)
                prob = env.get_transition_prob(state, action, next_state)
                next_value = values[next_state_y*policy.shape[0]+next_state_x]

                new_q[state_y][state_x][actions_dict[action]] \
                    += prob *(reward + gamma * next_value)
    return new_q

def policy_improvement(policy, q):
    next_policy = np.zeros(policy.shape)
    for state in env.get_all_states():
        state_y,state_x = state

        idx_action = np.argmax(q[state_y][state_x])
        next_policy[state_y][state_x][idx_action] = 1
    return next_policy

def run_test(policy):
    total_reward = 0
    state = env.reset()
    
    for step in range(100):
        possible_actions = list(sorted(env.get_possible_actions(state)))
        state_y, state_x = state
        actions_distr = policy[state_y][state_x]

        action = np.random.choice(possible_actions,p=actions_distr)
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done: break
    return total_reward, step

if __name__=="__main__":
    env = fl.FrozenLakeEnv(map_name="4x4")
    env.reset()

    ## INIT the pocli, actions
    policy,actions_set = init_policy(env)
    actions_dict ={action:i for i,action in enumerate(sorted(actions_set))}
    actions_arr = [0]*4
    for k,v in actions_dict.items():
        actions_arr[v]=k
    actions_arr, actions_dict
    
    L = 15
    K = 97
    gamma = 0.8

    # train MDP
    values = init_values(policy)
    for k in tqdm.tqdm(range(K), desc='training'):
        for l in range(L):
            values = value_iteration(values, gamma)
        q = Q(policy, values, actions_dict)
        policy = policy_improvement(policy, q)
    policy

    runs = []
    for _ in tqdm.tqdm(range(1000), desc='testing'):
        runs.append(run_test(policy))
    runs = np.array(runs)
    rewards, steps = runs[::,0], runs[::,1]
    print('Average reward', np.mean(runs[::,0]))
