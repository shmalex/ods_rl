import os
import gym
import gym_maze
import numpy as np
import random
import time
import tqdm
from matplotlib import pyplot as plt

import json

for i in tqdm.tqdm_notebook(range(1)):
    pass
for i in tqdm.notebook.tqdm(range(1)):
    pass



class CrossEntropyMethod():
    def __init__(self, name, state_n, action_n):
        self.name = name
        self.state_n = state_n
        self.action_n = action_n
        self.policy = np.ones((self.state_n, self.action_n)) / self.action_n
    
    def get_action(self, state):
        return int(np.random.choice(np.arange(self.action_n), p=self.policy[state]))
    def get_det_action(self, state):
        return np.argmax(self.policy[state])
    
    def update_policy(self, elite_trajectories):
        pre_policy = np.zeros((self.state_n, self.action_n))
        
        # counter
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                pre_policy[state][action] += 1
        
        # 
        for state in range(self.state_n):
            if sum(pre_policy[state]) == 0:
                self.policy[state] = np.ones(self.action_n) / self.action_n
            else:
                self.policy[state] = pre_policy[state] / sum(pre_policy[state])
        self.save_policy()
        return None
    def save_policy(self):
        np.save(f'{env.spec.id}-{self.name}', self.policy)
    def load_policy(self):
        fname = f'{env.spec.id}-{self.name}.npy'
        if os.path.exists(fname):
            self.policy = np.load(fname)

def sample_policy(policy, action_n):
    ret_policy = np.zeros(policy.shape)
    actions = np.arange(action_n)
    for i, p in enumerate(policy):
        idx =np.random.choice(actions, p=p)
        ret_policy[i][idx] = 1
    return ret_policy
def get_trajectory(env, agent, trajectory_len):
    trajectory = {
        'states':[], 
        'actions':[],
        'total_reward': 0}
    state = env.reset()
    trajectory['states'].append(state)
    for _ in range(trajectory_len):
#         action = agent.get_action(state)
        action = agent.get_det_action(state)
        trajectory['actions'].append(action)
        state, reward, done, _ = env.step(action)
        trajectory['states'].append(state)
        trajectory['total_reward'] += reward
        if done:
            break
    return trajectory

name = f'det_CEM_{episode_n}_{trajectory_len}_{trajectory_n}_{q_param}_{lambd}_{det_policy_n}'

agent = CrossEntropyMethod(name, state_n, action_n)
env = gym.make("Taxi-v3")
episode_data = []
for _ in tqdm.notebook.tqdm(range(episode_n), desc='Episodes', position=0):

    saved_policy = agent.policy
    det_trajectories =[]
    det_policies_mean_total_reward = []

    for _ in range(det_policy_n):
        det_policy = sample_policy(saved_policy, action_n)

        agent.policy = det_policy
        trajectories = [get_trajectory(env, agent, trajectory_len) for _ in range(trajectory_n)]
        det_trajectories.append(trajectories)

        mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
        det_policies_mean_total_reward.append(mean_total_reward)

    elite_trajectories = get_elite_det_trajectories(det_trajectories,
                                                    det_policies_mean_total_reward,
                                                    q_param)

    episode_data.append((det_policies_mean_total_reward, len(elite_trajectories)))
    if len(elite_trajectories) > 0:
        agent.update_policy(elite_trajectories)
    else:
        agent.policy = saved_policy
    #         print(mean_total_reward, len(elite_trajectories))
    exp['episode_data'] = episode_data
    json.dump(exp, open(f'./experiments/{agent.name}.json','w'))