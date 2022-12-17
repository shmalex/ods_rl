
import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
import numpy as np
import gym 
import matplotlib.pyplot as plt
import tqdm
import time
import glob
import json
import os
import datetime



TASKS_DIR = './tasks/'
EXP_DIR = './experiments/'

state_dim = 8
action_n = 4
env = gym.make('LunarLander-v2')

class CrossEntropyMethod(nn.Module):
    def __init__(self, name, state_dim, action_n, layers_n, lr=0.01):
        super().__init__()
        self.name = name
        self.state_dim = state_dim
        self.action_n = action_n
        self.lr = lr
        
        if len(layers_n)==1:
            self.network = nn.Sequential(
                nn.Linear(self.state_dim, layers_n[0]),
                nn.ReLU(),
                nn.Linear(layers_n[0], self.action_n)
            )
        if len(layers_n)==2:
            self.network = nn.Sequential(
                nn.Linear(self.state_dim, layers_n[0]),
                nn.ReLU(),
                nn.Linear(layers_n[0], layers_n[1]),
                nn.ReLU(),
                nn.Linear(layers_n[1], self.action_n)
            )
        if len(layers_n)==3:
            self.network = nn.Sequential(
                nn.Linear(self.state_dim, layers_n[0]),
                nn.ReLU(),
                nn.Linear(layers_n[0], layers_n[1]),
                nn.ReLU(),
                nn.Linear(layers_n[1], layers_n[2]),
                nn.ReLU(),
                nn.Linear(layers_n[2], self.action_n)
            )
        
        self.softmax = nn.Softmax()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, _input):
        return self.network(_input)
    
    def get_action(self, state):
        state = torch.FloatTensor(state)# .to(device)
        logits = self.network(state)
        # action_prob = self.softmax(logits).cpu().detach().numpy()
        action_prob = self.softmax(logits).detach().numpy()
        action = np.random.choice(self.action_n, p=action_prob)
        return action

    def update_policy(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for trajectory in elite_trajectories:
            elite_states.extend(trajectory['states'])
            elite_actions.extend(trajectory['actions'])
        assert len(elite_states)==len(elite_actions), f'{len(elite_states)},{len(elite_actions)}'

        elite_states = torch.FloatTensor(elite_states)#.to(device)
        elite_actions = torch.LongTensor(elite_actions)#.to(device)
        loss = self.loss(self.forward(elite_states), elite_actions)
        # calculating gradients
        loss.backward()
        # update the weights
        self.optimizer.step()
        # zero grad
        self.optimizer.zero_grad()
        self.save_policy()

    def get_name(self):
        return os.path.join(EXP_DIR, self.name+'.nn')

    def save_policy(self):
        torch.save(self.state_dict(), self.get_name())

    def load_policy(self):
        if os.path.exists(self.get_name()):
            self.load_state_dict(torch.load(self.get_name()))
    def load_n_policy(self, n):
        files = list(glob.glob(os.path.join(EXP_DIR, self.name,str(n)+'_*.nn')))
        print('policies count', len(files))
#         start_from_start = len(files) == 0
        if len(files) > 0:
            file = files[0]# list(sorted(files, key=lambda x: int(x.split('/')[-1].split('_')[0])))[-1]
            print('loading last policy',file)
            self.load_state_dict(torch.load(file))
        else:
            print(">>>>>>>>>>>>>>>>>>>>policy not found")
    def load_last_policy(self):
        files = list(glob.glob(os.path.join(EXP_DIR, self.name,'*.nn')))
        print('policies count', len(files))
        start_from_start = len(files) == 0
        if len(files) > 0:
            file = list(sorted(files, key=lambda x: int(x.split('/')[-1].split('_')[0])))[-1]
            print('loading last policy',file)
            self.load_state_dict(torch.load(file))
        else:
            print(">>>>>>>>>>>>>>>>>>>>policy not found")

def get_trajectory(env, agent, trajectory_len, visualize=False):
    trajectory = {
        'states':[], 
        'actions':[],
        'total_reward': 0}
    state = env.reset()
#     trajectory['states'].append(state)
    for _ in range(trajectory_len):
        action = agent.get_action(state)
        trajectory['states'].append(state)
        trajectory['actions'].append(action)
        state, reward, done, _ = env.step(action)
#         if reward==-100:print('CRash', state, reward, done, _)
#         if reward==200:print(state, reward, done, _)
        trajectory['total_reward'] += reward
        if done:
            break
        
        if visualize:
            env.render()

    assert len(trajectory['actions'])==len(trajectory['states']), f"gt {len(trajectory['actions'])},{len(trajectory['states'])}"
    return trajectory

def get_elite_trajectories(trajectories, q_param):
    total_rewards = [trajectory['total_reward'] for trajectory in trajectories]
    quantile = np.quantile(total_rewards, q=q_param) 
    return [trajectory for trajectory in trajectories if trajectory['total_reward'] > quantile]



if __name__ == "__main__":
    layers_n=[200] 
    episode_n=2#130
    trajectory_len=1000 
    trajectory_n=300 
    q_param=0.7 
    lr=0.01
    agent = CrossEntropyMethod('test', state_dim, action_n, layers_n, lr)
    env = gym.make('LunarLander-v2')
    for i in tqdm.tqdm(range(episode_n)):
        trajectories = [get_trajectory(env, agent, trajectory_len) for _ in range(trajectory_n)]

        mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
#         print(mean_total_reward)
        elite_trajectories = get_elite_trajectories(trajectories, q_param)

        if len(elite_trajectories)>0:
            agent.update_policy(elite_trajectories)

    total_exp = 1000
    rewards = []
    for _ in tqdm.tqdm(range(total_exp)):
        t = get_trajectory(env, agent, trajectory_len)
        rewards.append(t['total_reward'])
    print(np.mean(rewards))
