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

PREF = '_mcc'
TASKS_DIR = './tasks' + PREF + '/'
EXP_DIR = './experiments' + PREF + '/'

state_dim = 2
action_n = 2
gym_name = 'MountainCarContinuous-v0'

env = gym.make(gym_name)

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
                nn.Linear(layers_n[0], self.action_n),
            )
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
    
    def forward(self, _input):
        return self.network(_input)
    
    def get_action(self, state):
        state = torch.FloatTensor(state)
        m, sigma = self.network(state)
        sigma = torch.exp(sigma)
        actions = torch.distributions.Normal(m, sigma)
        action = actions.sample(sample_shape=torch.Size([1]))
        action = self.tanh(action)
        return action.detach().numpy()

    def update_policy(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for trajectory in elite_trajectories:
            elite_states.extend(trajectory['states'])
            elite_actions.extend(trajectory['actions'])
        assert len(elite_states)==len(elite_actions), f'{len(elite_states)},{len(elite_actions)}'

        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.FloatTensor(elite_actions)
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
        self.load_policy_from_file(self.get_name())

    def load_policy_from_file(self, fname):
        if os.path.exists(self.get_name()):
            self.load_state_dict(torch.load(self.get_name()))
            print('policy loaded', fname)
        else:
            print("policy not found", fname)

    def load_n_policy(self, n):
        files = list(glob.glob(os.path.join(EXP_DIR, self.name,str(n)+'_*.nn')))
        print('policies count', len(files))
        if len(files) > 0:
            file = files[0]
            self.load_policy_from_file(files[0])
    def load_last_policy(self):
        files = list(glob.glob(os.path.join(EXP_DIR, self.name,'*.nn')))
        print('policies count', len(files))
        if len(files) > 0:
            file = list(sorted(files, key=lambda x: int(x.split('/')[-1].split('_')[0])))[-1]
            self.load_policy_from_file(files[0])
        else:
            print("policy files not found")

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

def run_experiment(eid, versions, layers_n, episode_n, trajectory_len, trajectory_n, q_param, lr):
    state_dim = 2
    action_n = 2

    agent = CrossEntropyMethod(eid, state_dim, action_n, layers_n, lr)

    exp = {
        'id': eid,
        'lr':lr,
        'version':versions,
        'layers_n': layers_n,
        'episode_n':episode_n,
        'trajectory_len':trajectory_len,
        'trajectory_n':trajectory_n,
        'q_param':q_param,
        'total_elapsed':0,
        'finished':False,
        'episode_data':[]
    }
    episode_data_path = os.path.join(EXP_DIR, f'{eid}.json')
    mean_total_rewards = []
    episode_data = []
    start =  datetime.datetime.now()
    json.dump(exp, open(episode_data_path,'w'))

    # env = gym.make('CartPole-v1')
    env = gym.make(gym_name)
    os.makedirs(os.path.join(EXP_DIR, eid))
    for i in tqdm.tqdm(range(episode_n)):
        trajectories = [get_trajectory(env, agent, trajectory_len) for _ in range(trajectory_n)]

        mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
        mean_total_rewards.append(mean_total_reward)
        
        elite_trajectories = get_elite_trajectories(trajectories, q_param)
        episode_data.append((mean_total_reward, len(elite_trajectories)))

        if len(elite_trajectories)>0:
            agent.update_policy(elite_trajectories)
            torch.save(agent.state_dict(), self.get_name())

        exp['episode_data'] = episode_data
        exp['total_elapsed'] = (datetime.datetime.now() - start).total_seconds()
        json.dump(exp, open(episode_data_path,'w'))
    exp['episode_data'] = episode_data
    exp['finished'] = True
    exp['total_elapsed'] = (datetime.datetime.now() - start).total_seconds()
    
    json.dump(exp, open(episode_data_path,'w'))

def executor(tasks_dir, experiments_dir):
    tasks = list(glob.glob(os.path.join(tasks_dir, '*.json')))
    for task in tasks:
        task = json.load(open(task, 'r'))
        eid = task['id']
        try:
            f = open(os.path.join(experiments_dir, eid+'.json'),'x')
            f.close()
            print(eid)
            print(task)
            lr             = task['lr']
            versions       = task['version']
            layers_n       = task['layers_n']
            episode_n      = task['episode_n']
            trajectory_len = task['trajectory_len']
            trajectory_n   = task['trajectory_n']
            q_param        = task['q_param']

            run_experiment(eid, versions, layers_n, episode_n, trajectory_len, trajectory_n, q_param, lr)

        except FileExistsError as err:
            pass


def replay(eid, episodes = 100, n=-1, get_stats=False):

    ed = json.load(open(os.path.join(EXP_DIR, eid+".json")))

    lr = ed['lr']
    layers_n = ed['layers_n']
    episode_n = ed['episode_n']
    trajectory_len = 100000 # ed['trajectory_len']
    trajectory_n = ed['trajectory_n']
    q_param = ed['q_param']
    total_elapsed = ed['total_elapsed']
    print(f'lr={lr}\nlayers_n={layers_n} \nepisode_n={episode_n}\ntrajectory_len={trajectory_len} \ntrajectory_n={trajectory_n} \nq_param={q_param} \ntotal_elapsed={total_elapsed}')
    agent = CrossEntropyMethod(eid, state_dim, action_n, layers_n=layers_n, lr=lr)
    if n==-1:
        agent.load_last_policy()
    else:
        agent.load_n_policy(n)

    if get_stats:
        tjs = []
        for _ in tqdm.tqdm(range(episodes)):
            t = get_trajectory(env, agent, trajectory_len, visualize=False)
            tjs.append(t['total_reward'])
        print(np.mean(tjs))
        plt.hist(tjs, bins=50)
        plt.show()
    else:
        tjs = []
        while True:
            t = get_trajectory(env, agent, trajectory_len, visualize=True)
            tjs.append(t['total_reward'])
            print(tjs[-1])

if __name__ == "__main__":
    print(gym_name)
#     replay(eid = '4400486824641',n=6)
#     replay(eid = '65968560182', n = 166)
#     replay(eid = '68796186114', n = 158)
#     replay(eid = '68866188997', n = 125)
#     replay(eid = '65968560182', n = 166)
#     replay(eid = '68866188997', n=124)
    replay(eid = '78998331192', n=125, get_stats=False, episodes=500)
#      399
#     68837581774

