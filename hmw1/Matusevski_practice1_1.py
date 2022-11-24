import os
import gym
import gym_maze
import numpy as np
import random
import time
import tqdm
from matplotlib import pyplot as plt

def get_trajectory(agent, trajectory_len):
    trajectory = {
        'states':[], 
        'actions':[],
        'total_reward': 0}
    state = env.reset()
    trajectory['states'].append(state)
    for _ in range(trajectory_len):
        action = agent.get_action(state)
        trajectory['actions'].append(action)
        state, reward, done, _ = env.step(action)
        trajectory['states'].append(state)
        trajectory['total_reward'] += reward
        if done:
            break
    return trajectory

def get_elite_trajectories(trajectories, q_param):
    total_rewards = [trajectory['total_reward'] for trajectory in trajectories]
    quantile = np.quantile(total_rewards, q=q_param) 
    return [trajectory for trajectory in trajectories if trajectory['total_reward'] > quantile]



class CrossEntropyMethod():
    def __init__(self, name, state_n, action_n):
        self.name = name
        self.state_n = state_n
        self.action_n = action_n
        self.policy = np.ones((self.state_n, self.action_n)) / self.action_n
    
    def get_action(self, state):
        return int(np.random.choice(np.arange(self.action_n), p=self.policy[state]))
    
    def update_policy(self, elite_trajectories):
        pre_policy = np.zeros((self.state_n, self.action_n))
        
        # counter
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                pre_policy[state][action] += 1
        
        for state in range(self.state_n):
            if sum(pre_policy[state]) == 0:
                self.policy[state] = np.ones(self.action_n) / self.action_n
            else:
                self.policy[state] = pre_policy[state] / sum(pre_policy[state])
        self.save_policy()
        return None
    def laplace_update_policy(self, elite_trajectories, lambd):
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
                self.policy[state] = (pre_policy[state]+lambd) / (sum(pre_policy[state])+lambd*self.action_n)
        self.save_policy()
        return None
    def policy_smoothing_update(self, elite_trajectories, lambd):
        pre_policy = np.zeros((self.state_n, self.action_n))
        
        # counter
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                pre_policy[state][action] += 1
        
        # 
        for state in range(self.state_n):
            if sum(pre_policy[state]) == 0:
                new_policy = np.ones(self.action_n) / self.action_n
            else:
                new_policy = pre_policy[state] / sum(pre_policy[state])
            self.policy[state] = lambd*new_policy+ (1-lambd)*self.policy[state]
        self.save_policy()
        return None
    def save_policy(self):
        np.save(f'{env.spec.id}-{self.name}', self.policy)
    def load_policy(self):
        fname = f'{env.spec.id}-{self.name}.npy'
        if os.path.exists(fname):
            self.policy = np.load(fname)

def delivered(state):
    taxi_row, taxi_col, passenger_location, destination = env.decode(state)
    return passenger_location == destination            

              
action_n       = 6
state_n        = 500 

episode_n      = 50
trajectory_len = 200
trajectory_n   = 160
q_param        = 0.2



name = f'cem_{episode_n}_{trajectory_len}_{trajectory_n}_{q_param}_0.99'
agent = CrossEntropyMethod(name, state_n, action_n)
env = gym.make("Taxi-v3")
episode_data = []
for _ in tqdm.tqdm(range(episode_n), desc='Episodes '+str(episode_n)+' of '+str(trajectory_n)+' trajectories'):
    trajectories = [get_trajectory(agent, trajectory_len) for _ in range(trajectory_n)]

    mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])

    elite_trajectories = get_elite_trajectories(trajectories, q_param)
    episode_data.append((mean_total_reward, len(elite_trajectories)))
    if len(elite_trajectories) > 0:
        agent.update_policy(elite_trajectories)

# agent2 = CrossEntropyMethod('agent2', state_n, action_n)
# agent2.policy = np.load('./Taxi-v3-policy_10_200_5120_0.9_0.5.npy')

#test
env = gym.make("Taxi-v3")

success = 0
total_exp = 100
for i in tqdm.tqdm(range(total_exp), desc='Test '+str(total_exp)+' runs'):
    state = env.reset()
    # env.render()
    # time.sleep(1)
    taxi_row, taxi_col, passenger_location, destination = env.decode(state)

    test_reward = 0
    for step in range(trajectory_len):
        action = agent.get_action(state)
        state, reward, done, _ = env.step(action)
        test_reward += reward
        # env.render()
        if done:
            break
        # time.sleep(0.125)
    if delivered(state):
        success +=1
    # time.sleep(2)
    taxi_row, taxi_col, passenger_location, f_destination = (env.decode(state))

print('success ', success/total_exp)