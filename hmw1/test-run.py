import os
import gym
import gym_maze
import numpy as np
import random
import time
import tqdm
from matplotlib import pyplot as plt


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
        
        # 
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
            
action_n = 6
state_n = 500
trajectory_len = 30
agent2 = CrossEntropyMethod('agent2', state_n, action_n)
# agent2.policy = np.load('./Taxi-v3-laplase_100_200_160_0.9_0.99.npy')
# agent2.policy = np.load('./Taxi-v3-laplase_100_200_320_0.9_0.9.npy')
# agent2.policy = np.load('./Taxi-v3-laplase_100_200_1280_0.9_0.5.npy')
# agent2.policy = np.load('./Taxi-v3-policy_100_200_1280_0.9_0.5.npy')
# agent2.policy = np.load('./Taxi-v3-policy_100_200_5120_0.9_0.1.npy')
# agent2.policy = np.load('./Taxi-v3-policy_10_200_5120_0.9_0.5.npy')
agent2.policy = np.load('./Taxi-v3-det_CEM_100_30_20_0.5_0.1.npy')

#test
env = gym.make("Taxi-v3")

pl = []
r = []
deliv = 0
# while True:
for _ in range(1000):
    final_rewards = []
    state = env.reset()
#     env.render()
#     time.sleep(1)
    taxi_row, taxi_col, passenger_location, destination = env.decode(state)
#     print(taxi_row, taxi_col, passenger_location, destination)
    test_reward = 0
    for i, step in tqdm.tqdm(enumerate(range(trajectory_len))):
        action = agent2.get_action(state)
        state, reward, done, _ = env.step(action)
        test_reward += reward
#         env.render()
        if done:
    #             print(state,f'{test_reward:04}', actions[action])
    #         зкште
            break
#         time.sleep(0.125)
    pl.append(i)
    r.append(test_reward)
    if delivered(state):
        print('Success')
        deliv += 1
    else:
        print('Fail')
#     time.sleep(2)
    taxi_row, taxi_col, passenger_location, f_destination = (env.decode(state))
#     print(taxi_row, taxi_col, passenger_location, destination, action)
    final_rewards.append([test_reward, action, step])
    #     print('Test Reward', test_reward)
print(deliv/1000)
print(np.mean(pl))
print(np.mean(r))