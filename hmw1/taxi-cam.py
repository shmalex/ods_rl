import os
import gym
import numpy as np
import random
import time
import tqdm


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
    def save_policy(self):
        np.save(f'{env.spec.id}-{self.name}', self.policy)
    def load_policy(self):
        fname = f'{env.spec.id}-{self.name}.npy'
        if os.path.exists(fname):
            self.policy = np.load(fname)


def get_trajectory(agent, trajectory_len):
    trajectory = {
        'states':[], 
        'actions':[],
        'total_reward': 0}

    state = env.reset()
    #     state = get_state(obs)
    trajectory['states'].append(state)
    
    
    for _ in range(trajectory_len):
        action = agent.get_action(state)
#         print(action)
        trajectory['actions'].append(action)
        
        state, reward, done, _ = env.step(action)
        trajectory['states'].append(state)
        trajectory['total_reward'] += reward
#         env.render()
#         time.sleep(0.1)
        if done:
#             print(done)
            break
    
    return trajectory

def get_elite_trajectories(trajectories, q_param):
    total_rewards = [trajectory['total_reward'] for trajectory in trajectories]
    quantile = np.quantile(total_rewards, q=q_param) 
    return [trajectory for trajectory in trajectories if trajectory['total_reward'] > quantile]


actions = {0: 'move south',
1: 'move north',
2: 'move east',
3: 'move west',
4: 'pickup passenger',
5: 'drop off passenger'}
    
action_n = 6
state_n = 500


episode_n = 500
trajectory_len = 10000
trajectory_n =  100
q_param = 0.9

name = '500x500'

agent = CrossEntropyMethod(name, state_n, action_n)
env = gym.make("Taxi-v3")
episode_data = []
for _ in tqdm.tqdm(range(episode_n)):
    trajectories = [get_trajectory(agent, trajectory_len) for _ in range(trajectory_n)]
    
    mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
    
    elite_trajectories = get_elite_trajectories(trajectories, q_param)

    episode_data.append((mean_total_reward, len(elite_trajectories)))
    print(mean_total_reward, len(elite_trajectories))
    
    if len(elite_trajectories) > 0:
        agent.update_policy(elite_trajectories)
    np.save('eposode_data'+name, np.array(episode_data))