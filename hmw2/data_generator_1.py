
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput


import gym
import datetime
import os
import json
import glob
import tqdm
import matplotlib.pyplot as plt
from torch import nn
import torch
import numpy as np

import threading
import time
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")


NUMBER_WORKER_THREADS = 20
PREF = '_slider'
TASKS_DIR = './tasks' + PREF + '/'
EXP_DIR = './experiments' + PREF + '/'

os.makedirs(TASKS_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)


state_dim = 8
action_n = 4

gym_name = 'LunarLander-v2'


class CrossEntropyMethod(nn.Module):
    def __init__(self, name, state_dim, action_n, layers_n, lr=0.01):
        super().__init__()
        self.name = name
        self.state_dim = state_dim
        self.action_n = action_n
        self.layers_n = layers_n
        self.lr = lr

        if len(layers_n) == 1:
            self.network = nn.Sequential(
                nn.Linear(self.state_dim, layers_n[0]),
                nn.ReLU(),
                nn.Linear(layers_n[0], self.action_n)
            )
        if len(layers_n) == 2:
            self.network = nn.Sequential(
                nn.Linear(self.state_dim, layers_n[0]),
                nn.ReLU(),
                nn.Linear(layers_n[0], layers_n[1]),
                nn.ReLU(),
                nn.Linear(layers_n[1], self.action_n)
            )
        if len(layers_n) == 3:
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
        state = torch.FloatTensor(state)
        logits = self.network(state)
        action_prob = self.softmax(logits).detach().numpy()
        action = np.random.choice(self.action_n, p=action_prob)
        return action

    def update_policy(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for trajectory in elite_trajectories:
            elite_states.extend(trajectory['states'])
            elite_actions.extend(trajectory['actions'])
        assert len(elite_states) == len(elite_actions), f'{len(elite_states)},{len(elite_actions)}'

        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.LongTensor(elite_actions)
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
        files = list(glob.glob(os.path.join(
            EXP_DIR, self.name, str(n)+'_*.nn')))
        print('policies count', len(files))
        if len(files) > 0:
            self.load_policy_from_file(files[0])

    def load_last_policy(self):
        files = list(glob.glob(os.path.join(EXP_DIR, self.name, '*.nn')))
        print('policies count', len(files))
        if len(files) > 0:
            file = list(sorted(files, key=lambda x: int(
                x.split('/')[-1].split('_')[0])))[-1]
            self.load_policy_from_file(files)
        else:
            print("policy files not found")

def get_trajectory(agent, trajectory_len, visualize=False):
    trajectory = {
        'states': [],
        'actions': [],
        'total_reward': 0}
    env = gym.make(gym_name)
    state = env.reset()
    for _ in range(trajectory_len):
        action = agent.get_action(state)
        # if _ < 10: 
        #     if _%2==0:
        #         action = 1
            # else: action=2
        # else:
            # if _%7==0: action=3
        trajectory['states'].append(state)
        trajectory['actions'].append(action)
        state, reward, done, _ = env.step(action)
        # if reward == -100:
        #     print('CRash', state, reward, done, _)
        # if reward == 200:
        #     print(state, reward, done, _)
        # x,y, dx, dy, a, da, tl, tr = state
        # print(x,y, dx, dy, a, da, tl, tr)
        mx = 0
        # if tl and tr:
            # if np.abs(dx)>0.01: mx=1000*np.abs(dx)
        trajectory['total_reward'] += reward+mx
        if done:
            break

        if visualize:
            env.render()
    env.close()
    # assert len(trajectory['actions']) == len(
        # trajectory['states']), f"gt {len(trajectory['actions'])},{len(trajectory['states'])}"
    return trajectory

def get_elite_trajectories(trajectories, q_param):
    total_rewards = [trajectory['total_reward'] for trajectory in trajectories]
    quantile = np.quantile(total_rewards, q=q_param)
    return [trajectory for trajectory in trajectories if trajectory['total_reward'] > quantile]


def worker_get_trajectory(task_queue, out_q, agent, trajectory_len):
    while True:
        if task_queue.empty(): break
        _ = task_queue.get()
        if _== -1: break
        traj = get_trajectory(agent, trajectory_len,  visualize=False)
        out_q.put(traj)
        task_queue.task_done()
    task_queue.task_done()

def start_threads_get_trajectories(num_worker_procs, agent, trajectory_n, trajectory_len):
    task_q = mp.JoinableQueue()
    out_q = mp.JoinableQueue()

    for i in range(trajectory_n): task_q.put_nowait(i)
    for index in range(num_worker_procs): task_q.put_nowait(-1)
    procs = []
    for index in range(num_worker_procs):
        proc = mp.Process(target=worker_get_trajectory, args=(task_q, out_q, agent, trajectory_len))
        procs.append(proc)

    for p in procs: p.start()
    ret = []
    while len(ret)!=trajectory_n:
        item = out_q.get()
        if item == None: continue
        ret.append(item)
        out_q.task_done()

    task_q.join()
    # print('process join')
    for p in procs: p.join()

    return ret


def continue_experiment(eid, versions, layers_n, episode_n, trajectory_len, trajectory_n, q_param, lr):
    os.makedirs(os.path.join(EXP_DIR, eid), exist_ok=True)

    agent = CrossEntropyMethod(eid, state_dim, action_n, layers_n, lr)

    # load last restore point
    files = list(glob.glob(os.path.join(EXP_DIR, eid, '*.nn')))
    start_from_start = len(files) == 0
    if len(files) > 0:
        file = list(sorted(files, key=lambda x: int(x.split('/')[-1].split('_')[0])))[-1]
        print(file)
        agent.load_state_dict(torch.load(file))
        # if there is no restore point - start from the start

    # load the experiment data
    episode_data_path = os.path.join(EXP_DIR, f'{eid}.json')
    if os.path.exists(episode_data_path) and (not start_from_start):
        exp = json.load(open(episode_data_path, 'r'))
        episode_data = exp['episode_data']
        start_episode = len(exp['episode_data'])
        start_time = datetime.datetime.now() - datetime.timedelta(seconds=exp['total_elapsed'])
    else:
        exp = {
            'id': eid,
            'lr': lr,
            'version': versions,
            'layers_n': layers_n,
            'episode_n': episode_n,
            'trajectory_len': trajectory_len,
            'trajectory_n': trajectory_n,
            'q_param': q_param,
            'total_elapsed': 0,
            'finished': False,
            'episode_data': []
        }
        episode_data = []
        start_episode = 0

        start_time = datetime.datetime.now()

    exp['inprogress'] = True
    exp['finished'] = False
    json.dump(exp, open(episode_data_path, 'w'))

    num_worker_threads = NUMBER_WORKER_THREADS
    print(start_episode, episode_n)

    try:
        t = tqdm.tqdm(desc='', total=episode_n, )
        for i in range(episode_n):            # trajectories = [get_trajectory(agent, trajectory_len) for _ in range(trajectory_n)]
            trajectories = start_threads_get_trajectories(num_worker_threads, agent, trajectory_n, trajectory_len)

            mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
            t.update()
            t.set_description(f"Mr{mean_total_reward:0.4}")
            elite_trajectories = get_elite_trajectories(trajectories, q_param)
            episode_data.append((mean_total_reward, len(elite_trajectories)))

            if len(elite_trajectories) > 0:
                agent.update_policy(elite_trajectories)
                torch.save(agent.state_dict(), os.path.join(EXP_DIR, eid, f'{i}_{eid}.nn'))

            exp['episode_data'] = episode_data
            exp['total_elapsed'] = (datetime.datetime.now() - start_time).total_seconds()
            json.dump(exp, open(episode_data_path, 'w'))
    except KeyboardInterrupt:
        exp['inprogress'] = False
        json.dump(exp, open(episode_data_path, 'w'))
        return
    exp['episode_data'] = episode_data
    exp['finished'] = True
    exp['inprogress'] = False
    exp['total_elapsed'] = (datetime.datetime.now() -
                            start_time).total_seconds()

    json.dump(exp, open(episode_data_path, 'w'))


def run_experiment(eid, versions, layers_n, episode_n, trajectory_len, trajectory_n, q_param, lr):
    agent = CrossEntropyMethod(eid, state_dim, action_n, layers_n, lr)
    exp = {
        'id': eid,
        'lr': lr,
        'version': versions,
        'layers_n': layers_n,
        'episode_n': episode_n,
        'trajectory_len': trajectory_len,
        'trajectory_n': trajectory_n,
        'q_param': q_param,
        'total_elapsed': 0,
        'finished': False,
        'episode_data': []
    }
    os.makedirs(os.path.join(EXP_DIR, eid))
    episode_data_path = os.path.join(EXP_DIR, f'{eid}.json')
    mean_total_rewards = []
    episode_data = []
    start = datetime.datetime.now()
    json.dump(exp, open(episode_data_path, 'w'))

    num_worker_threads = 22
    t = tqdm.tqdm(desc='', total=episode_n, )
    for i in range(episode_n):
        #         trajectories = [get_trajectory(env, agent, trajectory_len) for _ in range(trajectory_n)]
        trajectories = start_threads_get_trajectories(num_worker_threads, agent, trajectory_n, trajectory_len)

        mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
        mean_total_rewards.append(mean_total_reward)
        t.update()
        t.set_description(f"Mr{mean_total_reward}")
        elite_trajectories = get_elite_trajectories(trajectories, q_param)
        episode_data.append((mean_total_reward, len(elite_trajectories)))

        if len(elite_trajectories) > 0:
            agent.update_policy(elite_trajectories)
            torch.save(agent.state_dict(), os.path.join(EXP_DIR, eid, f'{i}_{eid}.nn'))

        exp['episode_data'] = episode_data
        exp['total_elapsed'] = (datetime.datetime.now() - start).total_seconds()
        json.dump(exp, open(episode_data_path, 'w'))
    exp['episode_data'] = episode_data
    exp['finished'] = True
    exp['total_elapsed'] = (datetime.datetime.now() - start).total_seconds()

    json.dump(exp, open(episode_data_path, 'w'))


def executor(tasks_dir, experiments_dir, iscontinue=False):
    tasks = list(glob.glob(os.path.join(tasks_dir, '*.json')))
    if len(tasks)==0:
        print('there is no tasks')
    print('.')
    print(tasks)
    for task in tasks:
        task = json.load(open(task, 'r'))
        eid = task['id']
        try:
            episode_data_path = os.path.join(experiments_dir, f'{eid}.json')
            if not iscontinue:
                f = open(os.path.join(experiments_dir, eid+'.json'), 'x')
                f.close()
            else:
                if os.path.exists(episode_data_path):
                    try:
                        exp = json.load(open(episode_data_path, 'r'))
                    except json.decoder.JSONDecodeError as ex:
                        print(ex)
                        print(episode_data_path)
                    if 'finished' in exp:
                        if exp['finished']:
                            print('Cotinue Mode: task '+eid+' Finished')
                            continue
                    if 'inprogress' in exp:
                        if exp['inprogress']:
                            print('Cotinue Mode: task '+eid+' In Progress')
                            continue
            print(eid)
            print(task)
            lr = task['lr']
            versions = task['version']
            layers_n = task['layers_n']
            episode_n = task['episode_n']
            trajectory_len = task['trajectory_len']
            trajectory_n = task['trajectory_n']
            q_param = task['q_param']

            # if not(versions in {'5', '6'}): continue
            if iscontinue:
                continue_experiment(eid, versions, layers_n, episode_n,
                                    trajectory_len, trajectory_n, q_param, lr)
            else:
                run_experiment(eid, versions, layers_n, episode_n, trajectory_len, trajectory_n, q_param, lr)
                exit()
        except FileExistsError as err:
            print('task '+eid+' allready done')
            pass

# def main():
    # executor(TASKS_DIR, EXP_DIR, iscontinue=True)
if __name__ == "__main__":
    executor(TASKS_DIR, EXP_DIR, iscontinue=True)
    # with PyCallGraph(output=GraphvizOutput()):
        # main()
    # import cProfile
    # outdat = 'output.dat'
    # cProfile.run('main()', outdat)


    # import pstats
    # from pstats import SortKey

    # with open('output_time.txt', 'w') as f:
    #     p = pstats.Stats(outdat, stream=f)
    #     p.sort_stats('time').print_stats()

    # with open('output_calls.txt', 'w') as f:
    #     p = pstats.Stats(outdat, stream=f)
    #     p.sort_stats('calls').print_stats()