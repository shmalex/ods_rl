
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

from multiprocessing import Process, JoinableQueue
import warnings
warnings.filterwarnings("ignore")

import Matusevski_practice4_1 as pr1
import Matusevski_practice4_2 as pr2


NUMBER_WORKER_THREADS = 18
PREF = ''
TASKS_DIR = './tasks' + PREF + '/'
EXP_DIR = './experiments' + PREF + '/'

os.makedirs(TASKS_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)

state_dim = 500
action_n = 6

gym_name = 'Taxi-v3'



def start_threads_get_trajectories(num_worker_procs, agent, trajectory_n, trajectory_len):
    task_q = JoinableQueue()
    out_q = JoinableQueue()

    for i in range(trajectory_n): task_q.put_nowait(i)
    for index in range(num_worker_procs): task_q.put_nowait(-1)
    procs = []
    for index in range(num_worker_procs):
        proc = Process(target=worker_get_trajectory, args=(task_q, out_q, agent, trajectory_len))
        procs.append(proc)

    for p in procs: p.start()
    ret = []
    while len(ret)!=trajectory_n:
        item = out_q.get()
        if item == None: continue
        ret.append(item)
        out_q.task_done()

    task_q.join()
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
        # t = tqdm.tqdm(desc='', total=episode_n, )
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

def worker_get_trajectory(task_queue, out_q):
    while True:
        if task_queue.empty(): break
        task = task_queue.get()
        if task == -1: break
        run_experiment(task)
        out_q.put(0)
        task_queue.task_done()
    task_queue.task_done()


def run_experiment(task):
    eid =  task['id']
    exp = {
        'id': eid,
        'name': task['name'],
        'version': task['version'],
        'params': task['params'],
        'total_elapsed': 0,
        'finished': False,
        'episode_data': []
    }

    # os.makedirs(os.path.join(EXP_DIR, eid))
    episode_data_path = os.path.join(EXP_DIR, f'{eid}.json')
    json.dump(exp, open(episode_data_path, 'w'))

    exps = {
        "mc":pr1.MC,
        "sarsa":pr1.SARSA,
        "ql":pr1.QLearning,
        "mc2":pr2.MC,
        "sarsa2":pr2.SARSA,
        "ql2":pr2.QLearning,
    }
    exp_func = exps[task['name']]
    start = datetime.datetime.now()
    rewards = exp_func(**task['params'])

    exp['episode_data'] = rewards
    exp['finished'] = True
    exp['total_elapsed'] = (datetime.datetime.now() - start).total_seconds()

    json.dump(exp, open(episode_data_path, 'w'))

def start_threads_get_trajectories(num_worker_procs, name, params):
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
    for p in procs: p.join()

    return ret

def executor(num_worker_procs, tasks_dir, experiments_dir, iscontinue=False):
    tasks = list(glob.glob(os.path.join(tasks_dir, '*.json')))
    task_q = JoinableQueue()
    tasks_to_execute = 0
    out_q = JoinableQueue()

    if len(tasks)==0:
        print('There is no tasks')
    print('.')
    print(len(tasks))
    np.random.shuffle(tasks)
    for task in tasks:
        task = json.load(open(task, 'r'))
        eid = task['id']
        if os.path.exists(os.path.join(experiments_dir, eid+'.json')): continue
        task_q.put_nowait(task)
        tasks_to_execute += 1
    for _ in range(num_worker_procs): task_q.put_nowait(-1)

    procs = [Process(target=worker_get_trajectory, args=(task_q, out_q)) for _ in range(num_worker_procs)]
    for p in procs: p.start()

    ret = []
    progress = tqdm.tqdm(total=tasks_to_execute)
    while len(ret) != tasks_to_execute:
        item = out_q.get()
        
        if item == None: continue
        progress.update()
        ret.append(item)
        out_q.task_done()

    task_q.join()
    for p in procs: p.join()

    return ret


if __name__ == "__main__":
    executor(NUMBER_WORKER_THREADS, TASKS_DIR, EXP_DIR, iscontinue=True)