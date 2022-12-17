# https://ilikeinterfaces.com/
# https://freesound.org/people/SamsterBirdies/sounds/467882/
# https://freesound.org/people/unfa/sounds/154894/
# https://freesound.org/people/ProjectsU012/sounds/340946/

import matplotlib.cm as cm
from  matplotlib.colors import Normalize

import PIL
from PIL import Image, ImageDraw, ImageFont
import gym
import datetime
import os
import json
import glob
import time
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch
import warnings
warnings.filterwarnings("ignore")


PREF = ''
TASKS_DIR = './tasks' + PREF + '/'
EXP_DIR = './experiments' + PREF + '/'

state_dim = 8
action_n = 4

gym_name = 'LunarLander-v2'


env = gym.make(gym_name)
fontcontrol = ImageFont.truetype(r'./PressStart2P-Regular.ttf', 20) 
fontspeed = ImageFont.truetype(r'./PressStart2P-Regular.ttf', 14) 
fontabort = ImageFont.truetype(r'./PressStart2P-Regular.ttf', 36)
fontscore = ImageFont.truetype(r'./PressStart2P-Regular.ttf', 12) 
fontnn = ImageFont.truetype(r'./PressStart2P-Regular.ttf', 6)

norms1 = Normalize(vmin = 0, vmax=1)
norms15 = Normalize(vmin = -1.5, vmax=1.5)
normsPi = Normalize(vmin = -3.1415927, vmax=3.1415927)
norms5 = Normalize(vmin = -5, vmax=5)
normsNN = Normalize(vmin = -5, vmax=5)

cmap = cm.seismic


m1 = cm.ScalarMappable(norm=norms1, cmap=cmap)
m15 = cm.ScalarMappable(norm=norms15, cmap=cmap)
mPi = cm.ScalarMappable(norm=normsPi, cmap=cmap)
m5 = cm.ScalarMappable(norm=norms5, cmap=cmap)
mNN = cm.ScalarMappable(norm=normsNN, cmap=cm.seismic)
dxx = 291.38560115900106
dyy = -212.9938306336957
is_recording = False

def minmax(l0_in):
    return l0_in.min(), l0_in.max()

class SaveOutput:
    def __init__(self):
        self.output = []
        self.images = []
        self.is_print = False
        self.l0_inmm_ar = []
        self.l0_outmm_ar = []
        self.r0_outmm_ar = []
        self.l1_outmm_ar = []
        self.r1_outmm_ar = []
        self.l2_outmm_ar = []
        self.l0_inmm  = (-1.7616844, 1.9726161)
        self.l0_outmm = (-1.3662803, 1.5597924)
        self.r0_outmm = (0.0, 1.5597924)
        self.l1_outmm = (-0.5895511, 1.29126)
        self.r1_outmm = (0.0, 1.29126)
        self.l2_outmm = (-0.5923617, 0.89912146)
    def __call__(self, module, module_in, module_out):
        self.output.append((module, module_in, module_out))
#         print(len(self.output))
    def reset(self):
        self.output = []
        self.image = []
    def clear(self):
        self.output = []
    def updateMinMax(self):
        self.l0_inmm = np.min(self.l0_inmm_ar), np.max(self.l0_inmm_ar)
        self.l0_outmm = np.min(self.l0_outmm_ar), np.max(self.l0_outmm_ar)
        self.r0_outmm = np.min(self.r0_outmm_ar), np.max(self.r0_outmm_ar)
        self.l1_outmm = np.min(self.l1_outmm_ar), np.max(self.l1_outmm_ar)
        self.r1_outmm = np.min(self.r1_outmm_ar), np.max(self.r1_outmm_ar)
        self.l2_outmm = np.min(self.l2_outmm_ar), np.max(self.l2_outmm_ar)
        self.l0_inmm_ar = []
        self.l0_outmm_ar = []
        self.r0_outmm_ar = []
        self.l1_outmm_ar = []
        self.r1_outmm_ar = []
        self.l2_outmm_ar = []

    def get_min_max(self):
        return [self.l0_inmm,self.l0_outmm,self.r0_outmm,self.l1_outmm,self.r1_outmm,self.l2_outmm]
    def push(self):
        l0_in = self.output[0][1][0].detach().numpy()
        self.l0_inmm_ar.append(minmax(l0_in))
        l0_out = self.output[0][2].detach().numpy()
        self.l0_outmm_ar.append(minmax(l0_out))
        r0_out = self.output[1][2].detach().numpy()
        self.r0_outmm_ar.append(minmax(r0_out))
        l1_out = self.output[2][2].detach().numpy()
        self.l1_outmm_ar.append(minmax(l1_out))
        r1_out = self.output[3][2].detach().numpy()
        self.r1_outmm_ar.append(minmax(r1_out))
        l2_out = self.output[4][2].detach().numpy()
        self.l2_outmm_ar.append(minmax(l2_out))
        self.images.append((l0_in, l0_out, r0_out, l1_out, r1_out, l2_out))
        self.clear()
class CrossEntropyMethod(nn.Module):
    def __init__(self, name, state_dim, action_n, layers_n, lr=0.01):
        super().__init__()
        self.name = name
        self.state_dim = state_dim
        self.action_n = action_n
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
        assert len(elite_states) == len(
            elite_actions), f'{len(elite_states)},{len(elite_actions)}'

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
        if os.path.exists(fname):
            self.load_state_dict(torch.load(fname))
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
            self.load_policy_from_file(file)
        else:
            print("policy files not found")

frame_counter = 0
score_total = 0
score_alive = 0
score_crash = 0
generation = 0
def render_image(env, state, reward, done, action, crashed, success, saveOutput, no_messages=False):
    global frame_counter, score_total, score_alive, score_crash
    
    x,y, dx, dy, a, da, tr, tl = state
    # x,y, dx, dy, a, da, tl, tr
    img = env.render(mode='rgb_array')
    # plt.imshow(img)
    def Y(y):
        return y*(dyy)+300
    def X(x):
        return x*(dxx*1.025)+300

    im = PIL.Image.frombytes(data=img, size=(600, 400), mode='RGB')

    speed = np.sqrt((dx**2+ dy**2))*50
    # print(speed)
    draw = ImageDraw.Draw(im)
    draw.arc([(X(x)-2, Y(y)-2-20),(X(x)+2,Y(y)+2-20)],0,360,fill='red')
    draw.line((X(x), Y(y)-20,X(x)      ,Y(y)-40*dy-20),fill='blue', width=4)
    draw.line((X(x), Y(y)-20,X(x)+40*dx,Y(y)-20      ),fill='green', width=4)
    draw.line((X(x), Y(y)-20,X(x)+40*dx,Y(y)-40*dy-20),fill='red', width=6)
    


    if a > 0:
        draw.arc([(X(x)-23, Y(y)-23-20),(X(x)+23,Y(y)+23-20)],-90-(a*180/np.pi),-90,fill='red', width=3)
    else:
        draw.arc([(X(x)-23, Y(y)-23-20),(X(x)+23,Y(y)+23-20)],-90,-90-(a*180/np.pi),fill='red', width=3)

    textl     = 'L'
    textd     = '  D'
    textr     = '    R'
    texttl    = '-'
    texttr    = '    -'
    texttlt   = '='
    texttrt   = '    ='
    textabort = 'ABORT MISSION' 
    textcongr = 'CONGRATULATIONS!'
    # drawing text size
    draw.text((5, 5), textl, font = fontcontrol, align ="left", fill='red' if action==3 else 'white')
    draw.text((5, 5), textd, font = fontcontrol, align ="left", fill='red' if action==2 else 'white')
    draw.text((5, 5), textr, font = fontcontrol, align ="left", fill='red' if action==1 else 'white')

    draw.text((455, 15), f"#{score_total:0>3}", font=fontspeed, align ="right")
    draw.text((455, 30), f"{score_alive}", font=fontscore, align ="right", fill='green')
    draw.text((495, 30), f"{score_crash}", font=fontscore, align ="left", fill='red')

    draw.text((525, 15), f"{speed}"[:4], font=fontspeed, align ="right")
    # draw.text((455, 15), f"{-(a*180/np.pi)}"[:4], font=fontspeed, align ="right")


    if tl: draw.text((5, 27), texttlt, font=fontcontrol, align ="left", fill='green') 
    else: draw.text((5, 27), texttl, font=fontcontrol, align ="left") 
    if tr: draw.text((5, 27), texttrt, font=fontcontrol, align ="left", fill='green') 
    else: draw.text((5, 27), texttr, font=fontcontrol, align ="left")

    draw.text((24, 82), '|', font=fontcontrol, align ="left", fill='lightgray') 
    draw.text((66, 82), '|', font=fontcontrol, align ="left", fill='lightgray') 


    draw.text((7, 65), 'in', font=fontnn, align ="left", fill='white') 
    draw.text((117, 100), 'out', font=fontnn, align ="left", fill='white') 
    states = [x,     y, dx, dy,  a, da,tl,tr]
    transf = [m15, m15, m5, m5,mPi,mPi,m1,m1]

    # LAYERS!!!!!!!!!!!

    ys = ((25-8)/2)*3

    for jj, (sta, tran) in enumerate(zip(states, transf)):
        _r, _g, _b, _a = tran.to_rgba(sta)
        draw.arc([(10, 50+jj*4+ys),(13,53+jj*4+ys)],0,360,fill=(int(_r*255), int(_g*255), int(_b*255)), width=7)

    ls =1
    lss = 7
    soMM = saveOutput.get_min_max()
    for somm, kk in zip(soMM, saveOutput.images[-1]):

        normsNN = Normalize(vmin = somm[0], vmax=somm[1])
        mNN     = cm.ScalarMappable(norm=normsNN, cmap=cm.seismic)
        ys = 0
        if kk.shape[0]==50:
            kkk = kk.reshape(-1,25)

            for jj, val in enumerate(kkk):
                for jj, val in enumerate(val):
                    _r, _g, _b, _a = mNN.to_rgba(val)
                    draw.arc([(18+ls*lss, 50+jj*4),(21+ls*lss,53+jj*4)],0,360,
                            fill=(int(_r*255), int(_g*255), int(_b*255)), width=7)
                ls+=1
        else:
            ys = ((25-len(kk))/2)*3

            for jj, val in enumerate(kk):
                _r, _g, _b, _a = mNN.to_rgba(val)
                draw.arc([(18+ls*lss, 50+jj*4+ys),(21+ls*lss,53+jj*4+ys)],0,360,
                        fill=(int(_r*255), int(_g*255), int(_b*255)), width=7)
            ls+=1
        ls+=1
    ls+=1

    ys = ((25-5)/2)*3

    for jj, a in enumerate(range(4)):
        if jj == action:
            _r, _g, _b, _a = m1.to_rgba(1)
        else:
            _r, _g, _b, _a = m1.to_rgba(0)
        draw.arc([(10+ls*lss, 50+jj*4+ys),(13+ls*lss,53+jj*4+ys)],0,360,fill=(int(_r*255), int(_g*255), int(_b*255)), width=7)

        
    draw.arc([(100,100),(120,120)],-90, -90, fill='red',width=3)


    if done:
        if not crashed:
            draw.arc([(115, 8),(125,18)],0,360,fill='green', width=7)
            if not no_messages and frame_counter%10<=5:
                draw.text((20, 155), textcongr, font = fontabort, align ="left", fill='green') 
        else:
            draw.arc([(115, 8),(125,18)],0,360,fill='red', width=7)
            if not no_messages and frame_counter%10<=5:
                draw.text((65+np.random.normal(0, scale=2), 155+np.random.normal(0, scale=2)), textabort, font = fontabort, align ="left", fill='red') 

    draw.arc([(113, 6),(127,20)],0,360,fill='white', width=2)


    # draw.text((5, 24), texttr, font = font, align ="left") 

    # im.save(f"./images/{frame_counter:0>8}.jpg", "JPEG", quality=99, optimize=True, progressive=True)
    im.save(f"./images5/{frame_counter:0>8}.png", "PNG", quality=99, optimize=True, progressive=True)
    # im
    frame_counter+=1

    # for i in range(5):
    #     state, reward, done, _ = env.step(0)
    #     x,y, dx, dy, a, da, tl, tr = state
    #     print(dx, dy, a, da, done)
    # #     x,y, dx, dy, a, da, tl, tr
    # done
    # im

def get_trajectory(env, agent, trajectory_len, visualize=False, saveOutput=None):
    trajectory = {
        'states': [],
        'actions': [],
        'total_reward': 0}
    state = env.reset()
    success = False
    crashed = False
    global frame_counter, score_total, score_alive, score_crash, is_recording

    # trajectory['states'].append(state)
    for _ in range(trajectory_len):
        action = agent.get_action(state)
        if saveOutput:
            saveOutput.push()
        # print(action)
        trajectory['states'].append(state)
        trajectory['actions'].append(action)
        state, reward, done, _ = env.step(action)
        if reward == -100:
            print('Crash :-(')
            crashed = True
            score_crash+=1
        if reward == 200:
            success = True
            print(state, reward, done, _)
        trajectory['total_reward'] += reward
        if done:
            break
        x,y, dx, dy, a, da, tl, tr = state
        # print(reward, x,y, dx, dy, a, da, tl, tr)
        if visualize:
            if is_recording:
                render_image(env, state, reward, done, action, crashed, success, saveOutput)
            else:
                env.render()
    if not is_recording: return trajectory

    if visualize:
        for _ in range(90):
            if not crashed:
                if _ == 30:
                    score_alive += 1
            if _ == 60:
                score_total += 1

            state, reward, done, _ = env.step(0)
            render_image(env, state, reward, done, action, crashed, success, saveOutput)
    else:
        if not crashed:
            score_alive+=1
        score_total += 1
        for _ in range(8):
            render_image(env, state, reward, done, action, crashed, success, saveOutput, no_messages=True)
    if saveOutput:
        saveOutput.clear()
        saveOutput.updateMinMax()
        print(saveOutput.get_min_max())
    # assert len(trajectory['actions']) == len(
        # trajectory['states']), f"gt {len(trajectory['actions'])},{len(trajectory['states'])}"
    # exit()
    return trajectory


def get_elite_trajectories(trajectories, q_param):
    total_rewards = [trajectory['total_reward'] for trajectory in trajectories]
    quantile = np.quantile(total_rewards, q=q_param)
    return [trajectory for trajectory in trajectories if trajectory['total_reward'] > quantile]


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

    env = gym.make(gym_name)
    for i in tqdm.tqdm(range(episode_n)):
        trajectories = [get_trajectory(env, agent, trajectory_len) for _ in range(trajectory_n)]

        mean_total_reward = np.mean(
            [trajectory['total_reward'] for trajectory in trajectories])
        mean_total_rewards.append(mean_total_reward)

        elite_trajectories = get_elite_trajectories(trajectories, q_param)
        episode_data.append((mean_total_reward, len(elite_trajectories)))

        if len(elite_trajectories) > 0:
            agent.update_policy(elite_trajectories)
            torch.save(agent.state_dict(), self.get_name())

        exp['episode_data'] = episode_data
        exp['total_elapsed'] = (
            datetime.datetime.now() - start).total_seconds()
        json.dump(exp, open(episode_data_path, 'w'))
    exp['episode_data'] = episode_data
    exp['finished'] = True
    exp['total_elapsed'] = (datetime.datetime.now() - start).total_seconds()

    json.dump(exp, open(episode_data_path, 'w'))


def executor(tasks_dir, experiments_dir):
    tasks = list(glob.glob(os.path.join(tasks_dir, '*.json')))
    for task in tasks:
        task = json.load(open(task, 'r'))
        eid = task['id']
        try:
            f = open(os.path.join(experiments_dir, eid+'.json'), 'x')
            f.close()
            print(eid)
            print(task)
            lr = task['lr']
            versions = task['version']
            layers_n = task['layers_n']
            episode_n = task['episode_n']
            trajectory_len = task['trajectory_len']
            trajectory_n = task['trajectory_n']
            q_param = task['q_param']

            run_experiment(eid, versions, layers_n, episode_n,
                           trajectory_len, trajectory_n, q_param, lr)

        except FileExistsError as err:
            pass


def show_evolution(eid, nth):
    global is_recording
    is_recording = True
    ed = json.load(open(os.path.join(EXP_DIR, eid+".json")))

    lr = ed['lr']
    layers_n = ed['layers_n']
    episode_n = ed['episode_n']
    trajectory_len = 1000  # ed['trajectory_len']
    trajectory_n = ed['trajectory_n']
    q_param = ed['q_param']
    total_elapsed = ed['total_elapsed']
    print(f'lr={lr}\nlayers_n={layers_n} \nepisode_n={episode_n}\ntrajectory_len={trajectory_len} \ntrajectory_n={trajectory_n} \nq_param={q_param} \ntotal_elapsed={total_elapsed}')
    agent = CrossEntropyMethod(eid, state_dim, action_n, layers_n=layers_n, lr=lr)
    saveOutput = SaveOutput()
    hooks = []
    for name, module in agent.named_modules():
        hook = module.register_forward_hook(saveOutput)
        hooks.append(hook)
    for n in tqdm.tqdm(range(0, episode_n)):
        agent.load_n_policy(n)
        if n%nth==0:
            t = get_trajectory(env, agent, trajectory_len, visualize=True, saveOutput=saveOutput)
        else:
            t = get_trajectory(env, agent, trajectory_len, visualize=False, saveOutput=saveOutput)
        print(t['total_reward'])

    agent.load_last_policy()
    t = get_trajectory(env, agent, trajectory_len, visualize=True, saveOutput=saveOutput)

def replay(eid, episodes=100, n=-1, get_stats=False,n_replay=-1):

    ed = json.load(open(os.path.join(EXP_DIR, eid+".json")))

    lr = ed['lr']
    layers_n = ed['layers_n']
    episode_n = ed['episode_n']
    trajectory_len = ed['trajectory_len']
    trajectory_n = ed['trajectory_n']
    q_param = ed['q_param']
    total_elapsed = ed['total_elapsed']
    print(f'lr={lr}\nlayers_n={layers_n} \nepisode_n={episode_n}\ntrajectory_len={trajectory_len} \ntrajectory_n={trajectory_n} \nq_param={q_param} \ntotal_elapsed={total_elapsed}')
    agent = CrossEntropyMethod(
        eid, state_dim, action_n, layers_n=layers_n, lr=lr)
    if n == -1:
        agent.load_last_policy()
    else:
        agent.load_n_policy(n)

    if get_stats:
        tjs = []
        for _ in tqdm.tqdm(range(episodes)):
            t = get_trajectory(env, agent, trajectory_len, visualize=False)
            tjs.append(t['total_reward'])
        print(np.mean(tjs))
        # plt.hist(tjs, bins=50)
        # plt.show()
    else:
        tjs = []
        if n_replay==-1:
            while True:
                t = get_trajectory(env, agent, trajectory_len, visualize=True)
                tjs.append(t['total_reward'])
                print(tjs[-1])
        else:
            tjs = []
            for _ in range(n_replay):
                t = get_trajectory(env, agent, trajectory_len, visualize=True)
                tjs.append(t['total_reward'])
                print(tjs[-1])
            print('Mean Reward for ', agent.name, np.mean(tjs))


if __name__ == "__main__":
    print(gym_name)
    # loves to land on the edge
    # show_evolution(eid = '2290748504803', nth=25) # best stoce
    # show_evolution(eid = '2292098547091', nth=5) # best stoce
    replay(eid = '2292098547091', n_replay=-1) # best stoce
    # replay(eid='slider2', get_stats=True, episodes=100)
    # replay(eid = 'slider2', n=25, n_replay=3) # best stoce

    # replay(eid = 'slider4', n_replay=3) 
    # show_evolution(eid = 'slider5', nth=25)
    # replay(eid = 'slider4')#, nth=100)

    # replay(eid = '6417427043')
    # replay(eid='2290748504803', n=139, get_stats=False, episodes=500)
    # replay(eid='6418223916', n=146)

    # replay(eid = '62295544359654')
    # replay(eid='2295195100690')
#     replay(eid='2295269733941')
