import gym
import tqdm
import numpy as np
from collections import defaultdict

def epsilon_greedy_policy_action(epsilon, state, q_value, n_actions):
    pol = np.zeros((n_actions,))+ epsilon/n_actions
    idx = np.argmax(q_value[state])
    pol[idx] += 1 - epsilon
    return np.random.choice(n_actions, p=pol)

def get_action(policy, state):
    return np.random.choice(policy.shape[1], p=policy[state])

def gratitude(gamma, rewards):
    G = np.zeros((len(rewards),))
    T = len(rewards)
    for t in range(T-1):
        for epsodes_n in range(t, T-1):
            G[t] += gamma**(epsodes_n-t) * rewards[epsodes_n]
    return G
def get_lunar_state_nbr(state):
    # The state is an 8-dimensional vector: the coordinates of the lander in x & y,
    # its linear velocities in x & y, its angle, its angular velocity, 
    # and two booleans that represent whether each leg is in contact with the ground or not.
    x,y,vx,vy,a,va,tl,tr = state
    mlt = 10
    new_state =[ x*mlt,y*mlt,vx*mlt,vy*mlt, a*mlt,va*mlt, tl, tl]
    return '_'.join([str(int(s)) for s in new_state])
def get_trajectory(env, policy, trajectory_n):
    rewards = []
    actions = []
    states = []
    state = env.reset()
    for _ in range(trajectory_n):
        states.append(state)
        action = get_action(policy, state)
        actions.append(action)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break
    return states, actions, rewards

def MC(epsodes_n, trajectory_n, epsilon, gamma):
    env = gym.make('LunarLander-v2')
    n_actions = 4
    N = defaultdict(lambda : np.zeros((n_actions,)))
    Q = defaultdict(lambda : np.zeros((n_actions,)))
    G = None
    total_rewards = []
    for ki in range(epsodes_n):
        
        rewards = []
        state = get_lunar_state_nbr(env.reset())
        actions = []
        states = []

        for _ in range(trajectory_n):
            action = epsilon_greedy_policy_action(epsilon, state, Q, n_actions)
            state, reward, done, _ = env.step(action)
            state = get_lunar_state_nbr(state)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            if done:
                break
            # updating Q
        # policy = epsilon_greedy_policy(epsilon, Q)
        #states, actions, rewards = get_trajectory(env, policy, trajectory_n)
        total_rewards.append(int(np.sum(rewards, dtype=np.int32)))
        G = gratitude(gamma, rewards)
        
        # updating Q
        for g, a, s in zip(G, actions, states):
            Q[s][a] += (g - Q[s][a])/(N[s][a]+1)
            N[s][a] += 1

        epsilon=1-ki/epsodes_n
#         epsilon*=0.8
    return total_rewards

def SARSA(epsodes_n, trajectory_n, alpha, epsilon, gamma):
    env = gym.make('LunarLander-v2')
    n_actions = 4
    Q = defaultdict(lambda : np.zeros((n_actions,)))

    # stats = []
    # policies = []
    total_rewards = []

    for ki in range(epsodes_n):
        rewards = []
        state = get_lunar_state_nbr(env.reset())
        action = epsilon_greedy_policy_action(epsilon, state, Q, n_actions)
        
        actions = [action]
        states = [state]

        for _ in range(trajectory_n):

            state, reward, done, _ = env.step(action)
            state = get_lunar_state_nbr(state)
            rewards.append(reward)
            states.append(state)
            action = epsilon_greedy_policy_action(epsilon, state, Q, n_actions)
            actions.append(action)
            if done:
                break
            # updating Q
            Q[states[-2]][actions[-2]] += alpha*(reward + gamma*Q[states[-2]][actions[-2]])-Q[states[-2]][actions[-2]]

        total_rewards.append(int(np.sum(rewards, dtype=np.int32)))
        epsilon=1/(ki+1)
    return total_rewards


def QLearning(epsodes_n, trajectory_n, alpha, epsilon, gamma):
    env = gym.make('LunarLander-v2')
    n_actions = 4
    Q = defaultdict(lambda : np.zeros((n_actions,)))

    # stats = []
    # policies = []
    total_rewards = []

    for ki in range(epsodes_n):
        rewards = []
        state = env.reset()
        state = get_lunar_state_nbr(state)
        action = epsilon_greedy_policy_action(epsilon, state, Q, n_actions)
        
        actions = [action]
        states = [state]

        for _ in range(trajectory_n):
            state, reward, done, _ = env.step(action)
            state = get_lunar_state_nbr(state)

            rewards.append(reward)
            states.append(state)
            action = epsilon_greedy_policy_action(epsilon, state, Q, n_actions)
            actions.append(action)
            if done:
                break

            # updating Q
            Q[states[-2]][actions[-2]] += alpha*(reward + gamma*max(Q[states[-1]]))-Q[states[-2]][actions[-2]]

        total_rewards.append(int(np.sum(rewards, dtype=np.int32)))
        epsilon=1/(ki+1)
    return total_rewards