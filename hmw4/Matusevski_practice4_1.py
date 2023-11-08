import gym
import tqdm
import numpy as np

def epsilon_greedy_policy(epsilon, q_value):
    n_actions = q_value.shape[1]
    pol = np.zeros(q_value.shape)+ epsilon/n_actions
    idx = np.argmax(q_value, axis=1)
    pol[np.arange(q_value.shape[0]),idx] += 1-epsilon
    return pol

def epsilon_greedy_policy_action(epsilon, state, q_value):
    n_actions = q_value.shape[1]
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
    env = gym.make('Taxi-v3')
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    N = np.zeros((n_states, n_actions))
    Q = np.zeros((n_states, n_actions))
    # stats = []
    # policies = []
    total_rewards = []
    for ki in range(epsodes_n):
        policy = epsilon_greedy_policy(epsilon, Q)
        # policies.append(policy)
        # stats.append(test_stats(env, policy, 100, n_actions))

        states, actions, rewards = get_trajectory(env, policy, trajectory_n)
        total_rewards.append(int(np.sum(rewards, dtype=np.int32)))
        G = gratitude(gamma, rewards)
        # updating Q
        for g, a, s in zip(G, actions, states):
            Q[s,a] += (g - Q[s,a])/(N[s,a]+1)
            N[s,a] += 1

        epsilon=1-ki/epsodes_n
    return total_rewards
    
def  SARSA(epsodes_n, trajectory_n, alpha, epsilon, gamma):
    env = gym.make('Taxi-v3')
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    # stats = []
    # policies = []
    total_rewards = []

    for ki in range(epsodes_n):
        rewards = []
        state = env.reset()
        action = epsilon_greedy_policy_action(epsilon, state, Q)
        
        actions = [action]
        states = [state]

        for _ in range(trajectory_n):

            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            states.append(state)
            action = epsilon_greedy_policy_action(epsilon, state, Q)
            actions.append(action)
            if done:
                break
            # updating Q
            Q[states[-2],actions[-2]] += alpha*(reward + gamma*Q[states[-2],actions[-2]])-Q[states[-2],actions[-2]]

        total_rewards.append(int(np.sum(rewards, dtype=np.int32)))
        epsilon=1/(ki+1)
    return total_rewards

def QLearning(epsodes_n, trajectory_n, alpha, epsilon, gamma):
    env = gym.make('Taxi-v3')
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    # stats = []
    # policies = []
    total_rewards = []

    for ki in range(epsodes_n):
        rewards = []
        state = env.reset()
        action = epsilon_greedy_policy_action(epsilon, state, Q)
        
        actions = [action]
        states = [state]

        for _ in range(trajectory_n):
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            states.append(state)
            action = epsilon_greedy_policy_action(epsilon, state, Q)
            actions.append(action)
            if done:
                break

            # updating Q
            Q[states[-2],actions[-2]] += alpha*(reward + gamma*max(Q[states[-1]]))-Q[states[-2],actions[-2]]

        total_rewards.append(int(np.sum(rewards, dtype=np.int32)))
        epsilon=1/(ki+1)
    return total_rewards
