env = gym.make('Taxi-v3')
n_states = 500
n_actions = 4
eps = 1
gamma = 0.1
N = np.zeros((n_states, n_actions))
Q = np.zeros((n_states, n_actions))
K = 1000
stats = []
policies = []
total_rewards = []
for ki in tqdm.tqdm(range(K)):
    policy = epsilon_greedy_policy(eps, Q)
#     policies.append(policy)
#     stats.append(test_stats(env, policy, 100, n_actions))
    states, actions, rewards = get_trajectory(env, policy)
    total_rewards.append(np.sum(rewards))
    G = gratitude(gamma, rewards)
    # updating Q
    for g, a, s in zip(G, actions, states):
        Q[s,a] += (g - Q[s,a])/(N[s,a]+1)
    # updating N
#     for a, s in zip(actions, states):      
        N[s,a] += 1
    eps= 1-ki/K