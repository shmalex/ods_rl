import tqdm
import utils as u
import numpy as np

if __name__ == "__main__":
    # Monte-Carlo
    # Epsilon
    # Gamma
    # K
    # params = {
    #     'epsilon': [1],
    #     'gamma' : np.linspace(0.001, 0.999, num=10),
    #     'epsodes_n' : [1000],
    #     'trajectory_n' : [50, 100, 300],
    # }
    # version = 2
    # tasks, keys = u.nest_permutate(params)
    # print("total tasks", len(tasks))
    # u.generate_tasks('mc', version, tasks, keys)

    # params['alpha' ] = np.linspace(0.001, 0.999, num=10)
    # tasks, keys = u.nest_permutate(params)
    # print("total tasks", len(tasks))
    # u.generate_tasks('sarsa', version, tasks, keys)

    # tasks, keys = u.nest_permutate(params)
    # print("total tasks", len(tasks))
    # u.generate_tasks('ql', version, tasks, keys)

    # practice 1 - MC
    # проверим работу
    # params = {
    #     'epsilon': [1],
    #     'gamma' : np.linspace(0.98, 0.999, num=100),
    #     'epsodes_n' : [1000],
    #     'trajectory_n' : [300],
    # }

    # version = 4
    # tasks, keys = u.nest_permutate(params)
    # print("total tasks", len(tasks))
    # u.generate_tasks('mc', version, tasks, keys)


    # practice 1 - MC
    # проверим работу
    params = {
        'epsilon': [1],
        'gamma' : np.linspace(0.1, 0.4, num=100),
        'epsodes_n' : [1000],
        'trajectory_n' : [300],
    }

    version = 5
    tasks, keys = u.nest_permutate(params)
    print("total tasks", len(tasks))
    u.generate_tasks('mc', version, tasks, keys)

    # practice 2 - parameters
    params = {
        'epsilon': [1],
        'gamma' : np.linspace(0.001, 0.999, num=50),
        'epsodes_n' : [1000],
        'trajectory_n' : [300],
    }
    version = 1
    tasks, keys = u.nest_permutate(params)
    print("total tasks", len(tasks))
    u.generate_tasks('mc2', version, tasks, keys)

    params['alpha' ] = np.linspace(0.001, 0.999, num=50)
    tasks, keys = u.nest_permutate(params)
    print("total tasks", len(tasks))
    u.generate_tasks('sarsa2', version, tasks, keys)

    tasks, keys = u.nest_permutate(params)
    print("total tasks", len(tasks))
    u.generate_tasks('ql2', version, tasks, keys)



    # practice 2 - parameters
    # 100 multiplier for space
    params = {
        'epsilon': [1],
        'gamma' : [0.8, 0.85, 0.9, 0.95],
        'epsodes_n' : [400],
        'trajectory_n' : [200],
    }
    version = 10
    tasks, keys = u.nest_permutate(params)
    print("total tasks", len(tasks))
    u.generate_tasks('mc2', version, tasks, keys)

    params['alpha' ] = [0.8,0.85,0.9,0.95]
    tasks, keys = u.nest_permutate(params)
    print("total tasks", len(tasks))
    u.generate_tasks('sarsa2', version, tasks, keys)

    tasks, keys = u.nest_permutate(params)
    print("total tasks", len(tasks))
    u.generate_tasks('ql2', version, tasks, keys)


    # practice 2 - parameters
    # 1000 multiplier for space
    params = {
        'epsilon': [1],
        'gamma' : [0.8, 0.85, 0.9, 0.95],
        'epsodes_n' : [1000],
        'trajectory_n' : [1000],
    }
    version = 11
    tasks, keys = u.nest_permutate(params)
    print("total tasks", len(tasks))
    u.generate_tasks('mc2', version, tasks, keys)

    params['alpha' ] = [0.8,0.85,0.9,0.95]
    tasks, keys = u.nest_permutate(params)
    print("total tasks", len(tasks))
    u.generate_tasks('sarsa2', version, tasks, keys)

    tasks, keys = u.nest_permutate(params)
    print("total tasks", len(tasks))
    u.generate_tasks('ql2', version, tasks, keys)


    # practice 2 - parameters
    # 1000 multiplier for space

    params = {
        'epsilon': [1],
        'gamma' : [0.8, 0.85, 0.9, 0.95],
        'epsodes_n' : [10000],
        'trajectory_n' : [1000],
    }
    version = 12
    tasks, keys = u.nest_permutate(params)
    print("total tasks", len(tasks))
    u.generate_tasks('mc2', version, tasks, keys)

    params['alpha' ] = [0.8,0.85,0.9,0.95]
    tasks, keys = u.nest_permutate(params)
    print("total tasks", len(tasks))
    u.generate_tasks('sarsa2', version, tasks, keys)

    tasks, keys = u.nest_permutate(params)
    print("total tasks", len(tasks))
    u.generate_tasks('ql2', version, tasks, keys)

    # practice 2 - parameters
    # 10 multiplier for space

    params = {
        'epsilon': [1],
        'gamma' : [0.8, 0.85, 0.9, 0.95],
        'epsodes_n' : [100000],
        'trajectory_n' : [1000],
    }
    version = 13
    tasks, keys = u.nest_permutate(params)
    print("total tasks", len(tasks))
    u.generate_tasks('mc2', version, tasks, keys)

    params['alpha' ] = [0.8,0.85,0.9,0.95]
    tasks, keys = u.nest_permutate(params)
    print("total tasks", len(tasks))
    u.generate_tasks('sarsa2', version, tasks, keys)

    tasks, keys = u.nest_permutate(params)
    print("total tasks", len(tasks))
    u.generate_tasks('ql2', version, tasks, keys)

