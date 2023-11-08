import os
import glob
import json
import time
import tqdm

TASKS_DIR = './tasks/'
EXP_DIR = './experiments/'

for d in [TASKS_DIR, EXP_DIR]:
    os.makedirs(d, exist_ok=True)


def nest_permutate(params):
    keys = list(sorted(params.keys()))
    combins = [params[key] for key in keys]
    tasks = []
    def permutate(seq, vals, opts):
        for i in opts[0]:
            if len(opts) == 1:
                seq.append(vals+[i])
            else:
                permutate(seq, vals+[i], opts[1:])
    permutate(tasks, [], combins)
    return tasks, keys


def generate_tasks(name, version, tasks, keys):

    created = 0
    # u.generate_tasks(name, version, **{k:v for k, v in zip(keys, task)})


    version = str(version)
    os.makedirs(TASKS_DIR, exist_ok=True)
    files = list(glob.glob(os.path.join(TASKS_DIR, '*.json')))

    exp_set = set()

    for file in files:
        ed = json.load(open(file, 'r'))
        exp_id = ed['name'] + '_' + ed['version'] + '_' + '_'.join([str(ed['params'][key]) for key in keys if key in ed['params']])
        exp_set.add(exp_id)

    for task in tqdm.tqdm(tasks):
        exp_id = name + '_' + version + '_' + '_'.join(map(str, task))
        if exp_id in exp_set:
            # print('Allready created', exp_id)
            continue

        created += generate_task(name, version, task, keys)

    print(name, 'created', created,'new tasks\n')

def generate_task(name, version, task, keys):
    exp = {
        'id': str(time.time_ns()),
        'version': version,
        'name': name,
        'params':{}
    }
    params = exp['params']
    for k, v in zip(keys, task):
        params[k] = v

    f = open(os.path.join(TASKS_DIR, f'{exp["id"]}.json'), 'w')
    json.dump(exp, f)
    return 1

