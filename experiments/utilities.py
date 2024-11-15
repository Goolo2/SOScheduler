import numpy as np
import os
import pickle
import random


def makefile(dir_name):
    if os.path.exists(f'{dir_name}.pkl'):
        i = 1
        while True:
            new_name = dir_name + "_" + str(i)
            if not os.path.exists(f'{new_name}.pkl'):
                dir_name = new_name
                break
            i += 1
    print("savedir:", dir_name)
    return f'{dir_name}.pkl'


def makedir(dir_name):
    if os.path.exists(f'{dir_name}'):
        i = 1
        while True:
            new_name = dir_name + "_" + str(i)
            if not os.path.exists(f'{new_name}'):
                dir_name = new_name
                break
            i += 1
    print("savedir:", dir_name)
    return f'{dir_name}'


def readpkl(pkldir, seed=None):
    with open(pkldir, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    if seed:
        data = data[seed]
    return data


def allocate_robots_to_fires(N, M):
    '''
    N: number of fires
    M: number of robots
    '''
    fires = list(range(N))
    robots = list(range(M))
    
    random.shuffle(fires)
    
    robot_allocations = {}
    for i, robot_id in enumerate(robots):
        fire_id = fires[i % N]
        robot_allocations[robot_id] = fire_id
    
    return robot_allocations

def print_metrics(logger):
    print(f'coverage: {np.mean(logger.save_data["coverage"]): .3f}')
    print(f'accuracy: {np.mean(logger.save_data["accuracy"]): .3f}')
    print(f'healthy: {logger.save_data["healthy"][-1]: .3f}')
    print(f'onfire: {logger.save_data["onfire"][-1]: .3f}')
    print(f'burnt: {logger.save_data["burnt"][-1]: .3f}')
