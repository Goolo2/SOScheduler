from pathlib import Path
import numpy as np
from copy import copy
import json
import pickle
import os
from experiments.utilities import makefile


class Logger:
    """Save all the variables for visualization."""

    def __init__(self, args):
        if args.env == 1:
            fire_map = "Lattice"
        elif args.env == 2:
            fire_map = "Harvest40x40"
        elif args.env == 3:
            fire_map = "Sub40x40"
        else:
            raise ValueError("Invalid environment selection.")
        
        self.save_data = {'info': {'dimension': args.dimension,
                                   'fire_size': args.fire_size,
                                   'fire_num': args.fire_num,
                                   'image_size': args.image_size,
                                   "update_interval": args.update_interval,
                                   "update_minutes": args.update_minutes,
                                   'fire_map': fire_map, # add the fireman
                                   'suppress_size': args.suppress_size}, 'time_series': dict(), 'coverage': [],
                          'accuracy': [], 'healthy': [], 'onfire': [], 'burnt': [], 'plan_time': [], 'runtime': 0.0}

        self.save_dir = args.save_dir
        os.makedirs(args.save_dir, exist_ok=True)
        # self.save_name = f'E{args.env}_D{args.dimension}_FS{args.fire_size}_FN{args.fire_num}_U{args.update_interval}_T{args.horizon}_C{args.communication}_N{args.num_robot}_I{args.image_size}_S{args.suppress_size}_M{args.measure_correct:.2f}_A{args.alpha:.2f}_B{args.beta:.2f}'
        self.save_name = args.save_name

    def append(self, t, env_state, team, coverage, accuracy, healthy, onfire, burnt):
        self.save_data['time_series'][t] = {
            'position': {label: team.robots[label].position for label in team.robots},
            'env_state': env_state,
            'team_belief': team.team_belief
        }
        self.save_data['coverage'].append(coverage)
        self.save_data['accuracy'].append(accuracy)
        self.save_data['healthy'].append(healthy)
        self.save_data['onfire'].append(onfire)
        self.save_data['burnt'].append(burnt)

    def append_weight(self, t, info_weights, suppress_weights):
        self.save_data['time_series'][t]['info_weights'] = info_weights
        self.save_data['time_series'][t]['suppress_weights'] = suppress_weights

    def append_confweight(self, t, conf_weights):
        self.save_data['time_series'][t]['conf_weights'] = conf_weights
        
    def append_time(self, time):
        self.save_data['plan_time'].append(time)
        
    def save(self, runtime) -> None:
        self.save_dir = makefile(f'{self.save_dir}/{self.save_name}')
        # self.save_dir = f'{self.save_dir}/{self.save_name}.pkl'
        self.save_data['save_name'] = self.save_name
        self.save_data['runtime'] = runtime
        with open(f"{self.save_dir}", 'wb') as handle:
            pickle.dump(self.save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print()
        print(f"Saved log.json to {self.save_dir}")
