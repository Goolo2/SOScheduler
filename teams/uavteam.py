from copy import deepcopy
import numpy as np
from models import bayesian
from collections import defaultdict


class UAVTeam(object):
    def __init__(self, args, robots):
        self.robots = robots
        self.dimension = args.dimension
        self.suppress = args.suppress
        self.measure_correct = args.measure_correct
        self.threshold = args.threshold
        self.delta_beta = args.delta_beta
        self.base_station = (0, 0)

        self.filter = bayesian.BayesianFilter(args)

        self.team_belief = deepcopy(robots[1].belief)
        self.team_observation = dict()
        self.team_control = defaultdict(lambda: (0.0, 0.0))
        self.runtime = []

    def sense(self, env):
        # update team observations
        self.team_observation = dict()
        for robot in self.robots.values():
            _, observation = robot.camera.get_image(robot.position, env)
            for key in observation.keys():
                if key not in self.team_observation:
                    self.team_observation[key] = []
                self.team_observation[key].append(observation[key])

    def perception(self, env, advance):
        # calculate team control
        if self.suppress:
            team_control_field = np.zeros(env.dims)
            for robot in self.robots.values():
                control_field = robot.extinguisher.get_control(robot.position, env)
                team_control_field = np.logical_or(team_control_field, control_field).astype(int)
                robot.capacity -= np.sum(control_field)
                
            for r in range(team_control_field.shape[0]):
                for c in range(team_control_field.shape[1]):
                    if team_control_field[r, c] == 1:
                        self.team_control[r, c] = (0.0, self.delta_beta)
        # update team belief
        self.team_belief = self.filter.update_belief(env.group, self.team_belief, advance,
                                                  self.team_observation, self.measure_correct, self.threshold, control=self.team_control)

        dense_belief = np.array([[np.argmax(self.team_belief[(r, c)]) for c in range(env.dims[1])]
                            for r in range(env.dims[0])]) 
        indices = np.where(dense_belief == 1)
        mean_r = np.mean(indices[0]).astype(np.uint8)
        mean_c = np.mean(indices[1]).astype(np.uint8)
        fire_center = (mean_r, mean_c) if indices[0].size > 0 else (0, 0)
        
        # estimate whether reach the fire center based on team belief
        for robot in self.robots.values():
            # robot.fire_center = fire_center
            r0, c0 = robot.position
            half_row = (robot.camera.image_size-1)//2
            half_col = (robot.camera.image_size-1)//2
            for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
                for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
                    r = r0 + dr
                    c = c0 + dc

                    if 0 <= r < env.dense_state().shape[0] and 0 <= c < env.dense_state().shape[1]:
                        if np.argmax(self.team_belief[(r, c)]) in [1, 2]:  # onfire or burnt
                            robot.reached_fire = True
                            break
                if robot.reached_fire:
                    break

    def navigation(self):
        for robot in self.robots.values():
            # return to base station if capacity is 0
            # print(robot.capacity)
            if robot.capacity <= 0:
                robot.position = self.base_station
                robot.capacity = robot.max_capacity
                robot.reached_fire = False
            else:
                robot.step()
