import networkx as nx
import numpy as np
from operator import itemgetter
import scipy.ndimage as sn
import scipy.stats as ss
from copy import deepcopy
from collections import defaultdict

from experiments import utilities, environments
from objectives import objective
from models import bayesian
from .base_strategy import BaseStrategy
import scipy.stats as ss


class ConfEntropy(BaseStrategy):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.confidence = np.zeros((self.dimension, self.dimension))
        self.omega = [[[] for _ in range(self.dimension)] for _ in range(self.dimension)]
        self.sigma_omega = 1

    def plan(self, env, robots, team_belief):
        predicted_belief = deepcopy(team_belief)
        belief_updates = int(self.horizon//self.update_interval)
        for _ in range(belief_updates):
            predicted_belief = self.filter.update_belief(env.group, predicted_belief, True,
                                                         dict(), self.measure_correct, self.threshold)

        information_gain = objective.compute_entropy(predicted_belief, self.dimension)
        suppress_gain = objective.compute_suppress_gain(predicted_belief, self.dimension)

        self.update_confidence(robots, team_belief)

        self.info_weights = [np.zeros((self.dimension, self.dimension)) for _ in range(self.num_robot)]
        self.suppress_weights = [np.zeros((self.dimension, self.dimension)) for _ in range(self.num_robot)]

        for ridx, robot in enumerate(robots.values()):
            if robot.reached_fire:
                suppress_weights = sn.filters.convolve(suppress_gain,
                                                       np.ones((self.suppress_size, self.suppress_size)),
                                                       mode='constant', cval=0)
                info_weights = sn.filters.convolve(information_gain,
                                                   np.ones((self.image_size, self.image_size)),
                                                   mode='constant', cval=0)

                norm_info_weights = info_weights / np.sum(info_weights)
                norm_suppress_weights = suppress_weights / np.sum(suppress_weights)

                scores = self.confidence*norm_suppress_weights + (1-self.confidence)*norm_info_weights
                # print(scores)

                robot_path = self.graph_search(robot.position, robot.horizon, scores)

                shape = env.dense_state().shape
                discount = 0.8
                for idx, location in enumerate(robot_path[::-1]):  # ! reverse
                    supp_half_row = (self.suppress_size-1)//2
                    supp_half_col = (self.suppress_size-1)//2
                    for ri, dr in enumerate(np.arange(-supp_half_row, supp_half_row+1, 1)):
                        for ci, dc in enumerate(np.arange(-supp_half_col, supp_half_col+1, 1)):
                            r = location[0] + dr
                            c = location[1] + dc

                            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                                suppress_gain[r, c] *= pow(discount, idx+1)

                    info_half_row = (self.image_size-1)//2
                    info_half_col = (self.image_size-1)//2
                    for ri, dr in enumerate(np.arange(-info_half_row, info_half_row+1, 1)):
                        for ci, dc in enumerate(np.arange(-info_half_col, info_half_col+1, 1)):
                            r = location[0] + dr
                            c = location[1] + dc

                            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                                information_gain[r, c] *= pow(discount, idx+1)

                robot.plan = robot_path[1:]

                # self.info_weights.append(norm_info_weights)
                # self.suppress_weights.append(norm_suppress_weights)
                self.info_weights[ridx] = norm_info_weights
                self.suppress_weights[ridx] = norm_suppress_weights

            else:
                distances = []
                # print(f'robot fire center = {robot.fire_center}')
                for (dr, dc) in self.movements:
                    if (dr, dc) == (0, 0):
                        continue
                    new_position = (robot.position[0] + dr, robot.position[1] + dc)
                    dist = np.linalg.norm(np.array(new_position) - np.array(robot.fire_center))
                    if 0 <= new_position[0] < self.dimension and 0 <= new_position[1] < self.dimension:
                        distances.append((new_position, dist))
                next_position, _ = min(distances, key=lambda x: x[1])
                robot.plan = [next_position]

    def update_confidence(self, robots, team_belief):
        for i in range(self.confidence.shape[0]):
            for j in range(self.confidence.shape[1]):
                for robot in robots.values():
                    dist2robot = self.get_chebyshev_dist((i, j), robot.position)
                    self.omega[i][j].append(ss.norm.pdf(dist2robot, loc=0, scale=1))

                while len(self.omega[i][j]) > 50:
                    self.omega[i][j].pop(0)
                    
                self.confidence[i][j] = 1 - np.exp(-np.sum(self.omega[i][j]) / (ss.entropy(team_belief[(i, j)]) + 0.1))
        self.confidence = np.clip(self.confidence, 0, 1)

    def get_chebyshev_dist(self, position1, position2):
        return max(abs(position1[0] - position2[0]), abs(position1[1] - position2[1]))
