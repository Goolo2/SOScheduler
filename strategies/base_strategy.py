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


class BaseStrategy(object):
    def __init__(self, args) -> None:
        self.update_interval = args.update_interval
        self.horizon = args.horizon
        self.image_size = args.image_size
        self.suppress_size = args.suppress_size
        self.movements = args.movements
        self.dimension = args.dimension
        self.threshold = args.threshold
        self.num_robot = args.num_robot
        self.measure_correct = args.measure_correct
        self.filter = bayesian.BayesianFilter(args)

        self.info_weights = [np.zeros((self.dimension, self.dimension)) for _ in range(self.num_robot)]
        self.suppress_weights = [np.zeros((self.dimension, self.dimension)) for _ in range(self.num_robot)]

    def plan(self, env, robots, team_belief):
        predicted_belief = deepcopy(team_belief)
        belief_updates = int(self.horizon//self.update_interval)
        for _ in range(belief_updates):
            predicted_belief = self.filter.update_belief(env.group, predicted_belief, True,
                                                         dict(), self.measure_correct, self.threshold)

        information_gain = objective.compute_entropy(predicted_belief, self.dimension)
        suppress_gain = objective.compute_suppress_gain(predicted_belief, self.dimension)

        pred_entropy = objective.compute_entropy(predicted_belief, self.dimension)
        alpha = np.mean(pred_entropy)  # !
        alpha = np.clip(alpha, 0.0, 1.0)

        self.info_weights = []
        self.suppress_weights = []

        for robot in robots.values():
            if robot.reached_fire:
                suppress_weights = sn.filters.convolve(suppress_gain,
                                                       np.ones((self.suppress_size, self.suppress_size)),
                                                       mode='constant', cval=0)
                info_weights = sn.filters.convolve(information_gain,
                                                   np.ones((self.image_size, self.image_size)),
                                                   mode='constant', cval=0)

                norm_info_weights = info_weights / np.sum(info_weights)
                norm_suppress_weights = suppress_weights / np.sum(suppress_weights)
                scores = alpha*norm_info_weights + (1-alpha)*norm_suppress_weights

                robot_path = self.graph_search(robot.position, robot.horizon, scores)

                shape = env.dense_state().shape
                discount = 0.8
                for idx, location in enumerate(robot_path[::-1]):  # !
                    supp_half_row = (self.suppress_size-1)//2
                    supp_half_col = (self.suppress_size-1)//2
                    for ri, dr in enumerate(np.arange(-supp_half_row, supp_half_row+1, 1)):
                        for ci, dc in enumerate(np.arange(-supp_half_col, supp_half_col+1, 1)):
                            r = location[0] + dr
                            c = location[1] + dc

                            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                                # suppress_gain[r, c] = 0
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

                self.info_weights.append(norm_info_weights)
                self.suppress_weights.append(norm_suppress_weights)

            else:
                distances = []
                for (dr, dc) in self.movements:
                    if (dr, dc) == (0, 0):
                        continue
                    new_position = (robot.position[0] + dr, robot.position[1] + dc)
                    dist = np.linalg.norm(np.array(new_position) - np.array(env.fire_center))
                    # env.fire_center needed? The real data has no fire_center
                    # assume at the middle of the grid for consistency
                    if 0 <= new_position[0] < self.dimension and 0 <= new_position[1] < self.dimension:
                        distances.append((new_position, dist))
                next_position, _ = min(distances, key=lambda x: x[1])
                robot.plan = [next_position]

    def graph_search(self, start, length, weights):

        graph = nx.DiGraph()
        nodes = [(start, length)]

        # for each node, find other nodes that can be moved to with the remaining amount of path length
        while nodes:
            current_node, current_length = nodes.pop(0)
            if current_length == 0:
                continue

            for (dr, dc) in self.movements:
                if (dr, dc) == (0, 0):
                    continue

                neighbor_node = (current_node[0] + dr, current_node[1] + dc)

                neighbor = (neighbor_node, int(current_length-1))
                edge = ((current_node, current_length), neighbor)
                if graph.has_edge(edge[0], edge[1]):
                    continue

                if 0 <= neighbor_node[0] < self.dimension and 0 <= neighbor_node[1] < self.dimension:
                    nodes.append(neighbor)
                    graph.add_edge(edge[0], edge[1], weight=1e-4+weights[neighbor_node[0], neighbor_node[1]])

        if len(graph.edges()) == 1:
            raise ValueError

        path = nx.algorithms.dag_longest_path(graph)
        path_weight = sum([graph.get_edge_data(path[i], path[i+1])['weight'] for i in range(len(path)-1)])
        path = [element[0] for element in path]

        return path
