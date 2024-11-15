import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from collections import defaultdict
import numpy as np


from simulators.fires.LatticeForest import LatticeForest

from baselines.ddrl.rlUtilities import latticeforest_image, actions2trajectory, xy2rc, move_toward_center, reward, heuristic

class UAV(object):
    """
    Autonomous unmanned aerial vehicle model.
    """
    healthy = 0
    on_fire = 1
    burnt = 2

    move_deltas = [(-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-1, -1), (0, -1), (1, -1)]  # excluded (0,0)
    fire_neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1)]

    def __init__(self, numeric_id=None, initial_position=None, fire_center=None, image_dims=(3, 3), camera=None, extinguisher=None, capacity=None, belief=None):
        self.numeric_id = numeric_id
        self.initial_position = initial_position
        self.fire_center = fire_center
        self.image_dims = image_dims

        self.position = self.initial_position
        self.next_position = None

        self.capacity = capacity
        self.max_capacity = capacity

        self.image = None
        self.reached_fire = False
        self.rotation_vector = None
        self.closest_agent_id = None
        self.closest_agent_position = None
        self.closest_agent_vector = None

        self.features = None

        self.actions = []
        self.rewards = []
        
        self.camera = camera
        self.extinguisher = extinguisher
        self.capacity = capacity
        self.belief = belief

    def reset(self):
        self.reached_fire = False
        self.next_position = None
        self.actions = []
        self.rewards = []

    def update_position(self):
        self.position = self.next_position
        self.next_position = None

    def update_features(self, forest_state, team):
        height, width = forest_state.shape
        self.image = latticeforest_image(forest_state, xy2rc(height, self.position), self.image_dims) 
        
        image_center = (self.image_dims[0]-1)//2, (self.image_dims[1]-1)//2
        if self.image[image_center[0], image_center[1]] in [self.on_fire, self.burnt]:
            self.reached_fire = True

        self.rotation_vector = self.position - self.fire_center
        norm = np.linalg.norm(self.rotation_vector, 2)
        if norm != 0:
            self.rotation_vector = self.rotation_vector / norm
        self.rotation_vector = np.array([self.rotation_vector[1], -self.rotation_vector[0]])

        # d = [(np.linalg.norm(self.position-agent.position, 2), agent.numeric_id, agent.position)
        #      for agent in team.values() if agent.numeric_id != self.numeric_id] # !for train
        
        d = [(np.linalg.norm(self.position-agent.position, 2), agent.numeric_id, agent.position)
             for agent in team.robots.values() if agent.numeric_id != self.numeric_id]
        
        _, self.closest_agent_id, self.closest_agent_position = min(d, key=lambda x: x[0])

        self.closest_agent_vector = self.position - self.closest_agent_position
        norm = np.linalg.norm(self.closest_agent_vector)
        if norm != 0:
            self.closest_agent_vector = self.closest_agent_vector/norm

        return np.concatenate((self.image.ravel(), self.rotation_vector,
                              np.asarray(self.numeric_id > self.closest_agent_id)[np.newaxis],
                               self.closest_agent_vector))