from experiments import utilities
#from gadentools.Utils import Vector3
# from dynamics import DubinsCar, Walk
import numpy as np
import math


class UAV(object):
    def __init__(
            self,
            label,
            fire_center,
            init_pos,
            image_size,
            belief,
            horizon,
            camera,
            extinguisher,
            capacity,
    ):

        self.label = label
        self.fire_center = fire_center
        self.position = init_pos
        self.image_size = image_size
        self.belief = belief
        self.horizon = horizon
        self.camera = camera
        self.extinguisher = extinguisher
        self.capacity = capacity
        self.max_capacity = capacity
        self.plan = []
        self.reached_fire = False


    def step(self) -> None:
        # Update state
        self.position = self.plan[0]
        self.plan.pop(0)
