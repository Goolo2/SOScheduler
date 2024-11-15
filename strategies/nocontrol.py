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


class NoControl(BaseStrategy):
    def __init__(self, args) -> None:
        super().__init__(args)

    def plan(self, env, robots, team_belief):
        for ridx, robot in enumerate(robots.values()):
            robot.plan = [robot.position]
