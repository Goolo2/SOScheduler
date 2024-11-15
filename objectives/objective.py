import numpy as np
import scipy.stats as ss
from experiments import utilities, environments


def compute_entropy(belief, dimension, offset=0.01):
    """
    Compute the entropy of a belief.

    :param belief: a dictionary describing the belief, as {Tree position: list of probabilities for each state} key
    value pairs.
    :param config: a Config class object which contains the meeting and planning parameters.

    :return: the entropy of the belief as a 2D numpy array, where each (row, col) corresponds to a Tree position
    """
    entropy = np.zeros((dimension, dimension))
    for key in belief.keys():
        entropy[key[0], key[1]] = ss.entropy(belief[key]) # something wrong with other data

    return entropy+offset


def compute_mutual_info(belief, simulation_group, measure_correct, dimension, threshold, offset=0.1):
    entropy = compute_entropy(belief, dimension)
    conditional_entropy = np.zeros((dimension, dimension))
    for key in belief.keys():
        element = simulation_group[key]
        p_yi_ci = np.asarray([[environments.measure_model(element, ci, yi, measure_correct) for ci in element.state_space]
                              for yi in element.state_space])
        p_yi = np.matmul(p_yi_ci, belief[key])
        for yi in element.state_space:
            if p_yi[yi] <= threshold:
                continue
            for ci in element.state_space:
                if belief[key][ci] <= threshold or p_yi_ci[yi, ci] <= threshold:
                    continue
                conditional_entropy[key[0], key[1]] -= p_yi_ci[yi, ci] * belief[key][ci] * (np.log(p_yi_ci[yi, ci])
                                                                                            + np.log(belief[key][ci])
                                                                                            - np.log(p_yi[yi]))

    return entropy - conditional_entropy + offset


def compute_suppress_gain(belief, dimension, offset=0.01):
    dense_belief = np.array([[np.argmax(belief[(r, c)]) for c in range(dimension)]
                             for r in range(dimension)])
    dense_belief[dense_belief == 2] = 1

    dense_belief = dense_belief.astype(np.uint8)

    edges = []
    for i in range(dense_belief.shape[0]):
        for j in range(dense_belief.shape[1]):
            if dense_belief[i][j] == 1:
                if i == 0 or dense_belief[i-1][j] == 0:
                    edges.append((i, j))
                elif i == dense_belief.shape[0]-1 or dense_belief[i+1][j] == 0:
                    edges.append((i, j))
                elif j == 0 or dense_belief[i][j-1] == 0:
                    edges.append((i, j))
                elif j == dense_belief.shape[1]-1 or dense_belief[i][j+1] == 0:
                    edges.append((i, j))

    suppress_gain = np.zeros_like(dense_belief)
    edges = np.array(edges)
    for loc in edges:
        if belief[(loc[0], loc[1])][1] > 0.5:
            suppress_gain[loc[0], loc[1]] = 1

    return suppress_gain + offset
