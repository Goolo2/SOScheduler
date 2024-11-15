import math
import numpy as np
from experiments import environments


class Camera(object):
    def __init__(self, uncertainty, image_size, measure_correct) -> None:
        self.uncertainty = uncertainty
        self.image_size = image_size
        self.measure_correct = measure_correct

    def get_image(self, position, simulation):
        '''
        Helper function to return an image of the LatticeForest for a UAV, with or without uncertainty.

        :param uav: a UAV object, where the UAV location in (row, col) units is used as the image center.
        :param simulation: a LatticeForest simulation object, whose underlying state is used to generate the image.
        :param config: Config class object containing the image size and measurement model parameters.
        :param uncertainty: a boolean indicating whether or not the true underlying state should be returned.

        :return: a tuple of a 2D numpy array and a dictionary. the numpy array, image, contains the Tree state values
        describing the image. the dictionary consists of {Tree position: observation} key value pairs if uncertainty = True,
        otherwise the dictionary is empty.
        '''
        state = simulation.dense_state()
        r0, c0 = position
        image = np.zeros((self.image_size, self.image_size)).astype(np.int8)
        observation = dict()

        # center image at UAV position
        half_row = (self.image_size-1)//2
        half_col = (self.image_size-1)//2
        for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
            for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
                r = r0 + dr
                c = c0 + dc

                if 0 <= r < state.shape[0] and 0 <= c < state.shape[1]:
                    if not self.uncertainty:
                        image[ri, ci] = state[r, c]
                    # sample according to the measurement model to generate an uncertain image
                    else:
                        element = simulation.group[(r, c)]
                        probs = [environments.measure_model(element, element.state, o, self.measure_correct) for o in element.state_space]
                        obs = np.random.choice(element.state_space, p=probs)
                        observation[(r, c)] = obs
                        image[ri, ci] = obs

        return image, observation


