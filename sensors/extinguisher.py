import numpy as np


class Extinguisher(object):
    def __init__(self, suppress_size, delta_beta) -> None:
        self.suppress_size = suppress_size
        self.delta_beta = delta_beta

    def get_control(self, position, simulation):
        control_field = np.zeros(simulation.dims)
        state = simulation.dense_state()
        r0, c0 = position

        # center image at UAV position
        half_row = (self.suppress_size-1)//2
        half_col = (self.suppress_size-1)//2
        for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
            for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
                r = r0 + dr
                c = c0 + dc

                if 0 <= r < state.shape[0] and 0 <= c < state.shape[1]:
                    if simulation.group[(r, c)].is_on_fire(simulation.group[(r, c)].state):
                        control_field[r, c] = 1

        return control_field
