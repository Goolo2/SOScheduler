from collections import defaultdict
import numpy as np
from experiments import environments


class BayesianFilter():
    def __init__(self, args) -> None:
        self.measure_correct = args.measure_correct
        self.threshold = args.threshold
        
        # # use ground truth paratemers
        # self.alpha = args.alpha
        # self.beta = args.beta
        
        # use the prior parameters
        self.alpha = args.filter_alpha
        self.beta = args.filter_beta
        
        self.healthy = 0
        self.on_fire = 1
        self.burnt = 2


    def update_belief(self, simulation_group, prior, advance, observation, measure_correct, threshold, control=None):
        """
        Update the belief given an observation for the LatticeForest simulator using an approximate filtering scheme.

        :param simulation_group: dictionary containing {Tree position: Tree object} key value pairs for the LatticeForest
        simulator. this is used to iterate through each element and update the corresponding belief.
        :param prior: a dictionary describing the prior belief, as {Tree position: list of probabilities for
        each state value}.
        :param advance: boolean indicating whether or not the LatticeForest is updated at the current time step.
        :param observation: dictionary describing the belief, as {Tree position: tree_observation} key value pairs.
        the value tree_observation may be an integer (single observation) or a list of integers (several observations of
        the same tree at a single time step).
        :param config: a Config class object containing the measurement model parameters.
        :param control: a dictionary describing the control effort applied at the current time step, as
        {Tree position: (delta_alpha, delta_beta)} key value pairs. this parameter defaults to None, indicating no control
        effort.

        :return: the updated belief as a dictionary of {Tree position: list of probabilities for each time step} key value
        pairs.
        """
        if control is None:
            control = defaultdict(lambda: (0, 0))

        posterior = dict()
        for key in simulation_group.keys():
            element = simulation_group[key]
            element_posterior = np.zeros(len(element.state_space))
            num_neighbors = len(element.neighbors)
            on_fire = 1

            #! Step1 the Tree dynamics are based on the number of neighbors on fire rather than the identity of the neighboring
            #! Trees. as a result, iterate over the possible number of Trees on fire to consider different state transitions.
            caf = np.zeros(num_neighbors+1)
            for state_idx in range(2**num_neighbors):
                xj = np.base_repr(state_idx, base=2).zfill(num_neighbors)
                active = xj.count('1')

                values = []
                for n in range(num_neighbors):
                    neighbor_key = element.neighbors[n]
                    prob = None
                    if int(xj[n]) == 0:
                        prob = 1 - prior[neighbor_key][on_fire]
                    elif int(xj[n]) == 1:
                        prob = prior[neighbor_key][on_fire]

                    values.append(prob)

                caf[active] += self.multiply_probabilities(values, threshold)

            #! Step2: perform open-loop dynamics update
            for x_t in element.state_space:
                for x_tm1 in element.state_space:
                    # if the simulator is updated, then consider possible state transitions
                    if advance:
                        for active in range(num_neighbors+1):
                            # values = [element.dynamics((x_tm1, active, x_t), control[key]), caf[active], prior[key][x_tm1]]
                            values = [self.dynamics_exponential((x_tm1, active, x_t), control[key]), caf[active], prior[key][x_tm1]]
                            element_posterior[x_t] += self.multiply_probabilities(values, threshold)
                    # otherwise, the dynamics are static
                    else:
                        element_posterior[x_t] += (x_t == x_tm1)*prior[key][x_tm1]

            #! Step3: adjust dynamics update based on observation(s)
            for x_t in element.state_space:
                if key in observation.keys():
                    # check for single observation or multiple observations
                    if type(observation[key]) == int:
                        element_posterior[x_t] *= environments.measure_model(element, x_t, observation[key], measure_correct)
                    elif type(observation[key]) == list:
                        element_posterior[x_t] *= np.prod([environments.measure_model(element, x_t, obs, measure_correct)
                                                           for obs in observation[key]])

            posterior[key] = element_posterior/np.sum(element_posterior)

        return posterior

    def dynamics_exponential(self, state_and_next_state, control=(0, 0)):
        """
        Implementation of transition distribution. The transition from healty to on fire
        is an exponential function of the number of neighbors on fire.
        """
        state, number_neighbors_on_fire, next_state = state_and_next_state
        delta_alpha, delta_beta = control

        if state == self.healthy:
            if next_state == self.healthy:
                return (1 - self.alpha + delta_alpha)**number_neighbors_on_fire
            elif next_state == self.on_fire:
                return 1 - (1 - self.alpha + delta_alpha)**number_neighbors_on_fire
            else:
                return 0

        elif state == self.on_fire:
            if next_state == self.healthy:
                return 0
            elif next_state == self.on_fire:
                return self.beta - delta_beta
            elif next_state == self.burnt:
                return 1 - self.beta + delta_beta

        else:
            if next_state is self.burnt:
                return 1
            else:
                return 0

    def multiply_probabilities(self, values, threshold):
        """
        Helper function to multiply a list of probabilities.

        :param values: an iterable object containing the probabilities to multiply.
        :param threshold: the minimum non-zero probability.

        :return: product of probabilities, which is zero if any value is below a specified threshold.
        """

        if any([v < threshold for v in values]):
            return 0
        else:
            sum_log = sum([np.log(v) for v in values])
            if sum_log <= np.log(threshold):
                return 0
            else:
                return np.exp(sum_log)
