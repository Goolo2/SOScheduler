import numpy as np


class Evaluator:
    def __init__(self) -> None:
        pass

    def eval_results(self, env, team):
        coverage = self.compute_coverage(env, team.robots)
        accuracy = self.compute_accuracy(env, team.team_belief)

        healthy, on_fire, burnt = self.compute_ratio(env.dense_state())

        return coverage, accuracy, healthy, on_fire, burnt
    

    def compute_coverage(self, env, robots):
        state = env.dense_state()
        if len(env.fires) == 0:
            return 1

        team_observation = set()
        for robot in robots.values():
            _, observation = robot.camera.get_image(robot.position, env)
            robot_observation = {key for key in observation.keys() if state[key[0], key[1]] == 1}
            team_observation |= robot_observation

        return len(team_observation)/len(env.fires)

    def compute_accuracy(self, env, belief):
        state = env.dense_state()
        accuracy = 0
        for key in belief.keys():
            if np.argmax(belief[key]) == state[key[0], key[1]]:
                accuracy += 1

        return accuracy/(env.dims[0]*env.dims[1])

    def compute_ratio(self, state):
        healthy = np.sum(state==0) / (state.shape[0]*state.shape[1])
        on_fire = np.sum(state==1) / (state.shape[0]*state.shape[1])
        burnt = np.sum(state==2) / (state.shape[0]*state.shape[1])
        return healthy, on_fire, burnt

    def compute_coverage_ddrl(self, env, robots, image_dims):
        state = env.dense_state()
        if len(env.fires) == 0:
            return 1

        team_observation = set()
        for robot in robots.values():
            robot_observation = set()
            _, observation = robot.camera.get_image(robot.position, env)
            
            
            half_row = (image_dims[0]-1)//2
            half_col = (image_dims[1]-1)//2

            for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
                for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
                    r = robot.position[0] + dr
                    c = robot.position[1] + dc                
                    if 0 <= r < state.shape[0] and 0 <= c < state.shape[1] and state[r, c] == 1:
                        robot_observation.add((r,c))
                
            team_observation |= robot_observation

        return len(team_observation)/len(env.fires)
    
    def eval_results_ddrl(self, env, team, image_dims):
        coverage = self.compute_coverage_ddrl(env, team.robots, image_dims)
        accuracy = 1.0

        healthy, on_fire, burnt = self.compute_ratio(env.dense_state())

        return coverage, accuracy, healthy, on_fire, burnt