import logging
import random
import time
from copy import deepcopy
import numpy as np

import experiments
import robots
import sensors
import strategies
import teams


def get_init_belief(env):
    initial_belief = dict()
    for key in env.group.keys():
        element = env.group[key]
        # exact belief
        initial_belief[key] = np.zeros(len(element.state_space))
        initial_belief[key][element.state] = 1
    return initial_belief


def get_init_position(num_robot):
    init_square_size = np.ceil(np.sqrt(num_robot)).astype(int)
    initial_positions = []
    for i in range(num_robot):
        position = np.unravel_index(i, (init_square_size, init_square_size), order='C')
        initial_positions.append(position)

    return initial_positions


def get_sensor(args):
    camera = sensors.camera.Camera(
        # rate=args.sensing_rate,
        uncertainty=args.uncertainty,
        image_size=args.image_size,
        measure_correct=args.measure_correct
    )
    extinguisher = sensors.extinguisher.Extinguisher(
        # rate=args.sensing_rate,
        suppress_size=args.suppress_size,
        delta_beta=args.delta_beta
    )
    return camera, extinguisher


def get_robot(args, label, fire_center, initial_belief, initial_position, camera, extinguisher):
    robot = robots.UAV(
        label=label,
        fire_center=fire_center,
        init_pos=initial_position,
        image_size=args.image_size,
        belief=initial_belief,
        horizon=args.horizon,
        camera=camera,
        extinguisher=extinguisher,
        capacity=args.capacity,
    )
    return robot


def get_multi_robots(args, dims, fire_centers, initial_belief, initial_positions, camera, extinguisher):
    robots = {}
        
    robot_fire_dict = experiments.utilities.allocate_robots_to_fires(args.fire_num, args.num_robot)
    for i in range(args.num_robot):
        robot = get_robot(args, i+1, fire_centers[robot_fire_dict[i]], deepcopy(initial_belief), initial_positions[i], camera, extinguisher)
        robots[i+1] = robot

    team = teams.UAVTeam(args, robots)
    return team


def get_strategy(args):
    if args.strategy == 'confentropy':
        return strategies.ConfEntropy(args)
    elif args.strategy == 'nocontrol':
        return strategies.NoControl(args)
    else:
        logging.error(f"{args.strategy} Not Implemented! ")


def get_evaluator():
    evaluator = experiments.Evaluator()
    return evaluator


def run(args, env, team, strategy, evaluator, logger):
    # initial log
    delta_p_time = 0
    coverage, accuracy, healthy, onfire, burnt = evaluator.eval_results(env, team)
    logger.append(0, env.dense_state(), team, coverage, accuracy, healthy, onfire, burnt)
    
    for t in range(1, args.total_iterations+1):
        print(f'simulationTime = {t}/{args.total_iterations}')

        if t > 1 and (t-1) % args.update_interval == 0:
            env.update(team.team_control)

        team.sense(env)

        advance = False
        if t > 1 and (t-1) % args.update_interval == 0:
            advance = True
        team.perception(env, advance)

        coverage, accuracy, healthy, onfire, burnt = evaluator.eval_results(env, team)
        print(f'healthy={healthy}')
        logger.append(t, env.dense_state(), team, coverage, accuracy, healthy, onfire, burnt)
        logger.append_time(delta_p_time)
        
        start_p_time = time.time()
        if (t-1) % args.communication == 0:
            strategy.plan(env, team.robots, team.team_belief)
        end_p_time = time.time()
        delta_p_time = end_p_time - start_p_time
        
        team.navigation()
        
        if env.end:
            print(f"Simulation ends at {t} steps")
            break


def main():
    args = experiments.argparser.parse_arguments()
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    env, fire_centers = experiments.environments.get_environment(args)
    logger = experiments.Logger(args)
    initial_belief = get_init_belief(env)
    initial_positions = get_init_position(args.num_robot)
    camera, extinguisher = get_sensor(args)
    team = get_multi_robots(args, env.dims, fire_centers, initial_belief, initial_positions, camera, extinguisher)
    strategy = get_strategy(args)
    evaluator = get_evaluator()

    print("Wrong dimension, fire size for now")

    start = time.time()
    run(args, env, team, strategy, evaluator, logger)
    end = time.time()
    logger.save(end-start)  

    experiments.utilities.print_metrics(logger)
    print(f"Time used: {end - start:.1f} seconds")


if __name__ == "__main__":
    main()  