from simulators.fires.LatticeForest import LatticeForest
# from simulators.Cell2Fire.Cell2Forest import Cell2Forest
import numpy as np
import itertools
import random


def get_environment(args):
    if args.env == 1:
        initial_fire, fire_centers = set_initial_fire(args.dimension, args.fire_size, args.fire_num)
        env = LatticeForest(args.dimension, args.seed, initial_fire, args.alpha, args.beta)
    elif args.env == 2:
        # env = Cell2Forest(fire_map="Harvest40x40", save_name=args.save_name, strategy=args.strategy, update_hrs=args.update_hrs)
        # remove the two lines if one want to use the default ignition points
        # initial_fire = set_initial_fire(env.dims, args.fire_size, args.fire_num)
        # env.dataReader.spawn() #! will this work?
        # env.rewrite_initial_fire(initial_fire)
        # env.initialize_fire()
        
        env = Cell2Forest(fire_map="Harvest40x40", save_name=args.save_name, strategy=args.strategy, update_minutes=args.update_minutes, ros_cv=args.ros_cv, seed=args.seed)
        initial_fire = set_initial_fire(env.dims, args.fire_size, args.fire_num)
        #env.start_fire(initial_fire)
        env.start_fire(initial_fire)

    # elif args.env == 3:
    #     env = Cell2Forest(fire_map="Sub40x40", save_name=args.save_name, strategy=args.strategy, update_minutes=args.update_minutes)
    #     # remove the two lines if one want to use the default ignition points
    #     # initial_fire = set_initial_fire(env.dims, args.fire_size, args.fire_num)
    #     # env.rewrite_initial_fire(initial_fire)
    #     env.start_fire()
    else:
        raise ValueError("Invalid environment selection.")
    return env, fire_centers


# def get_cell_environment(args, fire_map="Harvest40x40"):
#     env = Cell2Forest(fire_map)
#     # remove the two lines if one want to use the default ignition points
#     # initial_fire = set_initial_fire(env.dims, args.fire_size, args.fire_num)
#     # env.rewrite_initial_fire(initial_fire)
#     env.initialize_fire()
#     return env


def set_initial_fire(dim, fire_size, fire_num):
    dims = (dim, dim) if isinstance(dim, int) else dim
    overlap_thre = 2 #!!!!!!!!!
    # overlap_thre = 10 #!!!!!!!!!
    
    size_dict = {2: (-1, 1),
                 3: (-1, 2),
                 4: (-1, 3),
                 5: (-2, 3),
                 6: (-2, 4),
                 7: (-3, 4),
                 8: (-3, 5)}

    fires = []
    if fire_num == 1:
        if fire_size in size_dict:
            left, right = size_dict[fire_size]
        else:
            raise ValueError

        r_center = np.floor((dims[0] - 1) / 2).astype(np.uint8) #!!!!
        c_center = np.floor((dims[1] - 1) / 2).astype(np.uint8)
        
        # random center
        # start_coords = [(i, j) for i in range(dims[0] - 10) for j in range(dims[1] - 10)]
        # random.shuffle(start_coords)
        # r_center = start_coords[0][0]
        # c_center = start_coords[0][1]
        # r_center = 15
        # c_center = 15
        

        delta_r = [0] if dims[0] < 4 else [k for k in range(left, right)]
        delta_c = [0] if dims[1] < 4 else [k for k in range(left, right)]
        deltas = itertools.product(delta_r, delta_c)

        for (dr, dc) in deltas:
            r, c = r_center + dr, c_center + dc
            fires.append((r, c))
            # group[(r, c)].set_on_fire()
        selected_coords = [(r_center, c_center)]

    else:
        start_coords = [(i, j) for i in range(dims[0] - fire_size) for j in range(dims[1] - fire_size)]#!!!!!!
        # start_coords = [(i, j) for i in range(dims[0] - 10) for j in range(dims[1] - 10)]
        
        random.shuffle(start_coords)
        selected_coords = [start_coords[0]]

        delta_r = [k for k in range(0, fire_size)]
        delta_c = [k for k in range(0, fire_size)]
        deltas = itertools.product(delta_r, delta_c)

        for (dr, dc) in deltas:
            r, c = start_coords[0][0] + dr, start_coords[0][1] + dc
            fires.append((r, c))

        # Step 4: select subsequent non-overlapping patches
        for coord in start_coords[1:]:
            # Check if patch overlaps with any previously selected patches
            overlaps = False
            for sel_coord in selected_coords:
                if np.abs(sel_coord[0] - coord[0]) <= overlap_thre and np.abs(sel_coord[1] - coord[1]) <= overlap_thre:
                    overlaps = True
                    break
            if not overlaps:
                deltas = itertools.product(delta_r, delta_c)
                for (dr, dc) in deltas:
                    r, c = coord[0] + dr, coord[1] + dc
                    fires.append((r, c))
                selected_coords.append(coord)
            # Stop when enough non-overlapping patches have been selected
            if len(fires) == fire_num * fire_size * fire_size:
                break
        
        print(f'selected initial coords: {selected_coords}' )
        
    return fires, selected_coords


def measure_model(element, state, observation, measure_correct):
    """
    Measurement model describing the likelihood function p(observation | state).

    :param element: a Tree object from the LatticeForest simulator.
    :param state: state value, from Tree state space, to determine probability.
    :param observation: observation value, from Tree state space, to determine probability.

    :return: the probability value p(observation | state).
    """
    measure_wrong = (1 / (len(element.state_space) - 1)) * (1 - measure_correct)

    if state != observation:
        return measure_wrong
    elif state == observation:
        return measure_correct
