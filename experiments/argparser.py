import configargparse
import yaml
import sys

def parse_arguments(verbose=False):
    parser = configargparse.ArgParser()
    parser.add_argument("--config",
                        required=False,
                        is_config_file=True,
                        # default='./simulations/configs/FNUM_HEUR.yaml',
                        default='./testbed/configs/MSCIDC.yaml',
                        help="Configuration file path.")
    # Experiment settings
    parser.add_argument("--strategy",
                        type=str,
                        # required=True,
                        default="MSCIDC",
                        help="Sampling strategy.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--verbose",
                        type=bool,
                        default=False,
                        help="Printing details.")
    parser.add_argument("--env",
                        type=int,
                        default=1,
                        help="Simulator selection, 1=grid, 2=cell.")
    parser.add_argument("--dimension",
                        type=int,
                        default=25,
                        help="Environment dimension.")
    parser.add_argument("--ros_cv",
                        type=float,
                        default=0.5,
                        help="Rate of Spread.")
    parser.add_argument("--fire_size",
                        type=int,
                        default=4,
                        help="Initial fire area dimension.")
    parser.add_argument("--fire_num",
                        type=int,
                        default=1,
                        help="Initial fire number.")
    parser.add_argument("--windx",
                        type=float,
                        default=0.5,
                        help="Wind magnitude in x-aixs direction.")
    parser.add_argument("--windy",
                        type=float,
                        default=0.5,
                        help="Wind magnitude in y-aixs direction.")
    parser.add_argument("--update_interval",
                        type=int,
                        default=2,
                        help="Simulation update interval.")
    parser.add_argument("--update_minutes",
                        type=int,
                        default=4,
                        help="minutes per update")
    parser.add_argument("--total_iterations",
                        type=int,
                        default=100,
                        help="Simulation time limits in seconds.")
    parser.add_argument("--alpha",
                        type=float,
                        default=0.2763,
                        help="Fire model parameter.")
    parser.add_argument("--beta",
                        type=float,
                        default=0.90483,
                        help="Fire model parameter.")
    parser.add_argument("--delta_beta",
                        type=float,
                        default=0.15/0.2763,
                        help="Fire suppression ability.")
    parser.add_argument("--num_robot",
                        type=int,
                        default=1,
                        help="Number of robots.")
    parser.add_argument("--horizon",
                        type=int,
                        default=8,
                        help="Planning horizon of robot.")
    parser.add_argument("--communication",
                        type=int,
                        default=1,
                        help="Communication interval.")
    parser.add_argument("--image_size",
                        type=int,
                        default=3,
                        help="Field of view of camera.")
    parser.add_argument("--suppress_size",
                        type=int,
                        default=3,
                        help="Field of view of fire extinguisher.")
    parser.add_argument("--uncertainty",
                        type=bool,
                        default=True,
                        help="Whether or not observe the true state.")
    parser.add_argument("--suppress",
                        action='store_true',
                        help="Whether or not suppress the fire.")
    parser.add_argument("--control_rate",
                        type=float,
                        default=10.0,
                        help="Control update rate.")
    parser.add_argument("--sensing_rate",
                        type=float,
                        default=10.0,
                        help="Sensing data update rate.")
    parser.add_argument("--measure_correct",
                        type=float,
                        default=0.95,
                        help="Probability of correct measurement for camera observation model.")
    parser.add_argument("--threshold",
                        type=float,
                        default=1e-100,
                        help="The minimum non-zero probability for belief update.")
    parser.add_argument("--weight",
                        type=float,
                        default=0.1,
                        help="The weight coefficient for mix strategy.")
    parser.add_argument("--capacity",
                        type=int,
                        default=16,
                        help="Fire suppress capacity for each robot.")
    parser.add_argument("--filter_alpha",
                        type=float,
                        default=0.2763,
                        help="Fire model parameter for filter.")
    parser.add_argument("--filter_beta",
                        type=float,
                        default=0.90483,
                        help="Fire model parameter for filter.")
    parser.add_argument('--movements', type=list, default=[(-1, -1), (-1, 0), (-1, 1), (0, -1),
                        (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)], required=False, help='Movements choice.')
    parser.add_argument("--save_dir",
                        type=str,
                        default="/home/xuecheng/projects/firehunter/testbed/output/MSCIDC/UI_2",
                        help="Directory for logs.")
    parser.add_argument("--ckpt_dir",
                        type=str,
                        default="/home/xuecheng/projects/firehunter/baselines/ddrl/madqn-07-Jul-2023-2028.pth.tar",
                        help="Directory for logs.")
    args = parser.parse_args()

    if args.verbose:
        print(parser.format_values())
        
    if sys.platform == 'linux':
        print("当前系统是Linux")
    elif sys.platform == 'win32':
        print("当前系统是Windows")
    
    # args.save_name = f'SEED_{args.seed}_E{args.env}_D{args.dimension}_FS{args.fire_size}_FN{args.fire_num}_UI{args.update_interval}_UM{args.update_minutes}_I{args.image_size}_S{args.suppress_size}_N{args.num_robot}_C{args.capacity}'
    args.save_name = f'SEED_{args.seed}_E{args.env}_D{args.dimension}_FS{args.fire_size}_FN{args.fire_num}_UI{args.update_interval}_UM{args.update_minutes}_I{args.image_size}_S{args.suppress_size}_N{args.num_robot}_C{args.capacity}_A{args.alpha:.2f}'
        
    return args
