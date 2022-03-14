import configargparse

def str2bool(inp):
    return inp.lower() in 'true'

def config_parser(is_optimization = False):
    parser = configargparse.ArgumentParser()

    # Experiment Setup
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--dataset_root', type=str, default='/your/dataset/root/datasets', help="data folder's directory path")
    parser.add_argument("-fn", "--data_name", type=str, default="test")

    if is_optimization:
        # scene parameters
        parser.add_argument("--radius", type=float, default=0.6)
        parser.add_argument("-PS", "--plane_thresh", type=float, default=0.01)
        parser.add_argument("--window_frames", type=int, default=50, help="window of the frame to be used")

        # Optimization parameters - global
        parser.add_argument("--iterations", type=int, default=100)
        parser.add_argument("--learn_rate", type=float, default=0.001)
        parser.add_argument("--opt_vars", type=str, default="trans_allpose",
                            help="trans|trans_allpose|trans_glob")
        parser.add_argument("--wt_ft_vel", type=float, default=300)
        parser.add_argument("--wt_ft_cont", type=float, default=300)
        parser.add_argument("--wt_rot_smth", type=float, default=300)
        parser.add_argument("--wt_trans_imu_smth", type=float, default=300)
        parser.add_argument("--wt_pose_prior", type=float, default=2000)

        parser.add_argument("--imu_smt_mode", type=str, default='XYZ', help ='optmization mode use XY or XYZ')

        # Optimization parameters - connecting
        parser.add_argument("--init_vel_weight", type=float, default=10)
        parser.add_argument("--init_trans_weight", type=float, default=400)
        parser.add_argument("--init_rot_weight", type=float, default=400)

    return parser


def get_config(is_optimized = False):
    parser = config_parser(is_optimized)
    cfg = parser.parse_args()

    return cfg
