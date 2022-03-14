import sys
import os
sys.path.append(os.path.dirname(os.path.split(os.path.abspath( __file__))[0]))
from configs.config_loader import config_parser
from tools import bvh_tool

def get_config(is_optimized = False):
    parser = config_parser(is_optimized)
    cfg = parser.parse_args()
    return cfg
