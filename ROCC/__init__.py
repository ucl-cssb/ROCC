import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# file path for fitted_Q_agents
FQ_DIR = os.path.dirname(os.path.abspath(__file__))
FQ_DIR = os.path.join(FQ_DIR, 'fitted_Q_iteration')
sys.path.append(FQ_DIR)

# file path for chemostat_env
C_DIR = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.join(C_DIR, 'chemostat_env')
sys.path.append(C_DIR)


from chemostat_envs import *
from fitted_Q_agents import *
from reward_func import *
