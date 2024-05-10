# This file represent configuration settings and constants in the project
import pathlib

# Paths
PROJECT_DIR = pathlib.Path(__file__).parent.parent.absolute()
RESULT_DIR = pathlib.Path("/media/banana/data/results/udacity-gym")
CHECKPOINT_DIR = pathlib.Path("/media/banana/data/models/udacity-gym")
LOG_DIR = pathlib.Path("/media/banana/data/logs/udacity-gym")

# Device settings
ACCELERATOR = "gpu"  # choose between gpu or cpu
DEVICE = 1
DEFAULT_DEVICE = f'cuda:{DEVICE}' if ACCELERATOR == 'gpu' else 'cpu'
