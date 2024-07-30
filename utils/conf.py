# This file represent configuration settings and constants in the project
import pathlib

# Paths
PROJECT_DIR = pathlib.Path(__file__).parent.parent.absolute()
RESULT_DIR = pathlib.Path("/media/banana/data/results/udacity-gym")  # TODO: remove hardcoded path
CHECKPOINT_DIR = pathlib.Path("/media/banana/data/models/udacity-gym")  # TODO: remove hardcoded path
LOG_DIR = pathlib.Path("/media/banana/data/logs/udacity-gym")  # TODO: remove hardcoded path

# Device settings
ACCELERATOR = "gpu"  # choose between gpu or cpu
DEVICE = 0  # if multiple gpus are available
DEFAULT_DEVICE = f'cuda:{DEVICE}' if ACCELERATOR == 'gpu' else 'cpu'

# TODO: add code to override default settings
