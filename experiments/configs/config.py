"""
Configuration file for RL Final Project
Student ID: 40415484
"""

STUDENT_ID = "40415484"
SEED = 8  # Second to last digit of student ID

MAZE_SIZE = 15 + (SEED % 4) 
MIN_WALL_PERCENTAGE = 0.15
MIN_PENALTY_CELLS = 5

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
NUM_ACTIONS = 4

INTENDED_ACTION_PROB = 0.8
PERPENDICULAR_ACTION_PROB = 0.1 # احتمال انحراف به هر جهت عمود

# Reward Structure
REWARDS = {
    'sparse': {
        'step': -0.1,           # Small cost per step
        'wall_collision': -1.0,  # Collision penalty
        'penalty_cell': -5.0,    # Penalty cell
        'key': 50.0,             # Getting the key
        'locked_door': -2.0,     # Trying locked door
        'goal': 100.0,           # Reaching goal
    },
    'shaped': {
        'step': -0.1,
        'wall_collision': -1.0,
        'penalty_cell': -5.0,
        'key': 50.0,
        'locked_door': -2.0,
        'goal': 100.0,
        'closer_to_key': 0.5,    # Reward shaping: moving closer to key
        'closer_to_goal': 0.5,   # Reward shaping: moving closer to goal (after key)
        'safe_navigation': 0.2,  # Avoiding penalty cells
    }
}

# Episode Configuration
MAX_STEPS_PER_EPISODE = 3 * (MAZE_SIZE * MAZE_SIZE)

# Value Iteration Parameters
VI_GAMMA = 0.99          # Discount factor
VI_THETA = 1e-6          # Convergence threshold
VI_MAX_ITERATIONS = 1000

# Q-Learning Parameters
QL_ALPHA = 0.1           # Learning rate
QL_GAMMA = 0.99          # Discount factor
QL_EPSILON_START = 1.0   # Initial exploration rate
QL_EPSILON_END = 0.01    # Final exploration rate
QL_EPSILON_DECAY = 0.995 # Epsilon decay (exponential)
QL_EPISODES = 5000       # Number of training episodes

# SARSA(λ) Parameters
SARSA_ALPHA = 0.1        # Learning rate
SARSA_GAMMA = 0.99       # Discount factor
SARSA_EPSILON_START = 1.0
SARSA_EPSILON_END = 0.01
SARSA_EPSILON_DECAY = 0.995
SARSA_LAMBDA_VALUES = [0.0, 0.3, 0.7, 0.9]  # Different λ values to test
SARSA_EPISODES = 5000
SARSA_TRACE_TYPE = 'accumulating'  # or 'replacing'

# Transfer Learning Parameters
TL_SOURCE_EPISODES = 5000
TL_TARGET_EPISODES = 3000
TL_BETA_VALUES = [0.25, 0.50, 0.75]  # Transfer scaling factors
TL_SIMILAR_CHANGE_PERCENTAGE = 0.15  # 15-20% obstacles change
TL_DIFFERENT_CHANGE_PERCENTAGE = 0.35  # 35%+ obstacles change

# Evaluation Parameters
EVAL_EPISODES = 100
EVAL_MAX_STEPS = MAX_STEPS_PER_EPISODE

# Visualization Parameters
CELL_SIZE = 40  # pixels
FPS = 10        # Animation speed

# File Paths
MAPS_DIR = "environments/maps"
RESULTS_DIR = "results"
RAW_DATA_DIR = "results/raw_data"
MODELS_DIR = "results/models"
FIGURES_DIR = "results/figures"
VIDEOS_DIR = "results/videos"

# Logging
LOG_EVERY = 100  # Log every N episodes
SAVE_EVERY = 500  # Save model every N episodes

# Additional Feature Selection
# Choose one: 'limited_energy', 'slippery_cells', 'teleporter', 'moving_obstacle', 'periodic_gate'
ADDITIONAL_FEATURE = 'limited_energy'

# Feature-specific parameters
LIMITED_ENERGY_MAX = MAZE_SIZE * MAZE_SIZE  # Maximum energy
LIMITED_ENERGY_COST = 1  # Energy cost per step
LIMITED_ENERGY_RECHARGE = 20  # Energy gained from recharge stations

# Colors for GUI (RGB)
COLORS = {
    'wall': (50, 50, 50),
    'empty': (255, 255, 255),
    'start': (0, 255, 0),
    'key': (255, 215, 0),
    'door_closed': (139, 69, 19),
    'door_open': (222, 184, 135),
    'goal': (255, 0, 0),
    'penalty': (255, 140, 0),
    'agent': (0, 0, 255),
    'path': (200, 200, 255),
    'visited': (240, 240, 240),
}
