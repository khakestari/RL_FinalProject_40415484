"""
RL Agents module
"""

# Import agents as they are implemented
try:
    from .value_iteration import ValueIteration
except ImportError:
    ValueIteration = None

try:
    from .q_learning import QLearning
except ImportError:
    QLearning = None

try:
    from .sarsa_lambda import SarsaLambda
except ImportError:
    SarsaLambda = None

__all__ = ['ValueIteration', 'QLearning', 'SarsaLambda']
