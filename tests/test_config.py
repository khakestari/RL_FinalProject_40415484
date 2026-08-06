"""
Unit tests for configuration
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.configs.config import *


def test_student_id():
    """Test student ID is correct"""
    assert STUDENT_ID == "40415484"


def test_seed():
    """Test seed is correctly calculated"""
    assert SEED == 8


def test_maze_size():
    """Test maze size is correctly calculated"""
    # Formula: N = 15 + (b mod 4), where b = SEED = 8
    # N = 15 + (8 mod 4) = 15 + 0 = 15
    assert MAZE_SIZE == 15


def test_actions():
    """Test actions are defined correctly"""
    assert len(ACTIONS) == 4
    assert NUM_ACTIONS == 4
    assert 'UP' in ACTIONS
    assert 'DOWN' in ACTIONS
    assert 'LEFT' in ACTIONS
    assert 'RIGHT' in ACTIONS


def test_transition_probabilities():
    """Test transition probabilities sum to 1"""
    total_prob = INTENDED_ACTION_PROB + 2 * PERPENDICULAR_ACTION_PROB
    assert abs(total_prob - 1.0) < 1e-6


def test_reward_structure():
    """Test reward structures are defined"""
    assert 'sparse' in REWARDS
    assert 'shaped' in REWARDS
    assert 'goal' in REWARDS['sparse']
    assert 'key' in REWARDS['sparse']


def test_max_steps():
    """Test max steps per episode"""
    expected_max_steps = 3 * (MAZE_SIZE * MAZE_SIZE)
    assert MAX_STEPS_PER_EPISODE == expected_max_steps


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
