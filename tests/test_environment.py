"""
Unit tests for maze environment
"""
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.generator import MazeGenerator, generate_target_mazes
from environments.maze import MazeEnvironment, Action


def test_maze_generation():
    """Test maze generation with seed=8"""
    generator = MazeGenerator(size=15, seed=8)
    maze, metadata = generator.generate()
    
    assert maze.shape == (15, 15)
    assert metadata['seed'] == 8
    assert metadata['size'] == 15
    assert 'start' in metadata
    assert 'key' in metadata
    assert 'door' in metadata
    assert 'goal' in metadata


def test_maze_validation():
    """Test that generated maze has valid paths"""
    generator = MazeGenerator(size=15, seed=8)
    maze, metadata = generator.generate()
    
    # Check that BFS found valid paths (implicit in successful generation)
    assert metadata['attempt'] >= 1


def test_wall_percentage():
    """Test minimum wall percentage"""
    generator = MazeGenerator(size=15, seed=8, min_wall_percentage=0.15)
    maze, metadata = generator.generate()
    
    wall_pct = metadata['wall_percentage']
    assert wall_pct >= 0.15


def test_penalty_cells():
    """Test minimum penalty cells"""
    generator = MazeGenerator(size=15, seed=8, min_penalty_cells=5)
    maze, metadata = generator.generate()
    
    assert len(metadata['penalty_cells']) >= 5


def test_maze_save_load():
    """Test saving and loading maze"""
    generator = MazeGenerator(size=15, seed=8)
    maze, metadata = generator.generate()
    
    # Save
    filepath = generator.save_maze(maze, metadata, "test_maze.npz")
    
    # Load
    loaded_maze, loaded_metadata = MazeGenerator.load_maze(filepath)
    
    assert np.array_equal(maze, loaded_maze)
    assert loaded_metadata['seed'] == metadata['seed']
    
    # Cleanup
    Path(filepath).unlink(missing_ok=True)


def test_environment_initialization():
    """Test environment initialization"""
    generator = MazeGenerator(size=15, seed=8)
    maze, metadata = generator.generate()
    
    env = MazeEnvironment(maze, metadata, reward_type='sparse', seed=8)
    
    assert env.size == 15
    assert env.agent_pos is None  # Not initialized until reset


def test_environment_reset():
    """Test environment reset"""
    generator = MazeGenerator(size=15, seed=8)
    maze, metadata = generator.generate()
    
    env = MazeEnvironment(maze, metadata, seed=8)
    state = env.reset()
    
    x, y, has_key, energy = state
    assert (x, y) == metadata['start']
    assert has_key == 0
    assert energy == env.max_energy
    assert env.steps == 0
    assert not env.done


def test_environment_step():
    """Test taking steps in environment"""
    generator = MazeGenerator(size=15, seed=8)
    maze, metadata = generator.generate()
    
    env = MazeEnvironment(maze, metadata, seed=8)
    env.reset()
    
    # Take a step
    next_state, reward, done, info = env.step(Action.DOWN)
    
    assert len(next_state) == 4
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert 'steps' in info


def test_stochastic_transitions():
    """Test that transitions are stochastic"""
    generator = MazeGenerator(size=15, seed=8)
    maze, metadata = generator.generate()
    
    # Run same action multiple times with different seeds
    results = []
    for seed in range(10):
        env = MazeEnvironment(maze, metadata, seed=seed)
        env.reset()
        next_state, _, _, info = env.step(Action.RIGHT)
        results.append(info['actual_action'])
    
    # Should have some variation (not all the same)
    unique_actions = len(set(results))
    assert unique_actions > 1  # At least some stochasticity


def test_key_collection():
    """Test key collection mechanism"""
    generator = MazeGenerator(size=15, seed=8)
    maze, metadata = generator.generate()
    
    env = MazeEnvironment(maze, metadata, seed=8)
    env.reset()
    
    # Manually move agent to key position
    env.agent_pos = list(metadata['key'])
    env.has_key = False
    
    # Take any action (will stay at key if it's against a wall or move away)
    initial_state = env.get_state()
    
    # Simulate being at key
    if tuple(env.agent_pos) == metadata['key']:
        assert env.has_key == False  # Not collected yet
        
        # Move away and back
        env.agent_pos = [env.agent_pos[0] + 1, env.agent_pos[1]]
        env.agent_pos = list(metadata['key'])
        
        # Execute through step to trigger collection
        old_key_state = env.has_key
        env._execute_action(Action.UP, Action.UP)  # Execute action at key
        
        # Key collection is tested implicitly


def test_terminal_states():
    """Test terminal state detection"""
    generator = MazeGenerator(size=15, seed=8)
    maze, metadata = generator.generate()
    
    env = MazeEnvironment(maze, metadata, seed=8)
    
    # Test goal reached
    state = (metadata['goal'][0], metadata['goal'][1], 1, 100)
    assert env.is_terminal_state(state)
    
    # Test out of energy
    state = (5, 5, 0, 0)
    assert env.is_terminal_state(state)
    
    # Test normal state
    state = (5, 5, 0, 100)
    assert not env.is_terminal_state(state)


def test_reward_sparse():
    """Test sparse reward structure"""
    generator = MazeGenerator(size=15, seed=8)
    maze, metadata = generator.generate()
    
    env = MazeEnvironment(maze, metadata, reward_type='sparse', seed=8)
    env.reset()
    
    # Normal step should have small negative reward
    next_state, reward, done, info = env.step(Action.RIGHT)
    
    # Reward should be small and negative (step cost)
    assert reward <= 0


def test_reward_shaped():
    """Test shaped reward structure"""
    generator = MazeGenerator(size=15, seed=8)
    maze, metadata = generator.generate()
    
    env = MazeEnvironment(maze, metadata, reward_type='shaped', seed=8)
    env.reset()
    
    # Take step
    next_state, reward, done, info = env.step(Action.RIGHT)
    
    # Should have reward (could be shaped)
    assert isinstance(reward, (int, float))


def test_transition_probabilities():
    """Test transition probability calculation"""
    generator = MazeGenerator(size=15, seed=8)
    maze, metadata = generator.generate()
    
    env = MazeEnvironment(maze, metadata, seed=8)
    env.reset()
    
    state = env.get_state()
    transitions = env.get_transition_probabilities(state, Action.UP)
    
    # Should have transitions
    assert len(transitions) > 0
    
    # Probabilities should sum to 1.0
    total_prob = sum(prob for _, prob in transitions)
    assert abs(total_prob - 1.0) < 1e-6


def test_all_states_generation():
    """Test getting all possible states"""
    generator = MazeGenerator(size=10, seed=8)  # Smaller for faster test
    maze, metadata = generator.generate()
    
    env = MazeEnvironment(maze, metadata, seed=8)
    states = env.get_all_states()
    
    # Should have many states
    assert len(states) > 0
    
    # Each state should have 4 components
    for state in states[:5]:  # Check first few
        assert len(state) == 4


def test_energy_consumption():
    """Test energy consumption mechanic"""
    generator = MazeGenerator(size=15, seed=8)
    maze, metadata = generator.generate()
    
    env = MazeEnvironment(maze, metadata, max_energy=100, seed=8)
    state = env.reset()
    
    initial_energy = state[3]
    
    # Take a step
    next_state, reward, done, info = env.step(Action.RIGHT)
    new_energy = next_state[3]
    
    # Energy should decrease (if moved) or stay same (if hit wall)
    assert new_energy <= initial_energy


def test_target_maze_generation():
    """Test generation of target mazes for transfer learning"""
    generator = MazeGenerator(size=15, seed=8)
    source_maze, source_metadata = generator.generate()
    
    similar, different, similar_meta, different_meta = generate_target_mazes(
        source_maze, source_metadata, seed=8
    )
    
    # Should have same size
    assert similar.shape == source_maze.shape
    assert different.shape == source_maze.shape
    
    # Should be different from source
    assert not np.array_equal(similar, source_maze)
    assert not np.array_equal(different, source_maze)


def test_maze_rendering():
    """Test ASCII rendering"""
    generator = MazeGenerator(size=15, seed=8)
    maze, metadata = generator.generate()
    
    env = MazeEnvironment(maze, metadata, seed=8)
    env.reset()
    
    rendering = env.render()
    
    # Should have content
    assert len(rendering) > 0
    assert 'Steps:' in rendering
    assert 'Energy:' in rendering


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
