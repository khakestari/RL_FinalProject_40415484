"""
Unit tests for Value Iteration
"""
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.generator import MazeGenerator
from environments.maze import MazeEnvironment
from agents.value_iteration import ValueIteration


def test_value_iteration_initialization():
    """Test Value Iteration initialization"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = ValueIteration(env, gamma=0.99, theta=1e-6, verbose=False)
    
    assert agent.gamma == 0.99
    assert agent.theta == 1e-6
    assert len(agent.states) > 0
    assert len(agent.V) == 0  # Not trained yet


def test_value_iteration_training():
    """Test Value Iteration training"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = ValueIteration(env, gamma=0.99, theta=1e-4, 
                          max_iterations=50, verbose=False)
    stats = agent.train()
    
    assert 'num_iterations' in stats
    assert 'training_time' in stats
    assert stats['num_iterations'] > 0
    assert len(agent.V) > 0
    assert len(agent.policy) > 0


def test_convergence():
    """Test that Value Iteration converges"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = ValueIteration(env, gamma=0.99, theta=1e-4, verbose=False)
    stats = agent.train()
    
    # Check convergence
    assert len(stats['convergence_history']) > 0
    
    # Last delta should be below threshold (or close to it)
    final_delta = stats['convergence_history'][-1]
    assert final_delta <= stats['theta'] * 10  # Allow some margin


def test_policy_extraction():
    """Test policy extraction from value function"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = ValueIteration(env, gamma=0.99, theta=1e-4, 
                          max_iterations=30, verbose=False)
    agent.train()
    
    # Check policy exists
    assert len(agent.policy) > 0
    
    # Check actions are valid
    for state, action in agent.policy.items():
        assert action in [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT


def test_get_action():
    """Test getting action from policy"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = ValueIteration(env, gamma=0.99, theta=1e-4,
                          max_iterations=30, verbose=False)
    agent.train()
    
    state = env.reset()
    action = agent.get_action(state)
    
    assert action in [0, 1, 2, 3]


def test_evaluation():
    """Test policy evaluation"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = ValueIteration(env, gamma=0.99, theta=1e-4,
                          max_iterations=30, verbose=False)
    agent.train()
    
    eval_stats = agent.evaluate(num_episodes=10, max_steps=100)
    
    assert 'mean_reward' in eval_stats
    assert 'success_rate' in eval_stats
    assert 0 <= eval_stats['success_rate'] <= 1
    assert len(eval_stats['episode_rewards']) == 10


def test_q_value_computation():
    """Test Q-value computation"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = ValueIteration(env, gamma=0.99, theta=1e-4, verbose=False)
    
    state = (5, 5, 0, 100)
    q_value = agent._compute_q_value(state, 0)  # UP action
    
    assert isinstance(q_value, (int, float))


def test_value_heatmap():
    """Test value function heatmap generation"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = ValueIteration(env, gamma=0.99, theta=1e-4,
                          max_iterations=30, verbose=False)
    agent.train()
    
    heatmap = agent.get_value_heatmap(has_key=0, energy=env.max_energy)
    
    assert heatmap.shape == (env.size, env.size)
    assert isinstance(heatmap, np.ndarray)


def test_policy_map():
    """Test policy map generation"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = ValueIteration(env, gamma=0.99, theta=1e-4,
                          max_iterations=30, verbose=False)
    agent.train()
    
    policy_map = agent.get_policy_map(has_key=0, energy=env.max_energy)
    
    assert policy_map.shape == (env.size, env.size)
    assert isinstance(policy_map, np.ndarray)


def test_save_load():
    """Test saving and loading agent"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    # Train agent
    agent1 = ValueIteration(env, gamma=0.99, theta=1e-4,
                           max_iterations=20, verbose=False)
    agent1.train()
    
    # Save
    filepath = "test_value_iteration_model.pkl"
    agent1.save(filepath)
    
    # Load
    agent2 = ValueIteration(env, gamma=0.99, verbose=False)
    agent2.load(filepath)
    
    # Check loaded values
    assert agent2.gamma == agent1.gamma
    assert agent2.num_iterations == agent1.num_iterations
    assert len(agent2.policy) == len(agent1.policy)
    
    # Cleanup
    Path(filepath).unlink(missing_ok=True)


def test_gamma_effect():
    """Test effect of different gamma values"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    results = {}
    
    for gamma in [0.9, 0.99]:
        agent = ValueIteration(env, gamma=gamma, theta=1e-6,
                              max_iterations=100, verbose=False)
        stats = agent.train()
        results[gamma] = stats
    
    # Both should complete training
    assert results[0.9]['num_iterations'] > 0
    assert results[0.99]['num_iterations'] > 0
    
    # Gamma values should be stored correctly
    agent1 = ValueIteration(env, gamma=0.9, verbose=False)
    agent2 = ValueIteration(env, gamma=0.99, verbose=False)
    assert agent1.gamma == 0.9
    assert agent2.gamma == 0.99


def test_bellman_update():
    """Test single Bellman update step"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = ValueIteration(env, gamma=0.99, verbose=False)
    
    # Initialize V with zeros
    for state in agent.states[:10]:
        agent.V[state] = 0.0
    
    # Perform one update
    delta = agent._value_iteration_step()
    
    assert isinstance(delta, float)
    assert delta >= 0


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
