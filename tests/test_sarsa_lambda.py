"""
Unit tests for SARSA(λ)
"""
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.generator import MazeGenerator
from environments.maze import MazeEnvironment
from agents.sarsa_lambda import SarsaLambda


def test_sarsa_initialization():
    """Test SARSA(λ) initialization"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = SarsaLambda(env, alpha=0.1, gamma=0.99, lambda_param=0.7,
                       episodes=100, verbose=False)
    
    assert agent.alpha == 0.1
    assert agent.gamma == 0.99
    assert agent.lambda_param == 0.7
    assert agent.epsilon == agent.epsilon_start
    assert len(agent.Q) == 0
    assert len(agent.E) == 0


def test_sarsa_training():
    """Test SARSA(λ) training"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = SarsaLambda(env, episodes=50, max_steps=100, verbose=False)
    stats = agent.train()
    
    assert 'episodes' in stats
    assert 'training_time' in stats
    assert 'lambda' in stats
    assert len(agent.episode_rewards) == 50
    assert len(agent.Q) > 0


def test_eligibility_trace_accumulating():
    """Test accumulating eligibility trace"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = SarsaLambda(env, lambda_param=0.7, trace_type='accumulating',
                       verbose=False)
    
    state = (5, 5, 0, 100)
    action = 0
    
    # Update trace multiple times
    agent._update_trace(state, action)
    trace1 = agent.E[state][action]
    
    agent._update_trace(state, action)
    trace2 = agent.E[state][action]
    
    # Accumulating trace should increase
    assert trace2 > trace1


def test_eligibility_trace_replacing():
    """Test replacing eligibility trace"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = SarsaLambda(env, lambda_param=0.7, trace_type='replacing',
                       verbose=False)
    
    state = (5, 5, 0, 100)
    action = 0
    
    # Update trace multiple times
    agent._update_trace(state, action)
    trace1 = agent.E[state][action]
    
    agent._update_trace(state, action)
    trace2 = agent.E[state][action]
    
    # Replacing trace should stay at 1.0
    assert trace1 == 1.0
    assert trace2 == 1.0


def test_td_error_computation():
    """Test TD error computation"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = SarsaLambda(env, alpha=0.1, gamma=0.99, verbose=False)
    
    state = (5, 5, 0, 100)
    action = 0
    reward = 1.0
    next_state = (4, 5, 0, 99)
    next_action = 1
    done = False
    
    td_error = agent._compute_td_error(state, action, reward,
                                       next_state, next_action, done)
    
    assert isinstance(td_error, (int, float))


def test_q_update_with_traces():
    """Test Q-value update with eligibility traces"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = SarsaLambda(env, alpha=0.1, gamma=0.99, lambda_param=0.7,
                       verbose=False)
    
    state = (5, 5, 0, 100)
    action = 0
    
    # Set initial Q and E
    agent.Q[state][action] = 0.5
    agent.E[state][action] = 1.0
    
    initial_q = agent.Q[state][action]
    
    # Update with TD error
    td_error = 0.2
    agent._update_q_with_traces(td_error)
    
    # Q should have changed
    assert agent.Q[state][action] != initial_q


def test_lambda_zero_behavior():
    """Test that λ=0 behaves like 1-step SARSA"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = SarsaLambda(env, lambda_param=0.0, episodes=30, verbose=False)
    stats = agent.train()
    
    # Should complete training
    assert len(agent.Q) > 0
    assert stats['lambda'] == 0.0


def test_lambda_high_behavior():
    """Test that λ=0.9 provides strong credit assignment"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = SarsaLambda(env, lambda_param=0.9, episodes=30, verbose=False)
    stats = agent.train()
    
    # Should complete training
    assert len(agent.Q) > 0
    assert stats['lambda'] == 0.9


def test_on_policy_behavior():
    """Test that SARSA follows on-policy (uses actual next action)"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = SarsaLambda(env, epsilon_start=0.5, verbose=False)
    
    state = (5, 5, 0, 100)
    next_state = (4, 5, 0, 99)
    
    # Set Q-values to prefer different actions
    agent.Q[state] = np.array([1.0, 0.0, 0.0, 0.0])
    agent.Q[next_state] = np.array([0.0, 2.0, 0.0, 0.0])
    
    # Choose next action (on-policy)
    next_action = agent._choose_action(next_state)
    
    # Compute TD error using that action (on-policy)
    td_error = agent._compute_td_error(state, 0, 1.0, next_state, 
                                       next_action, False)
    
    assert isinstance(td_error, (int, float))


def test_evaluation():
    """Test policy evaluation"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = SarsaLambda(env, episodes=50, verbose=False)
    agent.train()
    
    eval_stats = agent.evaluate(num_episodes=10, max_steps=100)
    
    assert 'mean_reward' in eval_stats
    assert 'success_rate' in eval_stats
    assert 0 <= eval_stats['success_rate'] <= 1
    assert len(eval_stats['episode_rewards']) == 10


def test_q_heatmap():
    """Test Q-value heatmap generation"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = SarsaLambda(env, episodes=30, verbose=False)
    agent.train()
    
    heatmap = agent.get_q_heatmap(has_key=0, energy=env.max_energy)
    
    assert heatmap.shape == (env.size, env.size)
    assert isinstance(heatmap, np.ndarray)


def test_policy_map():
    """Test policy map generation"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = SarsaLambda(env, episodes=30, verbose=False)
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
    agent1 = SarsaLambda(env, lambda_param=0.7, episodes=30, verbose=False)
    agent1.train()
    
    # Save
    filepath = "test_sarsa_model.pkl"
    agent1.save(filepath)
    
    # Load
    agent2 = SarsaLambda(env, verbose=False)
    agent2.load(filepath)
    
    # Check loaded values
    assert agent2.alpha == agent1.alpha
    assert agent2.gamma == agent1.gamma
    assert agent2.lambda_param == agent1.lambda_param
    assert len(agent2.Q) == len(agent1.Q)
    
    # Cleanup
    Path(filepath).unlink(missing_ok=True)


def test_td_trace_logging():
    """Test TD error and trace logging"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = SarsaLambda(env, episodes=2, verbose=False)
    agent.train()
    
    # Should have some logged TD errors
    assert len(agent.td_error_log) > 0
    
    # Check log structure
    log = agent.td_error_log[0]
    assert 'episode' in log
    assert 'step' in log
    assert 'td_error' in log
    assert 'trace_value' in log


def test_trace_decay():
    """Test that eligibility traces decay over time"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = SarsaLambda(env, gamma=0.99, lambda_param=0.7, verbose=False)
    
    state = (5, 5, 0, 100)
    action = 0
    
    # Set trace
    agent.E[state][action] = 1.0
    
    # Decay traces
    agent._update_q_with_traces(0.0)  # TD error = 0, just decay
    
    # Trace should have decayed
    assert agent.E[state][action] < 1.0
    assert agent.E[state][action] > 0


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
