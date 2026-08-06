"""
Unit tests for Q-Learning
"""
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.generator import MazeGenerator
from environments.maze import MazeEnvironment
from agents.q_learning import QLearning


def test_q_learning_initialization():
    """Test Q-Learning initialization"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = QLearning(env, alpha=0.1, gamma=0.99, episodes=100, verbose=False)
    
    assert agent.alpha == 0.1
    assert agent.gamma == 0.99
    assert agent.epsilon == agent.epsilon_start
    assert len(agent.Q) == 0  # Not trained yet


def test_q_learning_training():
    """Test Q-Learning training"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = QLearning(env, episodes=50, max_steps=100, verbose=False)
    stats = agent.train()
    
    assert 'episodes' in stats
    assert 'training_time' in stats
    assert len(agent.episode_rewards) == 50
    assert len(agent.Q) > 0


def test_epsilon_greedy():
    """Test epsilon-greedy action selection"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = QLearning(env, epsilon_start=1.0, verbose=False)
    state = (5, 5, 0, 100)
    
    # With epsilon=1.0, should always explore (random actions)
    actions = [agent._choose_action(state) for _ in range(20)]
    assert len(set(actions)) > 1  # Should have variety
    
    # With epsilon=0.0, should always exploit
    agent.epsilon = 0.0
    agent.Q[state] = np.array([1.0, 0.0, 0.0, 0.0])  # Best action is 0
    actions = [agent._choose_action(state) for _ in range(10)]
    assert all(a == 0 for a in actions)


def test_q_update():
    """Test Q-value update"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = QLearning(env, alpha=0.1, gamma=0.99, verbose=False)
    
    state = (5, 5, 0, 100)
    action = 0
    reward = 1.0
    next_state = (4, 5, 0, 99)
    done = False
    
    # Initial Q-value should be 0
    initial_q = agent.Q[state][action]
    assert initial_q == 0.0
    
    # Update Q-value
    agent._update_q(state, action, reward, next_state, done)
    
    # Q-value should have changed
    updated_q = agent.Q[state][action]
    assert updated_q != initial_q


def test_epsilon_decay_exponential():
    """Test exponential epsilon decay"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = QLearning(env, epsilon_start=1.0, epsilon_end=0.01,
                     epsilon_decay=0.99, epsilon_decay_type='exponential',
                     episodes=100, verbose=False)
    
    initial_epsilon = agent.epsilon
    
    # Simulate some episodes
    for episode in range(10):
        agent._update_epsilon(episode)
    
    # Epsilon should have decreased
    assert agent.epsilon < initial_epsilon
    assert agent.epsilon >= agent.epsilon_end


def test_epsilon_decay_linear():
    """Test linear epsilon decay"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = QLearning(env, epsilon_start=1.0, epsilon_end=0.0,
                     epsilon_decay_type='linear',
                     episodes=100, verbose=False)
    
    # After 50 episodes, epsilon should be around 0.5
    for episode in range(50):
        agent._update_epsilon(episode)
    
    assert 0.4 < agent.epsilon < 0.6


def test_get_action():
    """Test getting action"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = QLearning(env, episodes=20, verbose=False)
    agent.train()
    
    state = env.reset()
    
    # Greedy action
    action_greedy = agent.get_action(state, greedy=True)
    assert action_greedy in [0, 1, 2, 3]
    
    # Non-greedy (epsilon-greedy)
    action_explore = agent.get_action(state, greedy=False)
    assert action_explore in [0, 1, 2, 3]


def test_evaluation():
    """Test policy evaluation"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = QLearning(env, episodes=50, verbose=False)
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
    
    agent = QLearning(env, episodes=30, verbose=False)
    agent.train()
    
    heatmap = agent.get_q_heatmap(has_key=0, energy=env.max_energy)
    
    assert heatmap.shape == (env.size, env.size)
    assert isinstance(heatmap, np.ndarray)


def test_policy_map():
    """Test policy map generation"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = QLearning(env, episodes=30, verbose=False)
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
    agent1 = QLearning(env, episodes=30, verbose=False)
    agent1.train()
    
    # Save
    filepath = "test_q_learning_model.pkl"
    agent1.save(filepath)
    
    # Load
    agent2 = QLearning(env, verbose=False)
    agent2.load(filepath)
    
    # Check loaded values
    assert agent2.alpha == agent1.alpha
    assert agent2.gamma == agent1.gamma
    assert len(agent2.Q) == len(agent1.Q)
    
    # Check Q-values match
    test_state = list(agent1.Q.keys())[0] if len(agent1.Q) > 0 else (5, 5, 0, 100)
    if test_state in agent1.Q:
        assert np.allclose(agent2.Q[test_state], agent1.Q[test_state])
    
    # Cleanup
    Path(filepath).unlink(missing_ok=True)


def test_training_statistics():
    """Test that training statistics are properly collected"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = QLearning(env, episodes=50, verbose=False)
    stats = agent.train()
    
    # Check all statistics are present
    assert len(agent.episode_rewards) == 50
    assert len(agent.episode_lengths) == 50
    assert len(agent.episode_success) == 50
    assert len(agent.epsilon_history) == 50
    assert len(agent.wall_collisions_per_episode) == 50
    assert len(agent.penalty_visits_per_episode) == 50


def test_q_update_logging():
    """Test Q-update logging for manual inspection"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    env = MazeEnvironment(maze, metadata, seed=8)
    
    agent = QLearning(env, episodes=2, verbose=False)
    agent.train()
    
    # Should have some logged updates
    assert len(agent.q_update_log) > 0
    
    # Check log structure
    log = agent.q_update_log[0]
    assert 'episode' in log
    assert 'step' in log
    assert 'state' in log
    assert 'action' in log
    assert 'reward' in log
    assert 'current_q' in log
    assert 'td_error' in log
    assert 'new_q' in log


def test_different_reward_types():
    """Test Q-Learning with different reward types"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    
    # Sparse rewards
    env_sparse = MazeEnvironment(maze, metadata, reward_type='sparse', seed=8)
    agent_sparse = QLearning(env_sparse, episodes=30, verbose=False)
    stats_sparse = agent_sparse.train()
    
    # Shaped rewards
    env_shaped = MazeEnvironment(maze, metadata, reward_type='shaped', seed=8)
    agent_shaped = QLearning(env_shaped, episodes=30, verbose=False)
    stats_shaped = agent_shaped.train()
    
    # Both should complete training
    assert len(agent_sparse.Q) > 0
    assert len(agent_shaped.Q) > 0


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
