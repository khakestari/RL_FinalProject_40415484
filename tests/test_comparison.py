"""
Unit tests for algorithm comparison
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.generator import MazeGenerator
from experiments.comparison import AlgorithmComparison


def test_comparison_initialization():
    """Test comparison initialization"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    
    comparison = AlgorithmComparison(maze, metadata, seed=42)
    
    assert comparison.env is not None
    assert comparison.maze is not None
    assert len(comparison.results) == 0


def test_run_value_iteration():
    """Test running Value Iteration"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    
    comparison = AlgorithmComparison(maze, metadata, seed=42)
    result = comparison.run_value_iteration(gamma=0.99, theta=1e-4)
    
    assert 'agent' in result
    assert 'train' in result
    assert 'eval' in result
    assert result['algorithm_type'] == 'model-based'


def test_run_q_learning():
    """Test running Q-Learning"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    
    comparison = AlgorithmComparison(maze, metadata, seed=42)
    result = comparison.run_q_learning(episodes=50)
    
    assert 'agent' in result
    assert 'train' in result
    assert 'eval' in result
    assert result['algorithm_type'] == 'model-free'
    assert result['policy_type'] == 'off-policy'


def test_run_sarsa():
    """Test running SARSA"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    
    comparison = AlgorithmComparison(maze, metadata, seed=42)
    result = comparison.run_sarsa_lambda(episodes=50)
    
    assert 'agent' in result
    assert 'train' in result
    assert 'eval' in result
    assert result['algorithm_type'] == 'model-free'
    assert result['policy_type'] == 'on-policy'


def test_compare_performance():
    """Test performance comparison"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    
    comparison = AlgorithmComparison(maze, metadata, seed=42)
    
    # Run algorithms
    comparison.run_value_iteration(theta=1e-4)
    comparison.run_q_learning(episodes=30)
    comparison.run_sarsa_lambda(episodes=30)
    
    # Compare
    performance = comparison.compare_performance()
    
    assert len(performance) == 3
    assert 'value_iteration' in performance
    assert 'q_learning' in performance
    assert 'sarsa_lambda' in performance
    
    # Check metrics exist
    for name, metrics in performance.items():
        assert 'mean_reward' in metrics
        assert 'success_rate' in metrics
        assert 'training_time' in metrics


def test_policy_agreement():
    """Test policy agreement computation"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    
    comparison = AlgorithmComparison(maze, metadata, seed=42)
    
    # Run algorithms
    comparison.run_value_iteration(theta=1e-4)
    comparison.run_q_learning(episodes=30)
    
    # Compute agreement
    agreement = comparison.compute_policy_agreement(reference='value_iteration')
    
    assert 'agreements' in agreement
    assert 'disagreement_details' in agreement
    assert len(agreement['agreements']) > 0


def test_convergence_analysis():
    """Test convergence analysis"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    
    comparison = AlgorithmComparison(maze, metadata, seed=42)
    
    # Run algorithms
    comparison.run_value_iteration(theta=1e-4)
    comparison.run_q_learning(episodes=50)
    
    # Analyze convergence
    convergence = comparison.analyze_convergence()
    
    assert 'value_iteration' in convergence
    assert 'q_learning' in convergence
    
    # Check VI convergence info
    assert 'iterations_to_converge' in convergence['value_iteration']


def test_save_comparison_report():
    """Test saving comparison report"""
    generator = MazeGenerator(size=10, seed=8)
    maze, metadata = generator.generate()
    
    comparison = AlgorithmComparison(maze, metadata, seed=42)
    
    # Run algorithms
    comparison.run_value_iteration(theta=1e-4)
    comparison.run_q_learning(episodes=30)
    
    # Save report
    filepath = "test_comparison_report.json"
    comparison.save_comparison_report(filepath)
    
    # Check file exists
    assert Path(filepath).exists()
    
    # Cleanup
    Path(filepath).unlink(missing_ok=True)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
