"""
Run all experiments for RL Final Project
Student ID: 40415484

This script runs the complete experiment pipeline:
1. Value Iteration with different gamma values
2. Q-Learning with different epsilon decay strategies
3. SARSA(λ) with different lambda values
4. Comparison of all three algorithms
5. Transfer learning experiments
"""
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.generator import MazeGenerator
from environments.maze import MazeEnvironment
from agents.value_iteration import ValueIteration, test_gamma_values
from agents.q_learning import QLearning, compare_epsilon_decay
from agents.sarsa_lambda import SarsaLambda, compare_lambda_values


def ensure_dirs():
    """Create necessary directories."""
    dirs = [
        project_root / "results" / "models",
        project_root / "results" / "raw_data",
        project_root / "results" / "figures",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def load_maze():
    """Load or generate the source maze."""
    maze_path = project_root / "environments" / "maps" / "maze_seed8_size15.npz"
    
    try:
        maze, metadata = MazeGenerator.load_maze(str(maze_path))
        print("✓ Loaded saved maze")
    except Exception:
        print("Generating maze...")
        generator = MazeGenerator(size=15, seed=8)
        maze, metadata = generator.generate()
        generator.save_maze(maze, metadata)
        print("✓ Maze generated and saved")
    
    return maze, metadata


def run_value_iteration_experiments(env):
    """Run Value Iteration experiments with different gamma values."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Value Iteration - Gamma Analysis")
    print("=" * 60)
    
    gamma_values = [0.9, 0.95, 0.99]
    results = test_gamma_values(env, gamma_values, verbose=False)
    
    # Save best model
    best_gamma = max(results, key=lambda g: results[g]['eval']['success_rate'])
    best_agent = results[best_gamma]['agent']
    best_agent.save(str(project_root / "results" / "models" / f"value_iteration_gamma{best_gamma}.pkl"))
    
    print(f"\n✓ Best gamma: {best_gamma}")
    
    return results


def run_qlearning_experiments(env):
    """Run Q-Learning experiments."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Q-Learning - Epsilon Decay Comparison")
    print("=" * 60)
    
    results = compare_epsilon_decay(env, ['exponential', 'linear'], 
                                   episodes=3000, verbose=False)
    
    # Save best model
    best_decay = max(results, key=lambda d: results[d]['eval']['success_rate'])
    best_agent = results[best_decay]['agent']
    best_agent.save(str(project_root / "results" / "models" / "q_learning_demo.pkl"))
    
    # Save training logs
    best_agent.save_training_log(str(project_root / "results" / "raw_data" / "q_learning_training_log.csv"))
    best_agent.save_q_update_log(str(project_root / "results" / "raw_data" / "q_learning_q_update_log.csv"))
    
    print(f"\n✓ Best decay type: {best_decay}")
    
    return results


def run_sarsa_experiments(env):
    """Run SARSA(λ) experiments."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: SARSA(λ) - Lambda Comparison")
    print("=" * 60)
    
    lambda_values = [0.0, 0.3, 0.7, 0.9]
    results = compare_lambda_values(env, lambda_values, episodes=3000, verbose=False)
    
    # Save best model
    best_lambda = max(results, key=lambda l: results[l]['eval']['success_rate'])
    best_agent = results[best_lambda]['agent']
    best_agent.save(str(project_root / "results" / "models" / f"sarsa_lambda_{str(best_lambda).replace('.', '')}.pkl"))
    
    # Save training logs
    best_agent.save_training_log(str(project_root / "results" / "raw_data" / "sarsa_lambda_training_log.csv"))
    best_agent.save_td_trace_log(str(project_root / "results" / "raw_data" / "sarsa_lambda_td_trace_log.csv"))
    
    print(f"\n✓ Best lambda: {best_lambda}")
    
    return results


def run_transfer_learning(maze, metadata):
    """Run transfer learning experiments."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Transfer Learning")
    print("=" * 60)
    
    from transfer.transfer_learning import TransferLearning
    
    transfer = TransferLearning(maze, metadata, seed=42)
    results = transfer.run_full_experiment(source_episodes=3000, target_episodes=2000)
    transfer.save_results()
    
    return results


def main():
    """Run all experiments."""
    print("=" * 60)
    print("RL Final Project - Complete Experiment Suite")
    print("Student ID: 40415484")
    print("=" * 60)
    
    ensure_dirs()
    maze, metadata = load_maze()
    
    # Create environment
    env = MazeEnvironment(maze, metadata, reward_type='sparse', seed=42)
    
    # Run experiments
    vi_results = run_value_iteration_experiments(env)
    ql_results = run_qlearning_experiments(env)
    sarsa_results = run_sarsa_experiments(env)
    
    # Transfer learning
    tl_results = run_transfer_learning(maze, metadata)
    
    print("\n" + "=" * 60)
    print("✅ ALL EXPERIMENTS COMPLETED!")
    print("=" * 60)
    print(f"\nResults saved in: {project_root / 'results'}")
    print(f"Models saved in: {project_root / 'results' / 'models'}")
    print(f"Raw data saved in: {project_root / 'results' / 'raw_data'}")


if __name__ == '__main__':
    main()
