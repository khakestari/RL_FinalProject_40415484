"""
RL Final Project - Main Entry Point
Student ID: 40415484

Usage:
    python main.py                              # Launch GUI
    python main.py --mode train --algorithm q_learning
    python main.py --mode evaluate --algorithm value_iteration
    python main.py --mode experiment
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def load_or_generate_maze():
    """Load saved maze or generate a new one."""
    from environments.generator import MazeGenerator
    
    maze_path = project_root / "environments" / "maps" / "maze_seed8_size15.npz"
    
    try:
        maze, metadata = MazeGenerator.load_maze(str(maze_path))
        print("✓ Loaded saved maze")
    except Exception:
        print("Generating new maze...")
        generator = MazeGenerator(size=15, seed=8)
        maze, metadata = generator.generate()
        generator.save_maze(maze, metadata)
        print("✓ Maze generated and saved")
    
    return maze, metadata


def train_agent(args):
    """Train the specified algorithm."""
    from environments.maze import MazeEnvironment
    from agents.value_iteration import ValueIteration
    from agents.q_learning import QLearning
    from agents.sarsa_lambda import SarsaLambda
    
    maze, metadata = load_or_generate_maze()
    
    env = MazeEnvironment(maze, metadata, reward_type='sparse', seed=42)
    
    models_dir = project_root / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    raw_data_dir = project_root / "results" / "raw_data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    if args.algorithm == 'value_iteration':
        print(f"🏋️ Training Value Iteration (γ={0.99})...")
        agent = ValueIteration(env, gamma=0.99, theta=1e-6, verbose=True)
        stats = agent.train()
        eval_stats = agent.evaluate(num_episodes=100)
        
        save_path = str(models_dir / "value_iteration_gamma099.pkl")
        agent.save(save_path)
        
    elif args.algorithm == 'q_learning':
        print(f"🏋️ Training Q-Learning ({args.episodes} episodes)...")
        agent = QLearning(
            env, alpha=0.1, gamma=0.99,
            epsilon_decay_type='exponential',
            episodes=args.episodes, max_steps=500,
            verbose=True, log_every=500
        )
        stats = agent.train()
        eval_stats = agent.evaluate(num_episodes=100)
        
        save_path = str(models_dir / "q_learning_demo.pkl")
        agent.save(save_path)
        agent.save_training_log(str(raw_data_dir / "q_learning_training_log.csv"))
        agent.save_q_update_log(str(raw_data_dir / "q_learning_q_update_log.csv"))
        
    elif args.algorithm == 'sarsa':
        print(f"🏋️ Training SARSA(λ=0.7) ({args.episodes} episodes)...")
        agent = SarsaLambda(
            env, alpha=0.1, gamma=0.99,
            lambda_param=0.7,
            trace_type='accumulating',
            episodes=args.episodes, max_steps=500,
            verbose=True, log_every=500
        )
        stats = agent.train()
        eval_stats = agent.evaluate(num_episodes=100)
        
        save_path = str(models_dir / "sarsa_lambda_07.pkl")
        agent.save(save_path)
        agent.save_training_log(str(raw_data_dir / "sarsa_lambda_training_log.csv"))
    
    else:
        print(f"Unknown algorithm: {args.algorithm}")
        return
    
    # Print evaluation results
    print(f"\n📊 Evaluation Results:")
    print(f"   Success Rate: {eval_stats['success_rate']:.1%}")
    print(f"   Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
    print(f"   Mean Length: {eval_stats['mean_length']:.1f} ± {eval_stats['std_length']:.1f}")
    print(f"\n✅ Model saved to {save_path}")


def evaluate_agent(args):
    """Evaluate a trained agent."""
    from environments.maze import MazeEnvironment
    from agents.value_iteration import ValueIteration
    from agents.q_learning import QLearning
    from agents.sarsa_lambda import SarsaLambda
    
    maze, metadata = load_or_generate_maze()
    env = MazeEnvironment(maze, metadata, reward_type='sparse', seed=42)
    
    models_dir = project_root / "results" / "models"
    
    # Find model file
    model_files = {
        'value_iteration': list(models_dir.glob("value_iteration*.pkl")),
        'q_learning': list(models_dir.glob("q_learning*.pkl")),
        'sarsa': list(models_dir.glob("sarsa_lambda*.pkl")),
    }
    
    available = model_files.get(args.algorithm, [])
    if not available:
        print(f"❌ No trained model found for {args.algorithm}")
        print(f"   Run: python main.py --mode train --algorithm {args.algorithm}")
        return
    
    filepath = str(available[0])
    print(f"📂 Loading model from {filepath}")
    
    # Create agent and load
    if args.algorithm == 'value_iteration':
        agent = ValueIteration(env, verbose=False)
    elif args.algorithm == 'q_learning':
        agent = QLearning(env, verbose=False)
    elif args.algorithm == 'sarsa':
        agent = SarsaLambda(env, verbose=False)
    else:
        print(f"Unknown algorithm: {args.algorithm}")
        return
    
    agent.load(filepath)
    
    # Evaluate
    print(f"📊 Evaluating {args.algorithm} on {args.map} map...")
    eval_stats = agent.evaluate(num_episodes=100, max_steps=500)
    
    print(f"\n📊 Evaluation Results:")
    print(f"   Success Rate: {eval_stats['success_rate']:.1%}")
    print(f"   Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
    print(f"   Mean Length: {eval_stats['mean_length']:.1f} ± {eval_stats['std_length']:.1f}")


def main():
    """Main entry point for the RL maze project."""
    
    parser = argparse.ArgumentParser(
        description='RL Final Project: Intelligent Agent in Dynamic Maze'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='gui',
        choices=['gui', 'train', 'evaluate', 'experiment'],
        help='Run mode: gui (launch GUI), train (train agent), evaluate (evaluate agent), experiment (run experiments)'
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        default='q_learning',
        choices=['value_iteration', 'q_learning', 'sarsa'],
        help='Algorithm to use'
    )
    
    parser.add_argument(
        '--map',
        type=str,
        default='source',
        choices=['source', 'target_similar', 'target_different'],
        help='Which map to use'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=3000,
        help='Number of training episodes'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=8,
        help='Random seed (default: 8 based on student ID)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'gui':
        print("🎮 Launching GUI...")
        from gui.app import main as gui_main
        gui_main()
    
    elif args.mode == 'train':
        train_agent(args)
    
    elif args.mode == 'evaluate':
        evaluate_agent(args)
    
    elif args.mode == 'experiment':
        print("🔬 Running experiments...")
        from experiments.run_experiments import main as experiment_main
        experiment_main()
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
