"""
Comprehensive comparison of three RL algorithms
Value Iteration vs Q-Learning vs SARSA(λ)
"""

import sys
from pathlib import Path
import numpy as np
import time
import json
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.generator import MazeGenerator
from environments.maze import MazeEnvironment
from agents.value_iteration import ValueIteration
from agents.q_learning import QLearning
from agents.sarsa_lambda import SarsaLambda


class AlgorithmComparison:
    """
    Compare Value Iteration, Q-Learning, and SARSA(λ) on same environment.
    """
    
    def __init__(self, maze, metadata, seed: int = 42):
        """
        Initialize comparison.
        
        Args:
            maze: Maze array
            metadata: Maze metadata
            seed: Random seed
        """
        self.maze = maze
        self.metadata = metadata
        self.seed = seed
        self.env = MazeEnvironment(maze, metadata, reward_type='sparse', seed=seed)
        
        self.results = {}
        
    def run_value_iteration(self, gamma: float = 0.99, 
                           theta: float = 1e-6) -> Dict:
        """Run and evaluate Value Iteration."""
        print("\n" + "=" * 60)
        print("Training Value Iteration")
        print("=" * 60)
        
        agent = ValueIteration(self.env, gamma=gamma, theta=theta, verbose=True)
        
        start_time = time.time()
        train_stats = agent.train()
        training_time = time.time() - start_time
        
        # Evaluate
        eval_stats = agent.evaluate(num_episodes=100, max_steps=500)
        
        self.results['value_iteration'] = {
            'agent': agent,
            'train': train_stats,
            'eval': eval_stats,
            'algorithm_type': 'model-based',
            'policy_type': 'deterministic',
        }
        
        print(f"\n✓ Value Iteration completed")
        print(f"   Training time: {training_time:.2f}s")
        print(f"   Iterations: {train_stats['num_iterations']}")
        print(f"   Success rate: {eval_stats['success_rate']:.1%}")
        print(f"   Mean reward: {eval_stats['mean_reward']:.2f}")
        
        return self.results['value_iteration']
    
    def run_q_learning(self, alpha: float = 0.1, gamma: float = 0.99,
                      episodes: int = 3000) -> Dict:
        """Run and evaluate Q-Learning."""
        print("\n" + "=" * 60)
        print("Training Q-Learning")
        print("=" * 60)
        
        agent = QLearning(
            self.env,
            alpha=alpha,
            gamma=gamma,
            episodes=episodes,
            max_steps=500,
            verbose=True,
            log_every=300
        )
        
        train_stats = agent.train()
        
        # Evaluate
        eval_stats = agent.evaluate(num_episodes=100, max_steps=500)
        
        self.results['q_learning'] = {
            'agent': agent,
            'train': train_stats,
            'eval': eval_stats,
            'algorithm_type': 'model-free',
            'policy_type': 'off-policy',
        }
        
        print(f"\n✓ Q-Learning completed")
        print(f"   Training time: {train_stats['training_time']:.2f}s")
        print(f"   Episodes: {episodes}")
        print(f"   Q-table size: {train_stats['q_table_size']}")
        print(f"   Success rate: {eval_stats['success_rate']:.1%}")
        print(f"   Mean reward: {eval_stats['mean_reward']:.2f}")
        
        return self.results['q_learning']
    
    def run_sarsa_lambda(self, alpha: float = 0.1, gamma: float = 0.99,
                        lambda_param: float = 0.7, episodes: int = 3000) -> Dict:
        """Run and evaluate SARSA(λ)."""
        print("\n" + "=" * 60)
        print(f"Training SARSA(λ) with λ={lambda_param}")
        print("=" * 60)
        
        agent = SarsaLambda(
            self.env,
            alpha=alpha,
            gamma=gamma,
            lambda_param=lambda_param,
            episodes=episodes,
            max_steps=500,
            verbose=True,
            log_every=300
        )
        
        train_stats = agent.train()
        
        # Evaluate
        eval_stats = agent.evaluate(num_episodes=100, max_steps=500)
        
        self.results['sarsa_lambda'] = {
            'agent': agent,
            'train': train_stats,
            'eval': eval_stats,
            'algorithm_type': 'model-free',
            'policy_type': 'on-policy',
        }
        
        print(f"\n✓ SARSA(λ) completed")
        print(f"   Training time: {train_stats['training_time']:.2f}s")
        print(f"   Episodes: {episodes}")
        print(f"   Q-table size: {train_stats['q_table_size']}")
        print(f"   Success rate: {eval_stats['success_rate']:.1%}")
        print(f"   Mean reward: {eval_stats['mean_reward']:.2f}")
        
        return self.results['sarsa_lambda']
    
    def compare_performance(self) -> Dict:
        """Compare performance metrics across algorithms."""
        print("\n" + "=" * 60)
        print("Performance Comparison")
        print("=" * 60)
        
        comparison = {}
        
        for name, result in self.results.items():
            comparison[name] = {
                'mean_reward': result['eval']['mean_reward'],
                'std_reward': result['eval']['std_reward'],
                'mean_length': result['eval']['mean_length'],
                'std_length': result['eval']['std_length'],
                'success_rate': result['eval']['success_rate'],
                'training_time': result['train'].get('training_time', 0),
                'algorithm_type': result['algorithm_type'],
            }
        
        # Print table
        print(f"\n{'Algorithm':<20} {'Success Rate':<15} {'Mean Reward':<15} {'Mean Steps':<15} {'Time (s)':<15}")
        print("-" * 80)
        
        for name, metrics in comparison.items():
            display_name = name.replace('_', ' ').title()
            print(f"{display_name:<20} "
                  f"{metrics['success_rate']:<15.1%} "
                  f"{metrics['mean_reward']:<15.2f} "
                  f"{metrics['mean_length']:<15.1f} "
                  f"{metrics['training_time']:<15.2f}")
        
        return comparison
    
    def compute_policy_agreement(self, reference: str = 'value_iteration') -> Dict:
        """
        Compute policy agreement between algorithms.
        
        Args:
            reference: Reference algorithm (usually 'value_iteration')
            
        Returns:
            Dictionary with agreement percentages
        """
        print("\n" + "=" * 60)
        print(f"Policy Agreement Analysis (Reference: {reference})")
        print("=" * 60)
        
        if reference not in self.results:
            print(f"⚠️  Reference algorithm '{reference}' not found")
            return {}
        
        ref_agent = self.results[reference]['agent']
        energy = self.env.max_energy
        
        agreements = {}
        disagreement_details = {}
        
        for name, result in self.results.items():
            if name == reference:
                continue
            
            agent = result['agent']
            
            # Compare policies for both key states
            for has_key in [0, 1]:
                key_label = "with_key" if has_key else "no_key"
                
                total_cells = 0
                agreement_count = 0
                disagreements = []
                
                for x in range(self.env.size):
                    for y in range(self.env.size):
                        if not self.env._is_valid_position((x, y)):
                            continue
                        
                        state = (x, y, has_key, energy)
                        
                        # Get actions from both agents
                        ref_action = ref_agent.get_action(state)
                        agent_action = agent.get_action(state, greedy=True)
                        
                        total_cells += 1
                        
                        if ref_action == agent_action:
                            agreement_count += 1
                        else:
                            disagreements.append({
                                'position': (x, y),
                                'ref_action': ref_action,
                                'agent_action': agent_action,
                            })
                
                agreement_pct = (agreement_count / total_cells * 100) if total_cells > 0 else 0
                
                key_name = f"{name}_{key_label}"
                agreements[key_name] = {
                    'agreement_percentage': agreement_pct,
                    'agreement_count': agreement_count,
                    'total_cells': total_cells,
                    'disagreement_count': len(disagreements),
                }
                
                disagreement_details[key_name] = disagreements[:10]  # Store first 10
                
                print(f"\n{name.replace('_', ' ').title()} ({key_label}):")
                print(f"   Agreement: {agreement_pct:.1f}% ({agreement_count}/{total_cells})")
                print(f"   Disagreements: {len(disagreements)}")
        
        return {
            'agreements': agreements,
            'disagreement_details': disagreement_details
        }
    
    def analyze_convergence(self) -> Dict:
        """Analyze convergence speed and stability."""
        print("\n" + "=" * 60)
        print("Convergence Analysis")
        print("=" * 60)
        
        analysis = {}
        
        # Value Iteration convergence
        if 'value_iteration' in self.results:
            vi_result = self.results['value_iteration']
            analysis['value_iteration'] = {
                'iterations_to_converge': vi_result['train']['num_iterations'],
                'convergence_threshold': vi_result['train']['theta'],
                'final_delta': vi_result['train']['final_delta'],
            }
            print(f"\nValue Iteration:")
            print(f"   Iterations: {analysis['value_iteration']['iterations_to_converge']}")
            print(f"   Final delta: {analysis['value_iteration']['final_delta']:.2e}")
        
        # Q-Learning learning curve
        if 'q_learning' in self.results:
            ql_result = self.results['q_learning']
            rewards = ql_result['train']['episode_rewards']
            
            # Compute moving average
            window = min(100, len(rewards))
            if len(rewards) >= window:
                analysis['q_learning'] = {
                    'initial_avg_reward': np.mean(rewards[:window]),
                    'final_avg_reward': np.mean(rewards[-window:]),
                    'improvement': np.mean(rewards[-window:]) - np.mean(rewards[:window]),
                    'std_final': np.std(rewards[-window:]),
                }
                print(f"\nQ-Learning:")
                print(f"   Initial reward: {analysis['q_learning']['initial_avg_reward']:.2f}")
                print(f"   Final reward: {analysis['q_learning']['final_avg_reward']:.2f}")
                print(f"   Improvement: {analysis['q_learning']['improvement']:.2f}")
        
        # SARSA learning curve
        if 'sarsa_lambda' in self.results:
            sarsa_result = self.results['sarsa_lambda']
            rewards = sarsa_result['train']['episode_rewards']
            
            window = min(100, len(rewards))
            if len(rewards) >= window:
                analysis['sarsa_lambda'] = {
                    'initial_avg_reward': np.mean(rewards[:window]),
                    'final_avg_reward': np.mean(rewards[-window:]),
                    'improvement': np.mean(rewards[-window:]) - np.mean(rewards[:window]),
                    'std_final': np.std(rewards[-window:]),
                }
                print(f"\nSARSA(λ):")
                print(f"   Initial reward: {analysis['sarsa_lambda']['initial_avg_reward']:.2f}")
                print(f"   Final reward: {analysis['sarsa_lambda']['final_avg_reward']:.2f}")
                print(f"   Improvement: {analysis['sarsa_lambda']['improvement']:.2f}")
        
        return analysis
    
    def save_comparison_report(self, filepath: str = "results/raw_data/algorithm_comparison.json"):
        """Save comprehensive comparison report."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'maze_info': {
                'size': self.env.size,
                'seed': self.seed,
                'start': self.metadata['start'],
                'key': self.metadata['key'],
                'door': self.metadata['door'],
                'goal': self.metadata['goal'],
            },
            'algorithms': {}
        }
        
        for name, result in self.results.items():
            report['algorithms'][name] = {
                'algorithm_type': result['algorithm_type'],
                'policy_type': result['policy_type'],
                'training': {
                    'time': result['train'].get('training_time', 0),
                    'episodes': result['train'].get('episodes', result['train'].get('num_iterations', 0)),
                },
                'evaluation': {
                    'mean_reward': result['eval']['mean_reward'],
                    'std_reward': result['eval']['std_reward'],
                    'success_rate': result['eval']['success_rate'],
                    'mean_length': result['eval']['mean_length'],
                }
            }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Comparison report saved to {filepath}")


def run_full_comparison():
    """Run complete comparison experiment."""
    print("\n" + "🎯" * 30)
    print("Complete Algorithm Comparison")
    print("Student ID: 40415484")
    print("🎯" * 30)
    
    # Load maze
    try:
        maze, metadata = MazeGenerator.load_maze("environments/maps/maze_seed8_size15.npz")
        print("\n✓ Loaded maze from file")
    except:
        print("\n⚠️  Generating new maze...")
        generator = MazeGenerator(size=15, seed=8)
        maze, metadata = generator.generate()
        generator.save_maze(maze, metadata)
        print("✓ Maze generated and saved")
    
    # Initialize comparison
    comparison = AlgorithmComparison(maze, metadata, seed=42)
    
    # Run all algorithms
    comparison.run_value_iteration(gamma=0.99, theta=1e-6)
    comparison.run_q_learning(alpha=0.1, gamma=0.99, episodes=2000)
    comparison.run_sarsa_lambda(alpha=0.1, gamma=0.99, lambda_param=0.7, episodes=2000)
    
    # Comparative analysis
    performance = comparison.compare_performance()
    policy_agreement = comparison.compute_policy_agreement(reference='value_iteration')
    convergence = comparison.analyze_convergence()
    
    # Save report
    comparison.save_comparison_report()
    
    print("\n" + "=" * 60)
    print("Key Findings")
    print("=" * 60)
    
    # Find best performer
    best_success = max(comparison.results.items(), 
                      key=lambda x: x[1]['eval']['success_rate'])
    best_reward = max(comparison.results.items(),
                     key=lambda x: x[1]['eval']['mean_reward'])
    fastest = min(comparison.results.items(),
                 key=lambda x: x[1]['train'].get('training_time', float('inf')))
    
    print(f"\n🏆 Best Success Rate: {best_success[0]} ({best_success[1]['eval']['success_rate']:.1%})")
    print(f"🏆 Best Mean Reward: {best_reward[0]} ({best_reward[1]['eval']['mean_reward']:.2f})")
    print(f"⚡ Fastest Training: {fastest[0]} ({fastest[1]['train'].get('training_time', 0):.2f}s)")
    
    print("\n" + "=" * 60)
    print("✅ Complete comparison finished!")
    print("=" * 60)
    
    return comparison


if __name__ == '__main__':
    comparison = run_full_comparison()
