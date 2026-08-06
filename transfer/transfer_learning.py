import sys
from pathlib import Path
import numpy as np
import time
from typing import Dict, Tuple
from collections import defaultdict
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.generator import MazeGenerator, generate_target_mazes
from environments.maze import MazeEnvironment
from agents.q_learning import QLearning


class TransferLearning:
    """
    Transfer Learning experiments for Q-Learning.
    
    Train on source environment, then transfer to target environments.
    """
    
    def __init__(self, source_maze, source_metadata, seed: int = 42):
        """
        Initialize transfer learning.
        
        Args:
            source_maze: Source maze array
            source_metadata: Source maze metadata
            seed: Random seed
        """
        self.source_maze = source_maze
        self.source_metadata = source_metadata
        self.seed = seed
        
        # Generate target mazes
        print("Generating target mazes...")
        similar, different, similar_meta, different_meta = generate_target_mazes(
            source_maze, source_metadata, seed=seed
        )
        
        self.target_mazes = {
            'similar': (similar, similar_meta),
            'different': (different, different_meta)
        }
        
        print(f"✓ Target mazes generated")
        
        # Results storage
        self.results = {}
        
    def train_source_agent(self, episodes: int = 3000, 
                          alpha: float = 0.1, gamma: float = 0.99) -> QLearning:
        """
        Train Q-Learning agent on source environment.
        
        Returns:
            Trained agent
        """
        print("\n" + "=" * 60)
        print("Training on SOURCE Environment")
        print("=" * 60)
        
        env = MazeEnvironment(self.source_maze, self.source_metadata, 
                             reward_type='sparse', seed=self.seed)
        
        agent = QLearning(
            env,
            alpha=alpha,
            gamma=gamma,
            episodes=episodes,
            max_steps=500,
            verbose=True,
            log_every=300
        )
        
        train_stats = agent.train()
        eval_stats = agent.evaluate(num_episodes=100, max_steps=500)
        
        self.results['source'] = {
            'agent': agent,
            'train': train_stats,
            'eval': eval_stats
        }
        
        print(f"\n✓ Source training completed")
        print(f"   Success rate: {eval_stats['success_rate']:.1%}")
        print(f"   Mean reward: {eval_stats['mean_reward']:.2f}")
        
        # Save source agent
        agent.save("results/models/q_learning_source.pkl")
        
        return agent
    
    def transfer_experiment(self, source_agent: QLearning, 
                           target_type: str = 'similar',
                           strategy: str = 'full',
                           beta: float = 1.0,
                           episodes: int = 2000) -> Dict:
        """
        Run transfer learning experiment.
        
        Args:
            source_agent: Trained source agent
            target_type: 'similar' or 'different'
            strategy: 'scratch', 'full', 'scaled', or 'selective'
            beta: Scaling factor for 'scaled' strategy
            episodes: Training episodes on target
            
        Returns:
            Dictionary with results
        """
        print(f"\n{'=' * 60}")
        print(f"Transfer to {target_type.upper()} target - Strategy: {strategy}")
        if strategy == 'scaled':
            print(f"Beta: {beta}")
        print("=" * 60)
        
        # Get target maze
        target_maze, target_metadata = self.target_mazes[target_type]
        
        # Create target environment
        target_env = MazeEnvironment(target_maze, target_metadata,
                                    reward_type='sparse', seed=self.seed)
        
        # Create new agent for target
        target_agent = QLearning(
            target_env,
            alpha=source_agent.alpha,
            gamma=source_agent.gamma,
            episodes=episodes,
            max_steps=500,
            verbose=False,
            log_every=200
        )
        
        # Apply transfer strategy
        if strategy == 'scratch':
            # Train from scratch (Q initialized to zero)
            pass
        
        elif strategy == 'full':
            # Full transfer: copy entire Q-table
            target_agent.Q = defaultdict(lambda: np.zeros(4))
            for state, values in source_agent.Q.items():
                target_agent.Q[state] = values.copy()
            print(f"   Transferred {len(source_agent.Q)} states")
        
        elif strategy == 'scaled':
            # Scaled transfer: Q_target = beta * Q_source
            target_agent.Q = defaultdict(lambda: np.zeros(4))
            for state, values in source_agent.Q.items():
                target_agent.Q[state] = beta * values.copy()
            print(f"   Transferred {len(source_agent.Q)} states with β={beta}")
        
        elif strategy == 'selective':
            # Selective transfer: only transfer states with similar neighborhoods
            transferred = self._selective_transfer(
                source_agent, target_agent, 
                self.source_maze, target_maze
            )
            print(f"   Selectively transferred {transferred} states")
        
        # Evaluate initial performance
        initial_eval = target_agent.evaluate(num_episodes=30, max_steps=500)
        
        # Train on target
        print(f"   Training on target...")
        train_stats = target_agent.train()
        
        # Evaluate final performance
        final_eval = target_agent.evaluate(num_episodes=100, max_steps=500)
        
        # Compute learning speed
        learning_speed = self._compute_learning_speed(train_stats['episode_rewards'])
        
        result = {
            'agent': target_agent,
            'initial_eval': initial_eval,
            'train': train_stats,
            'final_eval': final_eval,
            'learning_speed': learning_speed,
            'strategy': strategy,
            'beta': beta if strategy == 'scaled' else None,
        }
        
        print(f"\n   Results:")
        print(f"      Initial success: {initial_eval['success_rate']:.1%}")
        print(f"      Final success: {final_eval['success_rate']:.1%}")
        print(f"      Initial reward: {initial_eval['mean_reward']:.2f}")
        print(f"      Final reward: {final_eval['mean_reward']:.2f}")
        print(f"      Learning speed: {learning_speed:.2f} episodes to threshold")
        
        return result
    
    def _selective_transfer(self, source_agent: QLearning, 
                           target_agent: QLearning,
                           source_maze: np.ndarray, 
                           target_maze: np.ndarray) -> int:
        """
        Selective transfer: only transfer states with unchanged neighborhoods.
        
        Returns:
            Number of states transferred
        """
        from environments.generator import MazeGenerator
        
        transferred = 0
        target_agent.Q = defaultdict(lambda: np.zeros(4))
        
        for state, values in source_agent.Q.items():
            x, y, has_key, energy = state
            
            # Check if position is valid in both mazes
            if x >= min(source_maze.shape[0], target_maze.shape[0]):
                continue
            if y >= min(source_maze.shape[1], target_maze.shape[1]):
                continue
            
            # Check if neighborhood is the same
            if self._is_neighborhood_same(x, y, source_maze, target_maze):
                target_agent.Q[state] = values.copy()
                transferred += 1
        
        return transferred
    
    def _is_neighborhood_same(self, x: int, y: int, 
                             maze1: np.ndarray, maze2: np.ndarray) -> bool:
        """
        Check if 3x3 neighborhood around (x,y) is the same in both mazes.
        """
        from environments.generator import MazeGenerator
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if nx < 0 or nx >= maze1.shape[0] or ny < 0 or ny >= maze1.shape[1]:
                    continue
                if nx < 0 or nx >= maze2.shape[0] or ny < 0 or ny >= maze2.shape[1]:
                    continue
                
                # Compare cell types (wall vs non-wall)
                is_wall1 = (maze1[nx, ny] == MazeGenerator.WALL)
                is_wall2 = (maze2[nx, ny] == MazeGenerator.WALL)
                
                if is_wall1 != is_wall2:
                    return False
        
        return True
    
    def _compute_learning_speed(self, episode_rewards: list, 
                               threshold: float = -50.0) -> float:
        """
        Compute learning speed: episodes needed to reach threshold.
        """
        window = 50
        for i in range(window, len(episode_rewards)):
            avg = np.mean(episode_rewards[i-window:i])
            if avg >= threshold:
                return i
        
        return len(episode_rewards)
    
    def run_full_experiment(self, source_episodes: int = 3000,
                           target_episodes: int = 2000) -> Dict:
        """
        Run complete transfer learning experiment.
        
        Tests all strategies on both target mazes.
        """
        print("\n" + "🔄" * 30)
        print("Transfer Learning - Complete Experiment")
        print("🔄" * 30)
        
        # Train source agent
        source_agent = self.train_source_agent(episodes=source_episodes)
        
        results = {}
        
        # Test on similar target
        print("\n" + "=" * 60)
        print("SIMILAR TARGET MAZE")
        print("=" * 60)
        
        results['similar'] = {}
        
        # Scratch
        results['similar']['scratch'] = self.transfer_experiment(
            source_agent, 'similar', 'scratch', episodes=target_episodes
        )
        
        # Full transfer
        results['similar']['full'] = self.transfer_experiment(
            source_agent, 'similar', 'full', episodes=target_episodes
        )
        
        # Scaled transfer (different betas)
        for beta in [0.25, 0.50, 0.75]:
            key = f'scaled_beta{beta}'
            results['similar'][key] = self.transfer_experiment(
                source_agent, 'similar', 'scaled', beta=beta, episodes=target_episodes
            )
        
        # Selective transfer
        results['similar']['selective'] = self.transfer_experiment(
            source_agent, 'similar', 'selective', episodes=target_episodes
        )
        
        # Test on different target
        print("\n" + "=" * 60)
        print("DIFFERENT TARGET MAZE")
        print("=" * 60)
        
        results['different'] = {}
        
        # Scratch
        results['different']['scratch'] = self.transfer_experiment(
            source_agent, 'different', 'scratch', episodes=target_episodes
        )
        
        # Full transfer
        results['different']['full'] = self.transfer_experiment(
            source_agent, 'different', 'full', episodes=target_episodes
        )
        
        # Scaled transfer (different betas)
        for beta in [0.25, 0.50, 0.75]:
            key = f'scaled_beta{beta}'
            results['different'][key] = self.transfer_experiment(
                source_agent, 'different', 'scaled', beta=beta, episodes=target_episodes
            )
        
        # Selective transfer
        results['different']['selective'] = self.transfer_experiment(
            source_agent, 'different', 'selective', episodes=target_episodes
        )
        
        self.results['transfer'] = results
        
        # Summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print summary of transfer learning results."""
        print("\n" + "=" * 60)
        print("TRANSFER LEARNING SUMMARY")
        print("=" * 60)
        
        for target_type in ['similar', 'different']:
            print(f"\n{target_type.upper()} Target:")
            print(f"{'Strategy':<25} {'Initial':<12} {'Final':<12} {'Speed':<12}")
            print("-" * 60)
            
            for strategy, result in results[target_type].items():
                initial_sr = result['initial_eval']['success_rate']
                final_sr = result['final_eval']['success_rate']
                speed = result['learning_speed']
                
                print(f"{strategy:<25} {initial_sr:<12.1%} {final_sr:<12.1%} {speed:<12.0f}")
    
    def save_results(self, filepath: str = "results/raw_data/transfer_learning.json"):
        """Save transfer learning results."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for JSON
        save_data = {
            'source': {
                'train_episodes': self.results['source']['train']['episodes'],
                'success_rate': self.results['source']['eval']['success_rate'],
                'mean_reward': self.results['source']['eval']['mean_reward'],
            },
            'transfer': {}
        }
        
        for target_type, strategies in self.results['transfer'].items():
            save_data['transfer'][target_type] = {}
            
            for strategy, result in strategies.items():
                save_data['transfer'][target_type][strategy] = {
                    'initial_success_rate': result['initial_eval']['success_rate'],
                    'initial_mean_reward': result['initial_eval']['mean_reward'],
                    'final_success_rate': result['final_eval']['success_rate'],
                    'final_mean_reward': result['final_eval']['mean_reward'],
                    'learning_speed': result['learning_speed'],
                    'training_time': result['train']['training_time'],
                }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n✓ Results saved to {filepath}")


def run_transfer_learning():
    """Run transfer learning experiment."""
    print("\n" + "=" * 60)
    print("Transfer Learning Experiment")
    print("Student ID: 40415484")
    print("=" * 60)
    
    # Load source maze
    try:
        maze, metadata = MazeGenerator.load_maze("environments/maps/maze_seed8_size15.npz")
        print("✓ Loaded source maze")
    except:
        print("Generating source maze...")
        generator = MazeGenerator(size=15, seed=8)
        maze, metadata = generator.generate()
        generator.save_maze(maze, metadata)
        print("✓ Source maze generated")
    
    # Initialize transfer learning
    transfer = TransferLearning(maze, metadata, seed=42)
    
    # Run experiment
    results = transfer.run_full_experiment(source_episodes=2000, target_episodes=1500)
    
    # Save results
    transfer.save_results()
    
    print("\n" + "=" * 60)
    print("✅ Transfer Learning Experiment Completed!")
    print("=" * 60)
    
    return transfer


if __name__ == '__main__':
    transfer = run_transfer_learning()
