"""
Visualization Module
Generate plots and charts for RL agents analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.generator import MazeGenerator

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """
    Comprehensive visualization toolkit for RL experiments.
    """
    
    def __init__(self, save_dir: str = "results/figures"):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save figures
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Figure settings
        self.fig_size = (10, 6)
        self.dpi = 100
    
    def plot_learning_curves(self, agents_data: Dict, 
                            metric: str = 'reward',
                            save_name: str = 'learning_curves.png'):
        """
        Plot learning curves for multiple agents.
        
        Args:
            agents_data: {agent_name: {'episodes': [...], metric: [...]}}
            metric: 'reward', 'length', or 'success_rate'
            save_name: Output filename
        """
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        for agent_name, data in agents_data.items():
            episodes = data.get('episodes', range(len(data[metric])))
            values = data[metric]
            
            # Smooth curve
            smoothed = self._smooth_curve(values, window=50)
            
            ax.plot(episodes, smoothed, label=agent_name, linewidth=2, alpha=0.8)
            
            # Optional: show raw data with transparency
            if len(values) < 500:
                ax.plot(episodes, values, alpha=0.2, linewidth=0.5)
        
        ax.set_xlabel('Episodes', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Learning Curves: {metric.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved: {save_path}")
    
    def plot_value_heatmap(self, value_map: Dict[Tuple[int, int], float],
                          maze: np.ndarray,
                          title: str = 'Value Function Heatmap',
                          save_name: str = 'value_heatmap.png'):
        """
        Plot value function as heatmap.
        
        Args:
            value_map: {(x, y): value}
            maze: Maze array
            title: Plot title
            save_name: Output filename
        """
        height, width = maze.shape
        value_grid = np.full((height, width), np.nan)
        
        # Fill grid
        for (x, y), value in value_map.items():
            if 0 <= x < height and 0 <= y < width:
                if maze[x, y] != MazeGenerator.WALL:
                    value_grid[x, y] = value
        
        fig, ax = plt.subplots(figsize=(8, 8), dpi=self.dpi)
        
        # Create heatmap
        im = ax.imshow(value_grid, cmap='RdYlGn', interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Value', fontsize=12)
        
        # Draw maze walls
        for i in range(height):
            for j in range(width):
                if maze[i, j] == MazeGenerator.WALL:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                              fill=True, color='black', alpha=0.8))
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.grid(False)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved: {save_path}")
    
    def plot_policy_map(self, policy_map: Dict[Tuple[int, int], int],
                       maze: np.ndarray,
                       metadata: Dict,
                       title: str = 'Policy Map',
                       save_name: str = 'policy_map.png'):
        """
        Plot policy as arrow map.
        
        Args:
            policy_map: {(x, y): action}
            maze: Maze array
            metadata: Maze metadata
            title: Plot title
            save_name: Output filename
        """
        height, width = maze.shape
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.dpi)
        
        # Draw maze
        ax.imshow(maze == MazeGenerator.WALL, cmap='gray', alpha=0.3)
        
        # Arrow directions
        arrow_map = {
            0: (0, -1),   # Up
            1: (0, 1),    # Down
            2: (-1, 0),   # Left
            3: (1, 0),    # Right
        }
        
        # Draw arrows
        for (x, y), action in policy_map.items():
            if 0 <= x < height and 0 <= y < width:
                if maze[x, y] != MazeGenerator.WALL:
                    dx, dy = arrow_map.get(action, (0, 0))
                    ax.arrow(y, x, dx*0.3, dy*0.3,
                           head_width=0.2, head_length=0.15,
                           fc='blue', ec='blue', alpha=0.7)
        
        # Mark special positions
        if 'start' in metadata:
            x, y = metadata['start']
            ax.plot(y, x, 'go', markersize=15, label='Start')
        
        if 'key' in metadata:
            x, y = metadata['key']
            ax.plot(y, x, 'yo', markersize=15, label='Key')
        
        if 'door' in metadata:
            x, y = metadata['door']
            ax.plot(y, x, 'mo', markersize=15, label='Door')
        
        if 'goal' in metadata:
            x, y = metadata['goal']
            ax.plot(y, x, 'r*', markersize=20, label='Goal')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved: {save_path}")
    
    def plot_comparison_bar(self, comparison_data: Dict,
                           metrics: List[str] = None,
                           save_name: str = 'comparison.png'):
        """
        Plot comparison bar chart.
        
        Args:
            comparison_data: {agent_name: {metric: value}}
            metrics: List of metrics to plot
            save_name: Output filename
        """
        if metrics is None:
            metrics = ['success_rate', 'mean_reward', 'mean_length']
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5), dpi=self.dpi)
        
        if n_metrics == 1:
            axes = [axes]
        
        agent_names = list(comparison_data.keys())
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            values = [comparison_data[agent].get(metric, 0) for agent in agent_names]
            
            bars = ax.bar(agent_names, values, alpha=0.7, edgecolor='black')
            
            # Color bars
            colors = plt.cm.Set3(np.linspace(0, 1, len(agent_names)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x labels if needed
            if len(max(agent_names, key=len)) > 10:
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved: {save_path}")
    
    def plot_convergence_analysis(self, agents_data: Dict,
                                  save_name: str = 'convergence.png'):
        """
        Plot convergence analysis.
        
        Args:
            agents_data: {agent_name: {'rewards': [...]}}
            save_name: Output filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)
        
        # Plot 1: Smoothed learning curves
        for agent_name, data in agents_data.items():
            rewards = data.get('rewards', data.get('episode_rewards', []))
            episodes = range(len(rewards))
            
            smoothed = self._smooth_curve(rewards, window=100)
            ax1.plot(episodes, smoothed, label=agent_name, linewidth=2)
        
        ax1.set_xlabel('Episodes', fontsize=12)
        ax1.set_ylabel('Smoothed Reward', fontsize=12)
        ax1.set_title('Convergence: Smoothed Rewards', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Final performance (last 100 episodes)
        agent_names = list(agents_data.keys())
        final_means = []
        final_stds = []
        
        for agent_name in agent_names:
            rewards = agents_data[agent_name].get('rewards', 
                      agents_data[agent_name].get('episode_rewards', []))
            last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
            final_means.append(np.mean(last_100))
            final_stds.append(np.std(last_100))
        
        ax2.bar(agent_names, final_means, yerr=final_stds, 
               alpha=0.7, capsize=5, edgecolor='black')
        ax2.set_ylabel('Mean Reward (Last 100 Episodes)', fontsize=12)
        ax2.set_title('Final Performance', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved: {save_path}")
    
    def plot_transfer_learning(self, transfer_data: Dict,
                              save_name: str = 'transfer_learning.png'):
        """
        Plot transfer learning results.
        
        Args:
            transfer_data: Transfer learning results
            save_name: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)
        
        # Extract data
        similar_data = transfer_data.get('similar', {})
        different_data = transfer_data.get('different', {})
        
        # Plot 1: Initial Performance (Similar)
        self._plot_transfer_metric(axes[0, 0], similar_data, 
                                   'initial_success_rate',
                                   'Initial Success Rate - Similar Target',
                                   'Success Rate')
        
        # Plot 2: Final Performance (Similar)
        self._plot_transfer_metric(axes[0, 1], similar_data,
                                   'final_success_rate',
                                   'Final Success Rate - Similar Target',
                                   'Success Rate')
        
        # Plot 3: Learning Speed (Similar)
        self._plot_transfer_metric(axes[1, 0], similar_data,
                                   'learning_speed',
                                   'Learning Speed - Similar Target',
                                   'Episodes')
        
        # Plot 4: Comparison (Similar vs Different)
        ax = axes[1, 1]
        strategies = list(similar_data.keys())
        
        similar_final = [similar_data[s].get('final_success_rate', 0) 
                        for s in strategies]
        different_final = [different_data.get(s, {}).get('final_success_rate', 0)
                          for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        ax.bar(x - width/2, similar_final, width, label='Similar', alpha=0.7)
        ax.bar(x + width/2, different_final, width, label='Different', alpha=0.7)
        
        ax.set_ylabel('Final Success Rate', fontsize=12)
        ax.set_title('Similar vs Different Target', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved: {save_path}")
    
    def _plot_transfer_metric(self, ax, data: Dict, metric: str,
                             title: str, ylabel: str):
        """Helper to plot transfer learning metric."""
        strategies = list(data.keys())
        values = [data[s].get(metric, 0) for s in strategies]
        
        bars = ax.bar(strategies, values, alpha=0.7, edgecolor='black')
        
        # Color bars
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(strategies)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_visit_frequency(self, visit_counts: Dict[Tuple[int, int], int],
                            maze: np.ndarray,
                            save_name: str = 'visit_frequency.png'):
        """
        Plot state visit frequency heatmap.
        
        Args:
            visit_counts: {(x, y): count}
            maze: Maze array
            save_name: Output filename
        """
        height, width = maze.shape
        visit_grid = np.zeros((height, width))
        
        for (x, y), count in visit_counts.items():
            if 0 <= x < height and 0 <= y < width:
                visit_grid[x, y] = count
        
        fig, ax = plt.subplots(figsize=(8, 8), dpi=self.dpi)
        
        im = ax.imshow(visit_grid, cmap='YlOrRd', interpolation='nearest')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Visit Count', fontsize=12)
        
        # Draw walls
        for i in range(height):
            for j in range(width):
                if maze[i, j] == MazeGenerator.WALL:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                              fill=True, color='black', alpha=0.8))
        
        ax.set_title('State Visit Frequency', fontsize=14, fontweight='bold')
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.grid(False)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved: {save_path}")
    
    def _smooth_curve(self, data: List[float], window: int = 50) -> np.ndarray:
        """Apply moving average smoothing."""
        if len(data) < window:
            return np.array(data)
        
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        # Pad to keep same length
        pad_size = len(data) - len(smoothed)
        smoothed = np.concatenate([data[:pad_size], smoothed])
        
        return smoothed


def generate_all_plots():
    """Generate all visualization plots."""
    print("\n" + "=" * 60)
    print("Generating All Visualization Plots")
    print("Student ID: 40415484")
    print("=" * 60)
    
    viz = Visualizer()
    
    # TODO: Load actual data and generate plots
    # This is a placeholder showing the workflow
    
    print("\n[NOTE] To generate plots with real data:")
    print("  1. Run experiments to generate training logs")
    print("  2. Load data from results/raw_data/")
    print("  3. Call viz.plot_*() methods")
    print("  4. Check results/figures/ for outputs")
    
    print("\n" + "=" * 60)
    print("[OK] Visualizer initialized and ready")
    print("=" * 60)


if __name__ == '__main__':
    generate_all_plots()
