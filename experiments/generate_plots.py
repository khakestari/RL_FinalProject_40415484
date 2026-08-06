"""
Generate All Plots from Experiment Data
Creates comprehensive visualizations for report
"""

import sys
from pathlib import Path
import json
import pickle
import csv
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.visualize import Visualizer
from environments.generator import MazeGenerator


def load_training_log(filepath: str) -> dict:
    """Load training log from CSV."""
    data = {
        'episodes': [],
        'reward': [],
        'length': [],
        'success_rate': []
    }
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data['episodes'].append(int(row['episode']))
                data['reward'].append(float(row.get('reward', row.get('mean_reward', 0))))
                data['length'].append(float(row.get('length', row.get('mean_length', 0))))
                data['success_rate'].append(float(row.get('success_rate', 0)))
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
    
    return data


def load_comparison_data(filepath: str) -> dict:
    """Load comparison results from JSON."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return {}


def load_agent_model(filepath: str):
    """Load trained agent model."""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def extract_value_map(agent, state_keys: list = ['x', 'y']) -> dict:
    """Extract value map from agent."""
    value_map = {}
    
    try:
        if isinstance(agent, dict):
            if 'V' in agent:
                for state, value in agent['V'].items():
                    if len(state) >= 2:
                        value_map[(state[0], state[1])] = value
            elif 'Q' in agent:
                for state, q_values in agent['Q'].items():
                    if len(state) >= 2:
                        value_map[(state[0], state[1])] = max(q_values)
        else:
            if hasattr(agent, 'V'):
                # Value Iteration
                for state, value in agent.V.items():
                    if len(state) >= 2:
                        value_map[(state[0], state[1])] = value
            
            elif hasattr(agent, 'Q'):
                # Q-Learning or SARSA
                for state in agent.Q.keys():
                    if len(state) >= 2:
                        q_values = agent.Q[state]
                        value_map[(state[0], state[1])] = max(q_values)
    
    except Exception as e:
        print(f"Warning: Could not extract value map: {e}")
    
    return value_map


def extract_policy_map(agent) -> dict:
    """Extract policy map from agent."""
    policy_map = {}
    
    try:
        if isinstance(agent, dict):
            if 'policy' in agent:
                for state, action in agent['policy'].items():
                    if len(state) >= 2:
                        policy_map[(state[0], state[1])] = action
            elif 'Q' in agent:
                for state, q_values in agent['Q'].items():
                    if len(state) >= 2:
                        action = int(np.argmax(q_values))
                        policy_map[(state[0], state[1])] = action
        else:
            if hasattr(agent, 'get_action'):
                # Try to extract policy
                if hasattr(agent, 'V'):
                    # Value Iteration - has explicit policy
                    for state in agent.V.keys():
                        if len(state) >= 2:
                            action = agent.get_action(state)
                            policy_map[(state[0], state[1])] = action
                
                elif hasattr(agent, 'Q'):
                    # Q-Learning or SARSA
                    for state in agent.Q.keys():
                        if len(state) >= 2:
                            q_values = agent.Q[state]
                            action = int(np.argmax(q_values))
                            policy_map[(state[0], state[1])] = action
    
    except Exception as e:
        print(f"Warning: Could not extract policy: {e}")
    
    return policy_map


def main():
    """Generate all plots."""
    print("\n" + "=" * 60)
    print("Generate All Plots for Report")
    print("Student ID: 40415484")
    print("=" * 60)
    
    viz = Visualizer(save_dir="results/figures")
    
    # Load maze
    print("\n[1/8] Loading maze...")
    try:
        maze, metadata = MazeGenerator.load_maze("environments/maps/maze_seed8_size15.npz")
        print("      [OK] Maze loaded")
    except:
        print("      [SKIP] No maze found")
        maze, metadata = None, {}
    
    # 1. Learning Curves
    print("\n[2/8] Generating learning curves...")
    try:
        agents_data = {}
        
        # Load training logs
        log_files = {
            'Value Iteration': 'results/raw_data/value_iteration_training.csv',
            'Q-Learning': 'results/raw_data/q_learning_training.csv',
            'SARSA(λ)': 'results/raw_data/sarsa_lambda_training.csv',
        }
        
        for agent_name, filepath in log_files.items():
            if Path(filepath).exists():
                data = load_training_log(filepath)
                if data['episodes']:
                    agents_data[agent_name] = data
        
        if agents_data:
            viz.plot_learning_curves(agents_data, metric='reward',
                                    save_name='learning_curves_reward.png')
            viz.plot_learning_curves(agents_data, metric='length',
                                    save_name='learning_curves_length.png')
            print("      [OK] Learning curves generated")
        else:
            print("      [SKIP] No training data found")
    
    except Exception as e:
        print(f"      [ERROR] {e}")
    
    # 2. Value Heatmaps
    print("\n[3/8] Generating value heatmaps...")
    try:
        if maze is not None:
            model_files = {
                'Value Iteration': 'results/models/value_iteration_gamma099.pkl',
                'Q-Learning': 'results/models/q_learning_source.pkl',
                'SARSA(λ)': 'results/models/sarsa_lambda_07.pkl',
            }
            
            for agent_name, filepath in model_files.items():
                if Path(filepath).exists():
                    agent = load_agent_model(filepath)
                    if agent:
                        value_map = extract_value_map(agent)
                        if value_map:
                            safe_name = agent_name.replace('(', '').replace(')', '').replace(' ', '_').lower()
                            viz.plot_value_heatmap(
                                value_map, maze,
                                title=f'{agent_name} - Value Function',
                                save_name=f'heatmap_{safe_name}.png'
                            )
            print("      [OK] Heatmaps generated")
        else:
            print("      [SKIP] No maze available")
    
    except Exception as e:
        print(f"      [ERROR] {e}")
    
    # 3. Policy Maps
    print("\n[4/8] Generating policy maps...")
    try:
        if maze is not None and metadata:
            for agent_name, filepath in model_files.items():
                if Path(filepath).exists():
                    agent = load_agent_model(filepath)
                    if agent:
                        policy_map = extract_policy_map(agent)
                        if policy_map:
                            safe_name = agent_name.replace('(', '').replace(')', '').replace(' ', '_').lower()
                            viz.plot_policy_map(
                                policy_map, maze, metadata,
                                title=f'{agent_name} - Policy',
                                save_name=f'policy_{safe_name}.png'
                            )
            print("      [OK] Policy maps generated")
        else:
            print("      [SKIP] No maze/metadata available")
    
    except Exception as e:
        print(f"      [ERROR] {e}")
    
    # 4. Comparison Chart
    print("\n[5/8] Generating comparison chart...")
    try:
        comparison_file = 'results/raw_data/comparison_results.json'
        if Path(comparison_file).exists():
            comparison_data = load_comparison_data(comparison_file)
            
            if comparison_data:
                # Extract metrics
                agents_comparison = {}
                for algo, data in comparison_data.items():
                    if isinstance(data, dict) and 'evaluation' in data:
                        eval_data = data['evaluation']
                        agents_comparison[algo] = {
                            'success_rate': eval_data.get('success_rate', 0),
                            'mean_reward': eval_data.get('mean_reward', 0),
                            'mean_length': eval_data.get('mean_length', 0),
                        }
                
                if agents_comparison:
                    viz.plot_comparison_bar(agents_comparison,
                                          save_name='comparison_algorithms.png')
                    print("      [OK] Comparison chart generated")
                else:
                    print("      [SKIP] No valid comparison data")
        else:
            print("      [SKIP] No comparison file found")
    
    except Exception as e:
        print(f"      [ERROR] {e}")
    
    # 5. Convergence Analysis
    print("\n[6/8] Generating convergence analysis...")
    try:
        if agents_data:
            # Convert format for convergence plot
            convergence_data = {}
            for agent_name, data in agents_data.items():
                convergence_data[agent_name] = {
                    'rewards': data['reward']
                }
            
            viz.plot_convergence_analysis(convergence_data,
                                        save_name='convergence_analysis.png')
            print("      [OK] Convergence analysis generated")
        else:
            print("      [SKIP] No training data available")
    
    except Exception as e:
        print(f"      [ERROR] {e}")
    
    # 6. Transfer Learning Plots
    print("\n[7/8] Generating transfer learning plots...")
    try:
        transfer_file = 'results/raw_data/transfer_learning.json'
        if Path(transfer_file).exists():
            transfer_data = load_comparison_data(transfer_file)
            
            if 'transfer' in transfer_data:
                viz.plot_transfer_learning(transfer_data['transfer'],
                                          save_name='transfer_learning.png')
                print("      [OK] Transfer learning plots generated")
            else:
                print("      [SKIP] No transfer data in file")
        else:
            print("      [SKIP] No transfer learning file found")
    
    except Exception as e:
        print(f"      [ERROR] {e}")
    
    # 7. Visit Frequency (if available)
    print("\n[8/8] Generating visit frequency map...")
    try:
        # This would need to be collected during training
        # For now, skip if not available
        print("      [SKIP] Visit frequency data not available")
    
    except Exception as e:
        print(f"      [ERROR] {e}")
    
    print("\n" + "=" * 60)
    print("[OK] Plot generation completed!")
    print(f"Check results/figures/ for output files")
    print("=" * 60)


if __name__ == '__main__':
    main()
