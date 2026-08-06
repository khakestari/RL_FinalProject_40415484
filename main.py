#!/usr/bin/env python3
"""
RL Final Project - Main Entry Point
Student ID: 40415484
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


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
        default=1000,
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
        print(f"🏋️ Training {args.algorithm} on {args.map} map...")
        # TODO: Implement training mode
        print("Training mode will be implemented in next phases.")
    
    elif args.mode == 'evaluate':
        print(f"📊 Evaluating {args.algorithm} on {args.map} map...")
        # TODO: Implement evaluation mode
        print("Evaluation mode will be implemented in next phases.")
    
    elif args.mode == 'experiment':
        print("🔬 Running experiments...")
        from experiments.run_experiments import main as experiment_main
        experiment_main()
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
