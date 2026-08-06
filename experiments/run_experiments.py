"""
Run all experiments for RL Final Project
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Run all experiments"""
    print("=" * 60)
    print("RL Final Project - Experiments")
    print("Student ID: 40415484")
    print("=" * 60)
    
    # TODO: Implement experiments in later phases
    print("\n⚠️  Experiments will be implemented in upcoming phases:")
    print("  - Phase 2: Environment generation and validation")
    print("  - Phase 3: Value Iteration experiments")
    print("  - Phase 4: Q-Learning experiments")
    print("  - Phase 5: SARSA(λ) experiments")
    print("  - Phase 6: Comparative analysis")
    print("  - Phase 7: Transfer learning experiments")
    print("  - Phase 9: Visual analytics generation")
    
    print("\n✅ Project structure is ready!")


if __name__ == '__main__':
    main()
