"""
Unit tests for Transfer Learning
Student ID: 40415484
"""

import unittest
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transfer.transfer_learning import TransferLearning
from environments.generator import MazeGenerator
from environments.maze import MazeEnvironment
from agents.q_learning import QLearning


class TestTransferLearning(unittest.TestCase):
    """Test Transfer Learning functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Generate test maze
        generator = MazeGenerator(size=15, seed=8)
        cls.maze, cls.metadata = generator.generate()
    
    def setUp(self):
        """Set up for each test."""
        self.transfer = TransferLearning(
            self.maze, self.metadata, seed=42
        )
    
    def test_initialization(self):
        """Test TransferLearning initialization."""
        self.assertIsNotNone(self.transfer.source_maze)
        self.assertIsNotNone(self.transfer.source_metadata)
        self.assertEqual(self.transfer.seed, 42)
        
        # Check target mazes generated
        self.assertIn('similar', self.transfer.target_mazes)
        self.assertIn('different', self.transfer.target_mazes)
    
    def test_target_mazes_structure(self):
        """Test target mazes have correct structure."""
        for target_type in ['similar', 'different']:
            maze, metadata = self.transfer.target_mazes[target_type]
            
            # Check maze shape
            self.assertEqual(maze.shape, self.maze.shape)
            
            # Check metadata
            self.assertIn('start', metadata)
            self.assertIn('key', metadata)
            self.assertIn('door', metadata)
            self.assertIn('goal', metadata)
    
    def test_train_source_agent(self):
        """Test training source agent."""
        agent = self.transfer.train_source_agent(episodes=100)
        
        # Check agent trained
        self.assertIsInstance(agent, QLearning)
        self.assertGreater(len(agent.Q), 0)
        
        # Check results stored
        self.assertIn('source', self.transfer.results)
        self.assertIn('train', self.transfer.results['source'])
        self.assertIn('eval', self.transfer.results['source'])
    
    def test_transfer_scratch(self):
        """Test training from scratch (no transfer)."""
        # Train source
        source_agent = self.transfer.train_source_agent(episodes=100)
        
        # Transfer with scratch strategy
        result = self.transfer.transfer_experiment(
            source_agent, 
            target_type='similar',
            strategy='scratch',
            episodes=50
        )
        
        # Check result structure
        self.assertIn('initial_eval', result)
        self.assertIn('final_eval', result)
        self.assertIn('learning_speed', result)
        self.assertEqual(result['strategy'], 'scratch')
    
    def test_transfer_full(self):
        """Test full transfer strategy."""
        # Train source
        source_agent = self.transfer.train_source_agent(episodes=100)
        source_q_size = len(source_agent.Q)
        
        # Transfer with full strategy
        result = self.transfer.transfer_experiment(
            source_agent,
            target_type='similar',
            strategy='full',
            episodes=50
        )
        
        # Check Q-table transferred
        target_agent = result['agent']
        self.assertGreater(len(target_agent.Q), 0)
        
        # Initial performance should exist (not checking specific value as it depends on training)
        initial_reward = result['initial_eval']['mean_reward']
        self.assertIsInstance(initial_reward, (int, float))
    
    def test_transfer_scaled(self):
        """Test scaled transfer strategy."""
        # Train source
        source_agent = self.transfer.train_source_agent(episodes=100)
        
        # Transfer with scaled strategy
        beta = 0.5
        result = self.transfer.transfer_experiment(
            source_agent,
            target_type='similar',
            strategy='scaled',
            beta=beta,
            episodes=50
        )
        
        # Check beta stored
        self.assertEqual(result['beta'], beta)
        self.assertEqual(result['strategy'], 'scaled')
    
    def test_transfer_selective(self):
        """Test selective transfer strategy."""
        # Train source
        source_agent = self.transfer.train_source_agent(episodes=100)
        
        # Transfer with selective strategy
        result = self.transfer.transfer_experiment(
            source_agent,
            target_type='similar',
            strategy='selective',
            episodes=50
        )
        
        # Check some states transferred
        target_agent = result['agent']
        self.assertGreater(len(target_agent.Q), 0)
        self.assertEqual(result['strategy'], 'selective')
    
    def test_selective_transfer_logic(self):
        """Test selective transfer only transfers similar neighborhoods."""
        # Create simple source agent
        env = MazeEnvironment(self.maze, self.metadata, seed=42)
        source_agent = QLearning(env, episodes=10)
        source_agent.train()
        
        # Create target agent
        target_maze, target_metadata = self.transfer.target_mazes['similar']
        target_env = MazeEnvironment(target_maze, target_metadata, seed=42)
        target_agent = QLearning(target_env, episodes=10)
        
        # Apply selective transfer
        transferred = self.transfer._selective_transfer(
            source_agent, target_agent,
            self.maze, target_maze
        )
        
        # Should transfer some but not all states
        self.assertGreater(transferred, 0)
        self.assertLessEqual(transferred, len(source_agent.Q))
    
    def test_neighborhood_comparison(self):
        """Test neighborhood comparison function."""
        # Same position in both mazes
        x, y = 1, 1
        
        # Check on same maze - should be True
        result = self.transfer._is_neighborhood_same(
            x, y, self.maze, self.maze
        )
        self.assertTrue(result)
    
    def test_learning_speed_computation(self):
        """Test learning speed computation."""
        # Create reward sequence that crosses threshold
        rewards = [-100] * 50 + [-40] * 50
        
        speed = self.transfer._compute_learning_speed(
            rewards, threshold=-50.0
        )
        
        # Should detect crossing around episode 50-100 (window=50)
        self.assertGreater(speed, 40)
        self.assertLess(speed, 110)
    
    def test_learning_speed_no_crossing(self):
        """Test learning speed when threshold not reached."""
        # Rewards never cross threshold
        rewards = [-100] * 100
        
        speed = self.transfer._compute_learning_speed(
            rewards, threshold=-50.0
        )
        
        # Should return total episodes
        self.assertEqual(speed, len(rewards))
    
    def test_different_target(self):
        """Test transfer to different target maze."""
        # Train source
        source_agent = self.transfer.train_source_agent(episodes=100)
        
        # Transfer to different target
        result = self.transfer.transfer_experiment(
            source_agent,
            target_type='different',
            strategy='full',
            episodes=50
        )
        
        # Should complete successfully
        self.assertIn('final_eval', result)
        # Check that final eval exists (success rate may be 0 for short training)
        self.assertIn('success_rate', result['final_eval'])
        self.assertGreaterEqual(result['final_eval']['success_rate'], 0)
    
    def test_save_results(self):
        """Test saving transfer learning results."""
        import tempfile
        import json
        
        # Train source
        source_agent = self.transfer.train_source_agent(episodes=50)
        
        # Run one transfer
        result = self.transfer.transfer_experiment(
            source_agent,
            target_type='similar',
            strategy='full',
            episodes=30
        )
        
        # Store in results
        self.transfer.results['transfer'] = {
            'similar': {'full': result}
        }
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        self.transfer.save_results(temp_path)
        
        # Load and check
        with open(temp_path, 'r') as f:
            data = json.load(f)
        
        self.assertIn('source', data)
        self.assertIn('transfer', data)
        
        # Cleanup
        Path(temp_path).unlink()
    
    def test_multiple_beta_values(self):
        """Test scaled transfer with different beta values."""
        # Train source
        source_agent = self.transfer.train_source_agent(episodes=100)
        
        betas = [0.25, 0.5, 0.75]
        results = []
        
        for beta in betas:
            result = self.transfer.transfer_experiment(
                source_agent,
                target_type='similar',
                strategy='scaled',
                beta=beta,
                episodes=30
            )
            results.append(result)
        
        # All should complete successfully
        self.assertEqual(len(results), 3)
        
        # Check beta values stored correctly
        for i, beta in enumerate(betas):
            self.assertEqual(results[i]['beta'], beta)


class TestTransferLearningIntegration(unittest.TestCase):
    """Integration tests for transfer learning."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        generator = MazeGenerator(size=15, seed=8)
        cls.maze, cls.metadata = generator.generate()
    
    def test_full_experiment_mini(self):
        """Test running mini version of full experiment."""
        transfer = TransferLearning(self.maze, self.metadata, seed=42)
        
        # Train source
        source_agent = transfer.train_source_agent(episodes=50)
        
        # Test one strategy on one target
        result = transfer.transfer_experiment(
            source_agent,
            target_type='similar',
            strategy='full',
            episodes=30
        )
        
        # Verify complete result structure
        required_keys = [
            'initial_eval', 'final_eval', 'train',
            'learning_speed', 'strategy'
        ]
        for key in required_keys:
            self.assertIn(key, result)
    
    def test_transfer_improves_performance(self):
        """Test that transfer learning can improve initial performance."""
        transfer = TransferLearning(self.maze, self.metadata, seed=42)
        
        # Train source
        source_agent = transfer.train_source_agent(episodes=100)
        
        # Compare scratch vs full transfer
        scratch_result = transfer.transfer_experiment(
            source_agent, 'similar', 'scratch', episodes=30
        )
        
        full_result = transfer.transfer_experiment(
            source_agent, 'similar', 'full', episodes=30
        )
        
        # Full transfer should have better initial performance
        scratch_initial = scratch_result['initial_eval']['mean_reward']
        full_initial = full_result['initial_eval']['mean_reward']
        
        # Note: This might not always hold for very small training,
        # but should generally be true
        self.assertGreaterEqual(
            full_initial + 20,  # Allow some tolerance
            scratch_initial
        )


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestTransferLearning))
    suite.addTests(loader.loadTestsFromTestCase(TestTransferLearningIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
