import numpy as np
import time
from typing import Dict, Tuple, List
import pickle
from pathlib import Path
from collections import defaultdict


class ValueIteration:
    """
    Value Iteration algorithm for solving MDPs.
    
    Uses Bellman optimality equation:
    V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γ*V_k(s')]
    """
    
    def __init__(self, env, gamma: float = 0.99, theta: float = 1e-6, 
                 max_iterations: int = 1000, verbose: bool = True):
        """
        Initialize Value Iteration agent.
        
        Args:
            env: MazeEnvironment instance
            gamma: Discount factor (0 < gamma <= 1)
            theta: Convergence threshold
            max_iterations: Maximum number of iterations
            verbose: Print progress
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Get all states (for simplicity, we'll use a subset with energy sampling)
        self.states = self._get_relevant_states()
        
        # Initialize value function
        self.V = defaultdict(float)
        
        # Initialize policy (action for each state)
        self.policy = {}
        
        # Training statistics
        self.convergence_history = []
        self.iteration_times = []
        self.num_iterations = 0
        self.training_time = 0.0
        
    def _get_relevant_states(self) -> List[Tuple[int, int, int, int]]:
        """
        Get relevant states for the maze.
        To make computation tractable, we sample energy levels.
        """
        states = []
        
        # Sample energy levels (every 25 units + boundaries)
        energy_levels = list(range(0, self.env.max_energy + 1, 25))
        if self.env.max_energy not in energy_levels:
            energy_levels.append(self.env.max_energy)
        
        for x in range(self.env.size):
            for y in range(self.env.size):
                # Skip walls
                if not self.env._is_valid_position((x, y)):
                    continue
                
                for has_key in [0, 1]:
                    for energy in energy_levels:
                        state = (x, y, has_key, energy)
                        
                        # Skip terminal states (except goal)
                        if energy == 0 and (x, y) != self.env.goal_pos:
                            continue
                            
                        states.append(state)
        
        return states
    
    def train(self) -> Dict:
        """
        Run Value Iteration to find optimal value function and policy.
        
        Returns:
            Dictionary with training statistics
        """
        if self.verbose:
            print("=" * 60)
            print("Value Iteration Training")
            print("=" * 60)
            print(f"States: {len(self.states)}")
            print(f"Actions: 4 (UP, DOWN, LEFT, RIGHT)")
            print(f"Gamma: {self.gamma}")
            print(f"Theta: {self.theta}")
            print(f"Max iterations: {self.max_iterations}")
            print()
        
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            iter_start = time.time()
            
            # One iteration of value iteration
            delta = self._value_iteration_step()
            
            iter_time = time.time() - iter_start
            self.iteration_times.append(iter_time)
            self.convergence_history.append(delta)
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1:4d} | Delta: {delta:.6f} | Time: {iter_time:.3f}s")
            
            # Check convergence
            if delta < self.theta:
                self.num_iterations = iteration + 1
                if self.verbose:
                    print(f"\n✓ Converged after {self.num_iterations} iterations!")
                break
        else:
            self.num_iterations = self.max_iterations
            if self.verbose:
                print(f"\n⚠ Reached maximum iterations ({self.max_iterations})")
        
        self.training_time = time.time() - start_time
        
        # Extract policy from value function
        self._extract_policy()
        
        if self.verbose:
            print(f"\n{'=' * 60}")
            print("Training Complete")
            print(f"{'=' * 60}")
            print(f"Total time: {self.training_time:.2f}s")
            print(f"Iterations: {self.num_iterations}")
            print(f"Average time per iteration: {self.training_time/self.num_iterations:.3f}s")
            print(f"Final delta: {self.convergence_history[-1]:.6e}")
            print(f"Policy states: {len(self.policy)}")
        
        return {
            'num_iterations': self.num_iterations,
            'training_time': self.training_time,
            'final_delta': self.convergence_history[-1],
            'convergence_history': self.convergence_history,
            'iteration_times': self.iteration_times,
            'gamma': self.gamma,
            'theta': self.theta,
        }
    
    def _value_iteration_step(self) -> float:
        """
        Perform one iteration of value iteration.
        
        Returns:
            Maximum change in value function (delta)
        """
        delta = 0.0
        new_V = {}
        
        for state in self.states:
            # Skip terminal states
            if self.env.is_terminal_state(state):
                new_V[state] = 0.0
                continue
            
            # Get current value
            old_value = self.V[state]
            
            # Compute value for each action
            action_values = []
            for action in range(4):  # 4 actions: UP, DOWN, LEFT, RIGHT
                q_value = self._compute_q_value(state, action)
                action_values.append(q_value)
            
            # Take maximum
            new_value = max(action_values)
            new_V[state] = new_value
            
            # Update delta
            delta = max(delta, abs(new_value - old_value))
        
        # Update value function
        self.V.update(new_V)
        
        return delta
    
    def _compute_q_value(self, state: Tuple[int, int, int, int], action: int) -> float:
        """
        Compute Q(s, a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ*V(s')]
        
        Args:
            state: Current state (x, y, has_key, energy)
            action: Action to take
            
        Returns:
            Q-value for state-action pair
        """
        q_value = 0.0
        
        # Get transition probabilities
        transitions = self.env.get_transition_probabilities(state, action)
        
        for next_state, prob in transitions:
            # Get reward
            reward = self.env.get_reward(state, action, next_state)
            
            # Get next state value
            next_value = self.V.get(next_state, 0.0)
            
            # Add to Q-value
            q_value += prob * (reward + self.gamma * next_value)
        
        return q_value
    
    def _extract_policy(self):
        """
        Extract greedy policy from value function.
        π(s) = argmax_a Q(s,a)
        """
        self.policy = {}
        
        for state in self.states:
            # Skip terminal states
            if self.env.is_terminal_state(state):
                continue
            
            # Find best action
            best_action = None
            best_value = float('-inf')
            
            for action in range(4):
                q_value = self._compute_q_value(state, action)
                
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
            
            self.policy[state] = best_action
    
    def get_action(self, state: Tuple[int, int, int, int]) -> int:
        """
        Get action according to learned policy.
        
        Args:
            state: Current state
            
        Returns:
            Action to take
        """
        # Find closest state in policy (for energy interpolation)
        x, y, has_key, energy = state
        
        # Try exact match first
        if state in self.policy:
            return self.policy[state]
        
        # Find closest energy level
        best_match = None
        min_energy_diff = float('inf')
        
        for policy_state in self.policy:
            px, py, pk, pe = policy_state
            if px == x and py == y and pk == has_key:
                energy_diff = abs(pe - energy)
                if energy_diff < min_energy_diff:
                    min_energy_diff = energy_diff
                    best_match = policy_state
        
        if best_match is not None:
            return self.policy[best_match]
        
        # Default: random action
        return np.random.randint(4)
    
    def get_value(self, state: Tuple[int, int, int, int]) -> float:
        """Get value of a state."""
        return self.V.get(state, 0.0)
    
    def evaluate(self, num_episodes: int = 100, max_steps: int = 1000) -> Dict:
        """
        Evaluate the learned policy.
        
        Args:
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary with evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            steps = 0
            
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    # Check if reached goal
                    if info.get('event') == 'goal_reached':
                        success_count += 1
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': success_count / num_episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
        }
    
    def get_value_heatmap(self, has_key: int = 0, energy: int = None) -> np.ndarray:
        """
        Get value function as 2D heatmap for visualization.
        
        Args:
            has_key: Whether agent has key (0 or 1)
            energy: Energy level (None for max energy)
            
        Returns:
            2D array of values
        """
        if energy is None:
            energy = self.env.max_energy
        
        heatmap = np.zeros((self.env.size, self.env.size))
        
        for x in range(self.env.size):
            for y in range(self.env.size):
                state = (x, y, has_key, energy)
                heatmap[x, y] = self.get_value(state)
        
        return heatmap
    
    def get_policy_map(self, has_key: int = 0, energy: int = None) -> np.ndarray:
        """
        Get policy as 2D map for visualization.
        
        Args:
            has_key: Whether agent has key (0 or 1)
            energy: Energy level (None for max energy)
            
        Returns:
            2D array of actions (-1 for walls/terminal)
        """
        if energy is None:
            energy = self.env.max_energy
        
        policy_map = np.full((self.env.size, self.env.size), -1, dtype=int)
        
        for x in range(self.env.size):
            for y in range(self.env.size):
                state = (x, y, has_key, energy)
                
                if state in self.policy:
                    policy_map[x, y] = self.policy[state]
                elif not self.env._is_valid_position((x, y)):
                    policy_map[x, y] = -1
        
        return policy_map
    
    def save(self, filepath: str):
        """Save the trained agent."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'V': dict(self.V),
            'policy': self.policy,
            'gamma': self.gamma,
            'theta': self.theta,
            'num_iterations': self.num_iterations,
            'training_time': self.training_time,
            'convergence_history': self.convergence_history,
            'iteration_times': self.iteration_times,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        if self.verbose:
            print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained agent."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.V = defaultdict(float, data['V'])
        self.policy = data['policy']
        self.gamma = data['gamma']
        self.theta = data['theta']
        self.num_iterations = data['num_iterations']
        self.training_time = data['training_time']
        self.convergence_history = data['convergence_history']
        self.iteration_times = data['iteration_times']
        
        if self.verbose:
            print(f"✓ Model loaded from {filepath}")


def test_gamma_values(env, gamma_values: List[float], 
                     theta: float = 1e-6, verbose: bool = False) -> Dict:
    """
    Test different gamma (discount factor) values.
    
    Args:
        env: Environment instance
        gamma_values: List of gamma values to test
        theta: Convergence threshold
        verbose: Print progress
        
    Returns:
        Dictionary with results for each gamma
    """
    results = {}
    
    print("=" * 60)
    print("Testing Different Gamma Values")
    print("=" * 60)
    
    for gamma in gamma_values:
        print(f"\n🎯 Testing γ = {gamma}")
        print("-" * 60)
        
        agent = ValueIteration(env, gamma=gamma, theta=theta, verbose=verbose)
        train_stats = agent.train()
        eval_stats = agent.evaluate(num_episodes=50)
        
        results[gamma] = {
            'train': train_stats,
            'eval': eval_stats,
            'agent': agent
        }
        
        print(f"   Iterations: {train_stats['num_iterations']}")
        print(f"   Training time: {train_stats['training_time']:.2f}s")
        print(f"   Success rate: {eval_stats['success_rate']:.2%}")
        print(f"   Mean reward: {eval_stats['mean_reward']:.2f}")
    
    print("\n" + "=" * 60)
    print("Gamma Comparison Summary")
    print("=" * 60)
    print(f"{'Gamma':<10} {'Iterations':<12} {'Time(s)':<10} {'Success':<10} {'Reward':<10}")
    print("-" * 60)
    
    for gamma in gamma_values:
        stats = results[gamma]
        print(f"{gamma:<10.2f} "
              f"{stats['train']['num_iterations']:<12} "
              f"{stats['train']['training_time']:<10.2f} "
              f"{stats['eval']['success_rate']:<10.2%} "
              f"{stats['eval']['mean_reward']:<10.2f}")
    
    return results
