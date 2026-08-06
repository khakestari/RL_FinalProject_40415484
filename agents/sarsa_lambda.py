import numpy as np
import time
from typing import Dict, Tuple, List
import pickle
from pathlib import Path
from collections import defaultdict
import csv


class SarsaLambda:
    """
    SARSA(λ) algorithm with eligibility traces.
    
    On-policy TD control with eligibility traces:
    δ_t = r_{t+1} + γ*Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
    E_t(s,a) = γλ*E_{t-1}(s,a) + 1{s=s_t, a=a_t}
    Q(s,a) ← Q(s,a) + α*δ_t*E_t(s,a)
    """
    
    def __init__(self, env,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 lambda_param: float = 0.7,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 trace_type: str = 'accumulating',
                 episodes: int = 5000,
                 max_steps: int = 500,
                 verbose: bool = True,
                 log_every: int = 100):
        """
        Initialize SARSA(λ) agent.
        
        Args:
            env: MazeEnvironment instance
            alpha: Learning rate
            gamma: Discount factor
            lambda_param: Trace decay parameter (0 ≤ λ ≤ 1)
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for epsilon
            trace_type: 'accumulating' or 'replacing'
            episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Print progress
            log_every: Log progress every N episodes
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_param = lambda_param
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.trace_type = trace_type
        self.episodes = episodes
        self.max_steps = max_steps
        self.verbose = verbose
        self.log_every = log_every
        
        # Initialize Q-table and eligibility traces
        self.Q = defaultdict(lambda: np.zeros(4))
        self.E = defaultdict(lambda: np.zeros(4))  # Eligibility traces
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self.wall_collisions_per_episode = []
        self.penalty_visits_per_episode = []
        self.epsilon_history = []
        
        # TD error and trace logging (for analysis)
        self.td_error_log = []
        self.trace_log = []
        
        # Current epsilon
        self.epsilon = epsilon_start
        
        # Training time
        self.training_time = 0.0
    
    def train(self) -> Dict:
        """
        Train SARSA(λ) agent.
        
        Returns:
            Dictionary with training statistics
        """
        if self.verbose:
            print("=" * 60)
            print("SARSA(λ) Training")
            print("=" * 60)
            print(f"Episodes: {self.episodes}")
            print(f"Max steps per episode: {self.max_steps}")
            print(f"Learning rate (α): {self.alpha}")
            print(f"Discount factor (γ): {self.gamma}")
            print(f"Lambda (λ): {self.lambda_param}")
            print(f"Trace type: {self.trace_type}")
            print(f"Epsilon: {self.epsilon_start} → {self.epsilon_end}")
            print()
        
        start_time = time.time()
        
        for episode in range(self.episodes):
            episode_reward, episode_length, success, stats = self._run_episode(episode)
            
            # Store statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_success.append(1 if success else 0)
            self.wall_collisions_per_episode.append(stats['wall_collisions'])
            self.penalty_visits_per_episode.append(stats['penalty_visits'])
            self.epsilon_history.append(self.epsilon)
            
            # Update epsilon
            self._update_epsilon(episode)
            
            # Log progress
            if self.verbose and (episode + 1) % self.log_every == 0:
                recent_rewards = self.episode_rewards[-self.log_every:]
                recent_success = self.episode_success[-self.log_every:]
                recent_lengths = self.episode_lengths[-self.log_every:]
                
                print(f"Episode {episode + 1:5d} | "
                      f"Reward: {np.mean(recent_rewards):7.2f} | "
                      f"Length: {np.mean(recent_lengths):6.1f} | "
                      f"Success: {np.mean(recent_success):5.1%} | "
                      f"ε: {self.epsilon:.4f}")
        
        self.training_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'=' * 60}")
            print("Training Complete")
            print(f"{'=' * 60}")
            print(f"Total time: {self.training_time:.2f}s")
            print(f"Episodes: {self.episodes}")
            print(f"Q-table size: {len(self.Q)} states")
            print(f"Final ε: {self.epsilon:.4f}")
            
            # Final statistics
            last_100 = min(100, len(self.episode_rewards))
            print(f"\nLast {last_100} episodes:")
            print(f"  Mean reward: {np.mean(self.episode_rewards[-last_100:]):.2f}")
            print(f"  Success rate: {np.mean(self.episode_success[-last_100:]):.1%}")
            print(f"  Mean length: {np.mean(self.episode_lengths[-last_100:]):.1f}")
        
        return {
            'episodes': self.episodes,
            'training_time': self.training_time,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_success': self.episode_success,
            'wall_collisions': self.wall_collisions_per_episode,
            'penalty_visits': self.penalty_visits_per_episode,
            'epsilon_history': self.epsilon_history,
            'q_table_size': len(self.Q),
            'alpha': self.alpha,
            'gamma': self.gamma,
            'lambda': self.lambda_param,
            'trace_type': self.trace_type,
        }
    
    def _run_episode(self, episode_num: int) -> Tuple[float, int, bool, Dict]:
        """
        Run a single episode using SARSA(λ).
        
        Returns:
            episode_reward, episode_length, success, stats
        """
        # Reset eligibility traces at start of episode
        self.E = defaultdict(lambda: np.zeros(4))
        
        state = self.env.reset()
        action = self._choose_action(state)
        
        total_reward = 0.0
        steps = 0
        success = False
        
        for step in range(self.max_steps):
            # Take action
            next_state, reward, done, info = self.env.step(action)
            
            # Choose next action (on-policy)
            next_action = self._choose_action(next_state)
            
            # Compute TD error
            td_error = self._compute_td_error(state, action, reward, 
                                             next_state, next_action, done)
            
            # Update eligibility trace for current state-action
            self._update_trace(state, action)
            
            # Update Q-values for all states using eligibility traces
            self._update_q_with_traces(td_error)
            
            # Log for analysis (first few episodes)
            if episode_num < 3 and step < 10:
                self._log_td_and_trace(episode_num, step, state, action,
                                      td_error, self.E[state][action])
            
            total_reward += reward
            steps += 1
            
            # Move to next state-action
            state = next_state
            action = next_action
            
            if done:
                if info.get('event') == 'goal_reached':
                    success = True
                break
        
        stats = {
            'wall_collisions': self.env.wall_collisions,
            'penalty_visits': self.env.penalty_visits,
        }
        
        return total_reward, steps, success, stats
    
    def _choose_action(self, state: Tuple) -> int:
        """
        Choose action using ε-greedy policy.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(4)
        else:
            return np.argmax(self.Q[state])
    
    def _compute_td_error(self, state: Tuple, action: int, reward: float,
                         next_state: Tuple, next_action: int, done: bool) -> float:
        """
        Compute TD error: δ = r + γ*Q(s',a') - Q(s,a)
        
        For SARSA (on-policy), we use the actual next action chosen by policy.
        """
        current_q = self.Q[state][action]
        
        if done:
            target_q = reward
        else:
            # SARSA uses Q(s',a') where a' is the action actually taken
            target_q = reward + self.gamma * self.Q[next_state][next_action]
        
        td_error = target_q - current_q
        return td_error
    
    def _update_trace(self, state: Tuple, action: int):
        """
        Update eligibility trace for current state-action.
        
        Accumulating traces: E(s,a) = γλE(s,a) + 1
        Replacing traces: E(s,a) = 1
        """
        if self.trace_type == 'accumulating':
            # Accumulating trace
            self.E[state][action] = self.gamma * self.lambda_param * \
                                   self.E[state][action] + 1.0
        elif self.trace_type == 'replacing':
            # Replacing trace
            self.E[state][action] = 1.0
        else:
            raise ValueError(f"Unknown trace type: {self.trace_type}")
    
    def _update_q_with_traces(self, td_error: float):
        """
        Update Q-values for all states using eligibility traces.
        
        For each state-action: Q(s,a) ← Q(s,a) + α*δ*E(s,a)
        Then decay traces: E(s,a) ← γλE(s,a)
        """
        # Update Q-values
        for state in list(self.E.keys()):
            for action in range(4):
                if self.E[state][action] > 1e-10:  # Only update if trace is non-zero
                    self.Q[state][action] += self.alpha * td_error * self.E[state][action]
        
        # Decay all traces
        for state in list(self.E.keys()):
            self.E[state] *= self.gamma * self.lambda_param
            
            # Remove traces that are too small (memory optimization)
            if np.all(self.E[state] < 1e-10):
                del self.E[state]
    
    def _update_epsilon(self, episode: int):
        """Update epsilon (exponential decay)."""
        self.epsilon = max(self.epsilon_end, 
                          self.epsilon * self.epsilon_decay)
    
    def _log_td_and_trace(self, episode: int, step: int, state: Tuple,
                         action: int, td_error: float, trace_value: float):
        """Log TD error and trace value for analysis."""
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        self.td_error_log.append({
            'episode': episode,
            'step': step,
            'state': state,
            'action': action_names[action],
            'td_error': td_error,
            'trace_value': trace_value,
            'q_value': self.Q[state][action]
        })
    
    def get_action(self, state: Tuple, greedy: bool = True) -> int:
        """
        Get action for a given state.
        
        Args:
            state: Current state
            greedy: If True, return best action; if False, use ε-greedy
            
        Returns:
            Action to take
        """
        if greedy:
            return np.argmax(self.Q[state])
        else:
            return self._choose_action(state)
    
    def evaluate(self, num_episodes: int = 100, max_steps: int = 500) -> Dict:
        """
        Evaluate the learned policy (greedy, no exploration).
        
        Args:
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary with evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        # Save current epsilon and set to 0 for evaluation
        old_epsilon = self.epsilon
        self.epsilon = 0.0
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            steps = 0
            
            for step in range(max_steps):
                action = self.get_action(state, greedy=True)
                next_state, reward, done, info = self.env.step(action)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    if info.get('event') == 'goal_reached':
                        success_count += 1
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        # Restore epsilon
        self.epsilon = old_epsilon
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': success_count / num_episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
        }
    
    def get_q_heatmap(self, has_key: int = 0, energy: int = None,
                     action: int = None) -> np.ndarray:
        """
        Get Q-values as 2D heatmap for visualization.
        """
        if energy is None:
            energy = self.env.max_energy
        
        heatmap = np.zeros((self.env.size, self.env.size))
        
        for x in range(self.env.size):
            for y in range(self.env.size):
                state = (x, y, has_key, energy)
                
                if action is None:
                    heatmap[x, y] = np.max(self.Q[state])
                else:
                    heatmap[x, y] = self.Q[state][action]
        
        return heatmap
    
    def get_policy_map(self, has_key: int = 0, energy: int = None) -> np.ndarray:
        """
        Get learned policy as 2D map for visualization.
        """
        if energy is None:
            energy = self.env.max_energy
        
        policy_map = np.full((self.env.size, self.env.size), -1, dtype=int)
        
        for x in range(self.env.size):
            for y in range(self.env.size):
                state = (x, y, has_key, energy)
                
                if state in self.Q and np.any(self.Q[state] != 0):
                    policy_map[x, y] = np.argmax(self.Q[state])
                elif not self.env._is_valid_position((x, y)):
                    policy_map[x, y] = -1
        
        return policy_map
    
    def save(self, filepath: str):
        """Save the trained agent."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'Q': dict(self.Q),
            'alpha': self.alpha,
            'gamma': self.gamma,
            'lambda_param': self.lambda_param,
            'trace_type': self.trace_type,
            'epsilon': self.epsilon,
            'episodes': self.episodes,
            'training_time': self.training_time,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_success': self.episode_success,
            'epsilon_history': self.epsilon_history,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        if self.verbose:
            print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained agent."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.Q = defaultdict(lambda: np.zeros(4))
        for state, values in data['Q'].items():
            self.Q[state] = values
        
        self.alpha = data['alpha']
        self.gamma = data['gamma']
        self.lambda_param = data['lambda_param']
        self.trace_type = data['trace_type']
        self.epsilon = data['epsilon']
        self.episodes = data['episodes']
        self.training_time = data['training_time']
        self.episode_rewards = data['episode_rewards']
        self.episode_lengths = data['episode_lengths']
        self.episode_success = data['episode_success']
        self.epsilon_history = data.get('epsilon_history', [])
        
        if self.verbose:
            print(f"✓ Model loaded from {filepath}")
    
    def save_training_log(self, filepath: str):
        """Save detailed training log as CSV."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'length', 'success',
                           'wall_collisions', 'penalty_visits', 'epsilon'])
            
            for i in range(len(self.episode_rewards)):
                writer.writerow([
                    i + 1,
                    self.episode_rewards[i],
                    self.episode_lengths[i],
                    self.episode_success[i],
                    self.wall_collisions_per_episode[i],
                    self.penalty_visits_per_episode[i],
                    self.epsilon_history[i]
                ])
        
        if self.verbose:
            print(f"✓ Training log saved to {filepath}")
    
    def save_td_trace_log(self, filepath: str):
        """Save TD error and trace log for analysis."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'step', 'state', 'action',
                           'td_error', 'trace_value', 'q_value'])
            
            for log in self.td_error_log:
                writer.writerow([
                    log['episode'],
                    log['step'],
                    str(log['state']),
                    log['action'],
                    log['td_error'],
                    log['trace_value'],
                    log['q_value']
                ])
        
        if self.verbose:
            print(f"✓ TD/Trace log saved to {filepath}")


def compare_lambda_values(env, lambda_values: List[float],
                         episodes: int = 2000, verbose: bool = False) -> Dict:
    """
    Compare different lambda values.
    
    Args:
        env: Environment instance
        lambda_values: List of lambda values to test
        episodes: Number of training episodes
        verbose: Print progress
        
    Returns:
        Dictionary with results for each lambda
    """
    results = {}
    
    print("=" * 60)
    print("Comparing Lambda (λ) Values")
    print("=" * 60)
    print(f"Testing λ values: {lambda_values}")
    print()
    
    for lambda_val in lambda_values:
        print(f"🎯 Testing λ = {lambda_val}")
        print("-" * 60)
        
        agent = SarsaLambda(env, lambda_param=lambda_val, 
                           episodes=episodes, verbose=verbose,
                           log_every=200)
        train_stats = agent.train()
        eval_stats = agent.evaluate(num_episodes=50)
        
        results[lambda_val] = {
            'train': train_stats,
            'eval': eval_stats,
            'agent': agent
        }
        
        print(f"   Training time: {train_stats['training_time']:.2f}s")
        print(f"   Success rate: {eval_stats['success_rate']:.2%}")
        print(f"   Mean reward: {eval_stats['mean_reward']:.2f}")
        print()
    
    # Summary
    print("=" * 60)
    print("Lambda Comparison Summary")
    print("=" * 60)
    print(f"{'Lambda':<10} {'Success Rate':<15} {'Mean Reward':<15} {'Training Time':<15}")
    print("-" * 60)
    
    for lambda_val, res in results.items():
        print(f"{lambda_val:<10.1f} "
              f"{res['eval']['success_rate']:<15.1%} "
              f"{res['eval']['mean_reward']:<15.2f} "
              f"{res['train']['training_time']:<15.2f}")
    
    return results
