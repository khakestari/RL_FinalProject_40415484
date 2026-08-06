import numpy as np
import time
from typing import Dict, Tuple, List
import pickle
from pathlib import Path
from collections import defaultdict
import csv


class QLearning:
    """
    Q-Learning algorithm for solving MDPs without a model.
    
    Off-policy TD control:
    Q(s,a) ← Q(s,a) + α[r + γ*max_a' Q(s',a') - Q(s,a)]
    """
    
    def __init__(self, env, 
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 epsilon_decay_type: str = 'exponential',
                 episodes: int = 5000,
                 max_steps: int = 500,
                 verbose: bool = True,
                 log_every: int = 100):
        """
        Initialize Q-Learning agent.
        
        Args:
            env: MazeEnvironment instance
            alpha: Learning rate (0 < alpha <= 1)
            gamma: Discount factor (0 < gamma <= 1)
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_decay_type: 'exponential' or 'linear'
            episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Print progress
            log_every: Log progress every N episodes
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_type = epsilon_decay_type
        self.episodes = episodes
        self.max_steps = max_steps
        self.verbose = verbose
        self.log_every = log_every
        
        # Initialize Q-table
        self.Q = defaultdict(lambda: np.zeros(4))  # 4 actions
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self.wall_collisions_per_episode = []
        self.penalty_visits_per_episode = []
        self.epsilon_history = []
        self.q_update_log = []  # For manual inspection
        
        # Current epsilon
        self.epsilon = epsilon_start
        
        # Training time
        self.training_time = 0.0
    
    def train(self) -> Dict:
        """
        Train Q-Learning agent.
        
        Returns:
            Dictionary with training statistics
        """
        if self.verbose:
            print("=" * 60)
            print("Q-Learning Training")
            print("=" * 60)
            print(f"Episodes: {self.episodes}")
            print(f"Max steps per episode: {self.max_steps}")
            print(f"Learning rate (α): {self.alpha}")
            print(f"Discount factor (γ): {self.gamma}")
            print(f"Epsilon: {self.epsilon_start} → {self.epsilon_end}")
            print(f"Epsilon decay: {self.epsilon_decay_type}")
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
        }
    
    def _run_episode(self, episode_num: int) -> Tuple[float, int, bool, Dict]:
        """
        Run a single episode.
        
        Returns:
            episode_reward, episode_length, success, stats
        """
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        success = False
        
        for step in range(self.max_steps):
            # Choose action (ε-greedy)
            action = self._choose_action(state)
            
            # Take action
            next_state, reward, done, info = self.env.step(action)
            
            # Q-Learning update
            self._update_q(state, action, reward, next_state, done)
            
            # Log first few updates for manual inspection
            if episode_num < 3 and step < 5:
                self._log_q_update(episode_num, step, state, action, 
                                  reward, next_state, done)
            
            total_reward += reward
            steps += 1
            state = next_state
            
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
        
        With probability ε: explore (random action)
        With probability 1-ε: exploit (best action)
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(4)
        else:
            # Exploit: best action
            return np.argmax(self.Q[state])
    
    def _update_q(self, state: Tuple, action: int, reward: float, 
                  next_state: Tuple, done: bool):
        """
        Update Q-value using Q-Learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ*max_a' Q(s',a') - Q(s,a)]
        """
        # Current Q-value
        current_q = self.Q[state][action]
        
        # Target Q-value
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.Q[next_state])
        
        # TD error
        td_error = target_q - current_q
        
        # Update Q-value
        self.Q[state][action] = current_q + self.alpha * td_error
    
    def _update_epsilon(self, episode: int):
        """Update epsilon based on decay strategy."""
        if self.epsilon_decay_type == 'exponential':
            # Exponential decay: ε = ε * decay
            self.epsilon = max(self.epsilon_end, 
                             self.epsilon * self.epsilon_decay)
        
        elif self.epsilon_decay_type == 'linear':
            # Linear decay: ε = start - (start - end) * (episode / episodes)
            decay_amount = (self.epsilon_start - self.epsilon_end) / self.episodes
            self.epsilon = max(self.epsilon_end, 
                             self.epsilon_start - decay_amount * (episode + 1))
    
    def _log_q_update(self, episode: int, step: int, state: Tuple, 
                     action: int, reward: float, next_state: Tuple, done: bool):
        """Log Q-update for manual inspection."""
        current_q = self.Q[state][action]
        
        if done:
            target_q = reward
            max_next_q = 0.0
        else:
            max_next_q = np.max(self.Q[next_state])
            target_q = reward + self.gamma * max_next_q
        
        td_error = target_q - current_q
        new_q = current_q + self.alpha * td_error
        
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        self.q_update_log.append({
            'episode': episode,
            'step': step,
            'state': state,
            'action': action_names[action],
            'reward': reward,
            'current_q': current_q,
            'max_next_q': max_next_q,
            'target_q': target_q,
            'td_error': td_error,
            'new_q': new_q,
            'done': done
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
    
    def get_q_value(self, state: Tuple, action: int) -> float:
        """Get Q-value for state-action pair."""
        return self.Q[state][action]
    
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
        
        Args:
            has_key: Whether agent has key (0 or 1)
            energy: Energy level (None for max energy)
            action: Specific action (None for max over actions)
            
        Returns:
            2D array of Q-values
        """
        if energy is None:
            energy = self.env.max_energy
        
        heatmap = np.zeros((self.env.size, self.env.size))
        
        for x in range(self.env.size):
            for y in range(self.env.size):
                state = (x, y, has_key, energy)
                
                if action is None:
                    # Max Q-value over all actions
                    heatmap[x, y] = np.max(self.Q[state])
                else:
                    # Q-value for specific action
                    heatmap[x, y] = self.Q[state][action]
        
        return heatmap
    
    def get_policy_map(self, has_key: int = 0, energy: int = None) -> np.ndarray:
        """
        Get learned policy as 2D map for visualization.
        
        Args:
            has_key: Whether agent has key (0 or 1)
            energy: Energy level (None for max energy)
            
        Returns:
            2D array of actions (-1 for walls/unvisited)
        """
        if energy is None:
            energy = self.env.max_energy
        
        policy_map = np.full((self.env.size, self.env.size), -1, dtype=int)
        
        for x in range(self.env.size):
            for y in range(self.env.size):
                state = (x, y, has_key, energy)
                
                # Check if state has been visited
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
            'epsilon': self.epsilon,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_decay_type': self.epsilon_decay_type,
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
        
        # Restore Q-table with proper defaultdict
        self.Q = defaultdict(lambda: np.zeros(4))
        for state, values in data['Q'].items():
            self.Q[state] = values
        
        self.alpha = data['alpha']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.epsilon_start = data.get('epsilon_start', 1.0)
        self.epsilon_end = data.get('epsilon_end', 0.01)
        self.epsilon_decay = data.get('epsilon_decay', 0.995)
        self.epsilon_decay_type = data.get('epsilon_decay_type', 'exponential')
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
    
    def save_q_update_log(self, filepath: str):
        """Save Q-update log for manual inspection."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'step', 'state', 'action', 'reward',
                           'current_q', 'max_next_q', 'target_q', 
                           'td_error', 'new_q', 'done'])
            
            for log in self.q_update_log:
                writer.writerow([
                    log['episode'],
                    log['step'],
                    str(log['state']),
                    log['action'],
                    log['reward'],
                    log['current_q'],
                    log['max_next_q'],
                    log['target_q'],
                    log['td_error'],
                    log['new_q'],
                    log['done']
                ])
        
        if self.verbose:
            print(f"✓ Q-update log saved to {filepath}")


def compare_epsilon_decay(env, decay_types: List[str], 
                         episodes: int = 2000, verbose: bool = False) -> Dict:
    """
    Compare different epsilon decay strategies.
    
    Args:
        env: Environment instance
        decay_types: List of decay types to test
        episodes: Number of training episodes
        verbose: Print progress
        
    Returns:
        Dictionary with results for each decay type
    """
    results = {}
    
    print("=" * 60)
    print("Comparing Epsilon Decay Strategies")
    print("=" * 60)
    
    for decay_type in decay_types:
        print(f"\n🎯 Testing {decay_type} decay")
        print("-" * 60)
        
        agent = QLearning(env, episodes=episodes, 
                         epsilon_decay_type=decay_type,
                         verbose=verbose, log_every=200)
        train_stats = agent.train()
        eval_stats = agent.evaluate(num_episodes=50)
        
        results[decay_type] = {
            'train': train_stats,
            'eval': eval_stats,
            'agent': agent
        }
        
        print(f"   Training time: {train_stats['training_time']:.2f}s")
        print(f"   Success rate: {eval_stats['success_rate']:.2%}")
        print(f"   Mean reward: {eval_stats['mean_reward']:.2f}")
    
    return results
