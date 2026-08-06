import numpy as np
from typing import Tuple, Dict, List, Optional
from enum import IntEnum
import logging


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class MazeEnvironment:
    """
    Dynamic Maze Environment with stochastic transitions.
    
    State representation: (x, y, has_key, energy)
    - x, y: agent position
    - has_key: 0 or 1 (whether agent has collected the key)
    - energy: remaining energy (for limited energy feature)
    
    This ensures Markov property is maintained.
    """
    
    def __init__(self, maze: np.ndarray, metadata: Dict,
                 reward_type: str = 'sparse',
                 rewards: Dict = None,
                 max_energy: int = 225,
                 intended_prob: float = 0.8,
                 perpendicular_prob: float = 0.1,
                 max_steps: int = 675,
                 seed: int = None):
        """
        Initialize maze environment.
        
        Args:
            maze: 2D maze array
            metadata: Maze metadata with special positions
            reward_type: 'sparse' or 'shaped'
            rewards: Custom reward dictionary
            max_energy: Maximum energy for limited energy feature
            intended_prob: Probability of intended action
            perpendicular_prob: Probability of perpendicular action
            max_steps: Maximum steps per episode
            seed: Random seed
        """
        self.maze = maze.copy()
        self.metadata = metadata
        self.size = maze.shape[0]
        self.reward_type = reward_type
        self.max_energy = max_energy
        self.intended_prob = intended_prob
        self.perpendicular_prob = perpendicular_prob
        self.max_steps = max_steps
        
        # Set random seed
        self.rng = np.random.RandomState(seed)
        
        # Special positions
        self.start_pos = metadata['start']
        self.key_pos = metadata['key']
        self.door_pos = metadata['door']
        self.goal_pos = metadata['goal']
        self.penalty_positions = set(metadata.get('penalty_cells', []))
        self.energy_stations = set(metadata.get('energy_stations', []))
        
        # Reward structure
        if rewards is None:
            from experiments.configs.config import REWARDS
            self.rewards = REWARDS[reward_type]
        else:
            self.rewards = rewards
        
        # State variables
        self.agent_pos = None
        self.has_key = False
        self.energy = max_energy
        self.steps = 0
        self.done = False
        self.door_opened = False
        
        # Episode statistics
        self.total_reward = 0.0
        self.wall_collisions = 0
        self.penalty_visits = 0
        
        # Event log
        self.event_log = []
        
        # Action mapping
        self.action_effects = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1),
        }
        
        # Perpendicular actions
        self.perpendicular_actions = {
            Action.UP: [Action.LEFT, Action.RIGHT],
            Action.DOWN: [Action.LEFT, Action.RIGHT],
            Action.LEFT: [Action.UP, Action.DOWN],
            Action.RIGHT: [Action.UP, Action.DOWN],
        }
        
    def reset(self) -> Tuple[int, int, int, int]:
        """
        Reset environment to initial state.
        
        Returns:
            Initial state: (x, y, has_key, energy)
        """
        self.agent_pos = list(self.start_pos)
        self.has_key = False
        self.energy = self.max_energy
        self.steps = 0
        self.done = False
        self.door_opened = False
        self.total_reward = 0.0
        self.wall_collisions = 0
        self.penalty_visits = 0
        self.event_log = []
        
        self._log_event('episode_start', self.agent_pos, 0.0)
        
        return self.get_state()
    
    def get_state(self) -> Tuple[int, int, int, int]:
        """
        Get current state.
        
        Returns:
            State tuple: (x, y, has_key, energy)
        """
        return (self.agent_pos[0], self.agent_pos[1], 
                int(self.has_key), self.energy)
    
    def step(self, action: int) -> Tuple[Tuple[int, int, int, int], float, bool, Dict]:
        """
        Execute action with stochastic transitions.
        
        Args:
            action: Action to execute (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            
        Returns:
            next_state: Next state tuple
            reward: Reward received
            done: Whether episode is finished
            info: Additional information dictionary
        """
        if self.done:
            return self.get_state(), 0.0, True, {'error': 'Episode already done'}
        
        # Sample actual action based on stochastic transitions
        actual_action = self._sample_action(action)
        
        # Execute action
        reward, info = self._execute_action(actual_action, intended_action=action)
        
        # Update step counter
        self.steps += 1
        self.total_reward += reward
        
        # Check termination conditions
        if self._check_termination():
            self.done = True
            self._log_event('episode_end', self.agent_pos, reward)
        
        info['steps'] = self.steps
        info['total_reward'] = self.total_reward
        info['intended_action'] = action
        info['actual_action'] = actual_action
        
        return self.get_state(), reward, self.done, info
    
    def _sample_action(self, intended_action: int) -> int:
        """
        Sample actual action based on stochastic transitions.
        
        With probability 0.8: execute intended action
        With probability 0.1 each: execute perpendicular actions
        """
        rand = self.rng.random()
        
        if rand < self.intended_prob:
            # Execute intended action
            return intended_action
        elif rand < self.intended_prob + self.perpendicular_prob:
            # Execute first perpendicular action
            return self.perpendicular_actions[intended_action][0]
        else:
            # Execute second perpendicular action
            return self.perpendicular_actions[intended_action][1]
    
    def _execute_action(self, action: int, intended_action: int) -> Tuple[float, Dict]:
        """
        Execute action and return reward and info.
        
        Returns:
            reward: Immediate reward
            info: Information dictionary
        """
        info = {}
        reward = self.rewards.get('step', -0.1)
        
        # Calculate new position
        dx, dy = self.action_effects[action]
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy
        new_pos = (new_x, new_y)
        
        # Check if new position is valid
        if self._is_valid_position(new_pos):
            # Move agent
            old_pos = tuple(self.agent_pos)
            self.agent_pos = [new_x, new_y]
            
            # Consume energy
            self.energy -= 1
            
            # Check for special events
            cell_type = self.maze[new_x, new_y]
            
            # Check if reached key
            if new_pos == self.key_pos and not self.has_key:
                self.has_key = True
                reward += self.rewards.get('key', 50.0)
                self._log_event('key_collected', new_pos, reward)
                info['event'] = 'key_collected'
            
            # Check if reached door
            elif new_pos == self.door_pos:
                if self.has_key:
                    self.door_opened = True
                    reward += self.rewards.get('step', -0.1)  # Just normal step
                    self._log_event('door_opened', new_pos, reward)
                    info['event'] = 'door_opened'
                else:
                    # Tried to open locked door
                    reward += self.rewards.get('locked_door', -2.0)
                    self._log_event('locked_door_attempt', new_pos, reward)
                    info['event'] = 'locked_door'
            
            # Check if reached goal
            elif new_pos == self.goal_pos:
                if self.door_opened:
                    reward += self.rewards.get('goal', 100.0)
                    self._log_event('goal_reached', new_pos, reward)
                    info['event'] = 'goal_reached'
                    self.done = True
                else:
                    # Reached goal but door not opened
                    reward += self.rewards.get('step', -0.1)
                    info['event'] = 'goal_without_door'
            
            # Check if stepped on penalty cell
            elif new_pos in self.penalty_positions:
                reward += self.rewards.get('penalty_cell', -5.0)
                self.penalty_visits += 1
                self._log_event('penalty_cell', new_pos, reward)
                info['event'] = 'penalty_cell'
            
            # Check if reached energy station
            elif new_pos in self.energy_stations:
                energy_gain = 20
                self.energy = min(self.energy + energy_gain, self.max_energy)
                reward += 0.5  # Small bonus for finding energy
                self._log_event('energy_recharge', new_pos, reward)
                info['event'] = 'energy_recharge'
            
            # Reward shaping (if enabled)
            if self.reward_type == 'shaped':
                reward += self._compute_shaped_reward(old_pos, new_pos)
            
            info['moved'] = True
            self._log_event('move', new_pos, reward)
            
        else:
            # Hit wall - stay in place
            reward += self.rewards.get('wall_collision', -1.0)
            self.wall_collisions += 1
            self._log_event('wall_collision', tuple(self.agent_pos), reward)
            info['moved'] = False
            info['event'] = 'wall_collision'
        
        return reward, info
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (not out of bounds or wall)."""
        x, y = pos
        
        # Check bounds
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        
        # Check if wall
        from environments.generator import MazeGenerator
        if self.maze[x, y] == MazeGenerator.WALL:
            return False
        
        return True
    
    def _compute_shaped_reward(self, old_pos: Tuple[int, int], 
                               new_pos: Tuple[int, int]) -> float:
        """
        Compute reward shaping bonus.
        
        Rewards agent for:
        - Moving closer to key (if don't have key)
        - Moving closer to goal (if have key and door opened)
        - Safe navigation (avoiding penalty cells)
        """
        shaped_reward = 0.0
        
        # Distance-based shaping
        if not self.has_key:
            # Moving towards key
            old_dist = self._manhattan_distance(old_pos, self.key_pos)
            new_dist = self._manhattan_distance(new_pos, self.key_pos)
            if new_dist < old_dist:
                shaped_reward += self.rewards.get('closer_to_key', 0.5)
        
        elif self.door_opened:
            # Moving towards goal
            old_dist = self._manhattan_distance(old_pos, self.goal_pos)
            new_dist = self._manhattan_distance(new_pos, self.goal_pos)
            if new_dist < old_dist:
                shaped_reward += self.rewards.get('closer_to_goal', 0.5)
        
        # Safe navigation bonus
        if new_pos not in self.penalty_positions:
            # Check if avoiding nearby penalty cells
            nearby_penalties = sum(1 for p in self.penalty_positions 
                                  if self._manhattan_distance(new_pos, p) <= 1)
            if nearby_penalties > 0:
                shaped_reward += self.rewards.get('safe_navigation', 0.2) * 0.5
        
        return shaped_reward
    
    def _manhattan_distance(self, pos1: Tuple[int, int], 
                           pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Goal reached
        if tuple(self.agent_pos) == self.goal_pos and self.door_opened:
            return True
        
        # Out of energy
        if self.energy <= 0:
            self._log_event('out_of_energy', tuple(self.agent_pos), 0.0)
            return True
        
        # Maximum steps exceeded
        if self.steps >= self.max_steps:
            self._log_event('max_steps_exceeded', tuple(self.agent_pos), 0.0)
            return True
        
        return False
    
    def _log_event(self, event_type: str, position: Tuple[int, int], reward: float):
        """Log an event for analysis."""
        self.event_log.append({
            'step': self.steps,
            'event': event_type,
            'position': position,
            'has_key': self.has_key,
            'energy': self.energy,
            'reward': reward
        })
    
    def get_transition_probabilities(self, state: Tuple[int, int, int, int], 
                                    action: int) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Get all possible next states and their probabilities for a given state-action pair.
        Used by model-based algorithms like Value Iteration.
        
        Returns:
            List of (next_state, probability) tuples
        """
        transitions = []
        
        # Get actual actions that could be executed
        actual_actions = [action] + self.perpendicular_actions[action]
        probabilities = [self.intended_prob, self.perpendicular_prob, self.perpendicular_prob]
        
        for actual_action, prob in zip(actual_actions, probabilities):
            next_state = self._simulate_action(state, actual_action)
            
            # Check if this transition already exists
            found = False
            for i, (s, p) in enumerate(transitions):
                if s == next_state:
                    transitions[i] = (s, p + prob)
                    found = True
                    break
            
            if not found:
                transitions.append((next_state, prob))
        
        return transitions
    
    def _simulate_action(self, state: Tuple[int, int, int, int], 
                        action: int) -> Tuple[int, int, int, int]:
        """
        Simulate an action from a state without changing environment state.
        Returns the resulting next state.
        """
        x, y, has_key, energy = state
        
        # Calculate new position
        dx, dy = self.action_effects[action]
        new_x, new_y = x + dx, y + dy
        new_pos = (new_x, new_y)
        
        # Check if move is valid
        if not self._is_valid_position(new_pos):
            # Hit wall, stay in place but lose energy
            return (x, y, has_key, max(0, energy - 1))
        
        # Move is valid
        new_has_key = has_key
        new_energy = max(0, energy - 1)
        
        # Check if collecting key
        if new_pos == self.key_pos and not has_key:
            new_has_key = 1
        
        # Check if at energy station
        if new_pos in self.energy_stations:
            new_energy = min(new_energy + 20, self.max_energy)
        
        return (new_x, new_y, new_has_key, new_energy)
    
    def get_reward(self, state: Tuple[int, int, int, int], 
                   action: int, 
                   next_state: Tuple[int, int, int, int]) -> float:
        """
        Get reward for a state-action-next_state transition.
        Used by model-based algorithms.
        """
        x, y, has_key, energy = state
        next_x, next_y, next_has_key, next_energy = next_state
        
        reward = self.rewards.get('step', -0.1)
        
        # Check if moved
        if (x, y) == (next_x, next_y):
            # Hit wall
            reward += self.rewards.get('wall_collision', -1.0)
            return reward
        
        next_pos = (next_x, next_y)
        
        # Check events
        if next_pos == self.key_pos and not has_key and next_has_key:
            reward += self.rewards.get('key', 50.0)
        
        if next_pos == self.door_pos and not has_key:
            reward += self.rewards.get('locked_door', -2.0)
        
        if next_pos == self.goal_pos and has_key:
            reward += self.rewards.get('goal', 100.0)
        
        if next_pos in self.penalty_positions:
            reward += self.rewards.get('penalty_cell', -5.0)
        
        return reward
    
    def is_terminal_state(self, state: Tuple[int, int, int, int]) -> bool:
        """Check if a state is terminal."""
        x, y, has_key, energy = state
        
        # Goal reached with key
        if (x, y) == self.goal_pos and has_key:
            return True
        
        # Out of energy
        if energy <= 0:
            return True
        
        return False
    
    def get_all_states(self) -> List[Tuple[int, int, int, int]]:
        """
        Get all possible states in the environment.
        Used by Value Iteration.
        """
        states = []
        
        for x in range(self.size):
            for y in range(self.size):
                # Skip walls
                if not self._is_valid_position((x, y)):
                    continue
                
                for has_key in [0, 1]:
                    for energy in range(0, self.max_energy + 1, 25):  # Sample energy levels
                        states.append((x, y, has_key, energy))
        
        return states
    
    def render(self) -> str:
        """Render current state as ASCII string."""
        from environments.generator import MazeGenerator
        
        symbols = {
            MazeGenerator.EMPTY: '  ',
            MazeGenerator.WALL: '██',
            MazeGenerator.START: 'S ',
            MazeGenerator.KEY: 'K ' if not self.has_key else '  ',
            MazeGenerator.DOOR: 'D ' if not self.door_opened else '□ ',
            MazeGenerator.GOAL: 'G ',
            MazeGenerator.PENALTY: 'X ',
            MazeGenerator.ENERGY_STATION: 'E ',
        }
        
        lines = []
        for i in range(self.size):
            line = ''
            for j in range(self.size):
                if [i, j] == self.agent_pos:
                    line += 'A '
                else:
                    line += symbols.get(self.maze[i, j], '? ')
            lines.append(line)
        
        # Add status
        status = f"\nSteps: {self.steps} | Energy: {self.energy}/{self.max_energy} | "
        status += f"Key: {'✓' if self.has_key else '✗'} | Door: {'Open' if self.door_opened else 'Closed'}"
        status += f"\nReward: {self.total_reward:.2f}"
        
        return '\n'.join(lines) + status
