import math
import numpy as np
from collections import deque
from typing import Tuple, List, Dict, Set
import json
from pathlib import Path


class MazeGenerator:
    """
    Generates dynamic maze environments with validation.
    
    Cell types:
    - 0: Empty (walkable)
    - 1: Wall (obstacle)
    - 2: Start position
    - 3: Key
    - 4: Door (closed)
    - 5: Goal
    - 6: Penalty cell
    - 7: Energy recharge station (for limited energy feature)
    """
    
    # Cell type constants
    EMPTY = 0
    WALL = 1
    START = 2
    KEY = 3
    DOOR = 4
    GOAL = 5
    PENALTY = 6
    ENERGY_STATION = 7
    
    def __init__(self, size: int = 15, seed: int = 8, 
                 min_wall_percentage: float = 0.15,
                 min_penalty_cells: int = 5):
        """
        Initialize maze generator.
        
        Args:
            size: Maze size (size x size)
            seed: Random seed for reproducibility
            min_wall_percentage: Minimum percentage of walls
            min_penalty_cells: Minimum number of penalty cells
        """
        self.size = size
        self.seed = seed
        self.min_wall_percentage = min_wall_percentage
        self.min_penalty_cells = min_penalty_cells
        self.rng = np.random.RandomState(seed)
        
    def generate(self, max_attempts: int = 100) -> Tuple[np.ndarray, Dict]:
        """
        Generate a valid maze.
        
        Returns:
            maze: 2D numpy array representing the maze
            metadata: Dictionary with maze information
        """
        for attempt in range(max_attempts):
            maze = self._generate_maze_structure()
            
            # Add special cells
            start_pos = self._place_special_cell(maze, self.START)
            key_pos = self._place_special_cell(maze, self.KEY)
            door_pos = self._place_special_cell(maze, self.DOOR)
            goal_pos = self._place_special_cell(maze, self.GOAL)
            
            # Add penalty cells
            penalty_positions = self._place_penalty_cells(maze)
            
            # Add energy recharge stations (for limited energy feature)
            energy_stations = self._place_energy_stations(maze, num_stations=2)
            
            # Validate maze
            if self._validate_maze(maze, start_pos, key_pos, door_pos, goal_pos):
                metadata = {
                    'size': self.size,
                    'seed': self.seed,
                    'start': start_pos,
                    'key': key_pos,
                    'door': door_pos,
                    'goal': goal_pos,
                    'penalty_cells': penalty_positions,
                    'energy_stations': energy_stations,
                    'wall_count': np.sum(maze == self.WALL),
                    'wall_percentage': np.sum(maze == self.WALL) / (self.size ** 2),
                    'attempt': attempt + 1
                }
                return maze, metadata
        
        raise RuntimeError(f"Failed to generate valid maze after {max_attempts} attempts")
    
    def _generate_maze_structure(self) -> np.ndarray:
        """Generate basic maze structure with walls distributed across the full grid."""
        maze = np.zeros((self.size, self.size), dtype=np.int32)
        
        # Calculate required number of walls across the full grid
        total_cells = self.size * self.size
        required_total_walls = math.ceil(self.min_wall_percentage * total_cells)
        
        # All cells are available for wall placement
        available_positions = [
            (i, j) for i in range(self.size)
            for j in range(self.size)
        ]
        
        num_walls_to_place = min(required_total_walls, len(available_positions))
        
        if num_walls_to_place > 0:
            wall_positions = self.rng.choice(
                len(available_positions),
                size=num_walls_to_place,
                replace=False
            )
        else:
            wall_positions = []
        
        for idx in wall_positions:
            i, j = available_positions[idx]
            maze[i, j] = self.WALL
        
        return maze
    
    def _place_special_cell(self, maze: np.ndarray, cell_type: int) -> Tuple[int, int]:
        """Place a special cell (start, key, door, goal) in the maze."""
        empty_cells = self._get_empty_cells(maze)
        
        if not empty_cells:
            raise RuntimeError("No empty cells available for placement")
        
        # Choose random position from empty cells
        idx = self.rng.choice(len(empty_cells))
        pos = empty_cells[idx]
        maze[pos[0], pos[1]] = cell_type
        
        return pos
    
    def _place_penalty_cells(self, maze: np.ndarray) -> List[Tuple[int, int]]:
        """Place penalty cells in the maze."""
        empty_cells = self._get_empty_cells(maze)
        
        num_penalties = max(
            self.min_penalty_cells,
            int(len(empty_cells) * 0.05)  # At least 5% of empty cells
        )
        
        if len(empty_cells) < num_penalties:
            num_penalties = len(empty_cells)
        
        penalty_indices = self.rng.choice(
            len(empty_cells),
            size=num_penalties,
            replace=False
        )
        
        penalty_positions = []
        for idx in penalty_indices:
            pos = empty_cells[idx]
            maze[pos[0], pos[1]] = self.PENALTY
            penalty_positions.append(pos)
        
        return penalty_positions
    
    def _place_energy_stations(self, maze: np.ndarray, num_stations: int = 2) -> List[Tuple[int, int]]:
        """Place energy recharge stations in the maze."""
        empty_cells = self._get_empty_cells(maze)
        
        if len(empty_cells) < num_stations:
            num_stations = len(empty_cells)
        
        if num_stations == 0:
            return []
        
        station_indices = self.rng.choice(
            len(empty_cells),
            size=num_stations,
            replace=False
        )
        
        station_positions = []
        for idx in station_indices:
            pos = empty_cells[idx]
            maze[pos[0], pos[1]] = self.ENERGY_STATION
            station_positions.append(pos)
        
        return station_positions
    
    def _get_empty_cells(self, maze: np.ndarray) -> List[Tuple[int, int]]:
        """Get list of empty cells in the maze."""
        empty_cells = []
        for i in range(self.size):
            for j in range(self.size):
                if maze[i, j] == self.EMPTY:
                    empty_cells.append((i, j))
        return empty_cells
    
    def _validate_maze(self, maze: np.ndarray, start: Tuple[int, int],
                      key: Tuple[int, int], door: Tuple[int, int],
                      goal: Tuple[int, int]) -> bool:
        """
        Validate maze using BFS to ensure valid paths exist.
        
        Must have:
        1. Path from start to key
        2. Path from key to door
        3. Path from door to goal
        """
        # Check path from start to key
        if not self._bfs_path_exists(maze, start, key):
            return False
        
        # Check path from key to door
        if not self._bfs_path_exists(maze, key, door):
            return False
        
        # Check path from door to goal
        if not self._bfs_path_exists(maze, door, goal):
            return False
        
        return True


    # Start ──BFS──▶ Key ──BFS──▶ Door ──BFS──▶ Goal
    def _bfs_path_exists(self, maze: np.ndarray, 
                        start: Tuple[int, int], 
                        end: Tuple[int, int]) -> bool:
        """
        Check if path exists between start and end using BFS.
        
        Args:
            maze: Maze array
            start: Start position (row, col)
            end: End position (row, col)
            
        Returns:
            True if path exists, False otherwise
        """
        if start == end:
            return True
        
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        # Four directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            row, col = queue.popleft()
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check bounds
                if (0 <= new_row < self.size and 
                    0 <= new_col < self.size and
                    (new_row, new_col) not in visited):
                    
                    # Check if we reached the end
                    if (new_row, new_col) == end:
                        return True
                    
                    # Check if cell is walkable (not a wall)
                    if maze[new_row, new_col] != self.WALL:
                        visited.add((new_row, new_col))
                        queue.append((new_row, new_col))
        
        return False
    
    def save_maze(self, maze: np.ndarray, metadata: Dict, 
                  filepath: str = None) -> str:
        """
        Save maze to file.
        
        Args:
            maze: Maze array
            metadata: Maze metadata
            filepath: Path to save file (optional)
            
        Returns:
            Path where maze was saved
        """
        if filepath is None:
            maps_dir = Path("environments/maps")
            maps_dir.mkdir(parents=True, exist_ok=True)
            filepath = maps_dir / f"maze_seed{self.seed}_size{self.size}.npz"
        
        # Convert tuples to lists for JSON serialization
        metadata_copy = metadata.copy()
        for key in ['start', 'key', 'door', 'goal']:
            if key in metadata_copy:
                metadata_copy[key] = list(metadata_copy[key])
        
        if 'penalty_cells' in metadata_copy:
            metadata_copy['penalty_cells'] = [list(p) for p in metadata_copy['penalty_cells']]
        
        if 'energy_stations' in metadata_copy:
            metadata_copy['energy_stations'] = [list(s) for s in metadata_copy['energy_stations']]
        
        # Convert numpy types to Python types
        for key in metadata_copy:
            if isinstance(metadata_copy[key], (np.integer, np.floating)):
                metadata_copy[key] = metadata_copy[key].item()
            elif isinstance(metadata_copy[key], np.ndarray):
                metadata_copy[key] = metadata_copy[key].tolist()
        
        # Save as numpy archive
        np.savez_compressed(
            filepath,
            maze=maze,
            metadata=json.dumps(metadata_copy)
        )
        
        return str(filepath)
    
    @staticmethod
    def load_maze(filepath: str) -> Tuple[np.ndarray, Dict]:
        """
        Load maze from file.
        
        Args:
            filepath: Path to maze file
            
        Returns:
            maze: Maze array
            metadata: Maze metadata
        """
        data = np.load(filepath, allow_pickle=True)
        maze = data['maze']
        metadata = json.loads(str(data['metadata']))
        
        # Convert lists back to tuples
        for key in ['start', 'key', 'door', 'goal']:
            if key in metadata:
                metadata[key] = tuple(metadata[key])
        
        if 'penalty_cells' in metadata:
            metadata['penalty_cells'] = [tuple(p) for p in metadata['penalty_cells']]
        
        if 'energy_stations' in metadata:
            metadata['energy_stations'] = [tuple(s) for s in metadata['energy_stations']]
        
        return maze, metadata
    
    def visualize_maze(self, maze: np.ndarray) -> str:
        """
        Create ASCII visualization of maze.
        
        Returns:
            String representation of maze
        """
        symbols = {
            self.EMPTY: '  ',
            self.WALL: '██',
            self.START: 'S ',
            self.KEY: 'K ',
            self.DOOR: 'D ',
            self.GOAL: 'G ',
            self.PENALTY: 'X ',
            self.ENERGY_STATION: 'E ',
        }
        
        lines = []
        for row in maze:
            line = ''.join(symbols.get(cell, '? ') for cell in row)
            lines.append(line)
        
        return '\n'.join(lines)


def generate_target_mazes(source_maze: np.ndarray, 
                         source_metadata: Dict,
                         seed: int,
                         similar_change_pct: float = 0.15,
                         different_change_pct: float = 0.35) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """
    Generate target mazes for transfer learning.
    
    Args:
        source_maze: Original maze
        source_metadata: Original metadata
        seed: Random seed
        similar_change_pct: Percentage of obstacles to change for similar maze
        different_change_pct: Percentage of obstacles to change for different maze
        
    Returns:
        similar_maze, different_maze, similar_metadata, different_metadata
    """
    rng = np.random.RandomState(seed + 1000)
    size = source_maze.shape[0]
    validator = MazeGenerator(size=size)
    
    # Generate similar target maze (15-20% change)
    for _ in range(100):
        similar_maze = source_maze.copy()
        similar_metadata = source_metadata.copy()
        _modify_maze(similar_maze, similar_metadata, rng, similar_change_pct, keep_special=True)
        if validator._validate_maze(similar_maze, similar_metadata['start'], 
                                   similar_metadata['key'], similar_metadata['door'], 
                                   similar_metadata['goal']):
            break
    else:
        raise RuntimeError("Failed to generate valid similar maze")
    
    # Generate different target maze (35%+ change)
    rng_diff = np.random.RandomState(seed + 2000)
    for _ in range(100):
        different_maze = source_maze.copy()
        different_metadata = source_metadata.copy()
        _modify_maze(different_maze, different_metadata, rng_diff, different_change_pct, keep_special=False)
        if validator._validate_maze(different_maze, different_metadata['start'], 
                                   different_metadata['key'], different_metadata['door'], 
                                   different_metadata['goal']):
            break
    else:
        raise RuntimeError("Failed to generate valid different maze")
    
    return similar_maze, different_maze, similar_metadata, different_metadata


def _modify_maze(maze: np.ndarray, metadata: Dict, rng: np.random.RandomState,
                change_pct: float, keep_special: bool = True):
    """Helper function to modify maze for transfer learning."""
    size = maze.shape[0]
    
    # Get modifiable cells (all cells in the grid)
    modifiable = []
    for i in range(size):
        for j in range(size):
            if keep_special:
                # Don't modify start, key, door, goal
                if maze[i, j] not in [MazeGenerator.START, MazeGenerator.KEY, 
                                     MazeGenerator.DOOR, MazeGenerator.GOAL]:
                    modifiable.append((i, j))
            else:
                modifiable.append((i, j))
    
    # Calculate number of cells to modify
    num_to_modify = int(len(modifiable) * change_pct)
    
    # Select cells to modify
    indices = rng.choice(len(modifiable), size=num_to_modify, replace=False)
    
    for idx in indices:
        i, j = modifiable[idx]
        
        # Toggle wall/empty with probability
        if maze[i, j] == MazeGenerator.WALL:
            maze[i, j] = MazeGenerator.EMPTY
        elif maze[i, j] == MazeGenerator.EMPTY:
            maze[i, j] = MazeGenerator.WALL
        elif maze[i, j] == MazeGenerator.PENALTY:
            # Change penalty to empty or wall
            maze[i, j] = MazeGenerator.WALL if rng.rand() > 0.5 else MazeGenerator.EMPTY
    
    # If not keeping special cells, relocate key or goal
    if not keep_special:
        # Move key to a new position
        empty_cells = [(i, j) for i in range(size) for j in range(size) 
                      if maze[i, j] == MazeGenerator.EMPTY]
        if empty_cells:
            new_key_pos = empty_cells[rng.choice(len(empty_cells))]
            # Clear old key
            old_key = metadata['key']
            maze[old_key[0], old_key[1]] = MazeGenerator.EMPTY
            # Place new key
            maze[new_key_pos[0], new_key_pos[1]] = MazeGenerator.KEY
            metadata['key'] = new_key_pos
