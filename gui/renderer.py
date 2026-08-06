"""
Maze Renderer for GUI
Handles visual representation of maze, agent, and overlays
"""

import numpy as np
from typing import Tuple, Optional, Dict
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.generator import MazeGenerator


class MazeRenderer:
    """
    Renders maze environment for Tkinter GUI.
    """
    
    # Color scheme
    COLORS = {
        'wall': '#2C3E50',           # Dark blue-gray
        'path': '#ECF0F1',           # Light gray
        'start': '#3498DB',          # Blue
        'key': '#F1C40F',            # Yellow
        'door_closed': '#8B4513',    # Brown
        'door_open': '#95A5A6',      # Gray
        'goal': '#2ECC71',           # Green
        'agent': '#E74C3C',          # Red
        'penalty': '#E67E22',        # Orange
        'charging': '#9B59B6',       # Purple
        'visited': '#BDC3C7',        # Light gray
        'grid': '#95A5A6',           # Gray
        'text': '#2C3E50',           # Dark
        'highlight': '#F39C12',      # Orange
    }
    
    def __init__(self, maze: np.ndarray, metadata: Dict, cell_size: int = 30):
        """
        Initialize renderer.
        
        Args:
            maze: Maze array
            metadata: Maze metadata
            cell_size: Size of each cell in pixels
        """
        self.maze = maze
        self.metadata = metadata
        self.cell_size = cell_size
        
        self.height, self.width = maze.shape
        self.canvas_width = self.width * cell_size
        self.canvas_height = self.height * cell_size
        
        # State tracking
        self.agent_pos = None
        self.has_key = False
        self.visited_cells = set()
        self.value_map = None
        self.policy_map = None
        
    def get_canvas_size(self) -> Tuple[int, int]:
        """Get canvas dimensions."""
        return (self.canvas_width, self.canvas_height)
    
    def cell_to_canvas(self, row: int, col: int) -> Tuple[int, int, int, int]:
        """
        Convert maze coordinates to canvas coordinates.
        
        Returns:
            (x1, y1, x2, y2) for rectangle
        """
        x1 = col * self.cell_size
        y1 = row * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        return (x1, y1, x2, y2)
    
    def get_cell_center(self, row: int, col: int) -> Tuple[int, int]:
        """Get center point of cell."""
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        return (x, y)
    
    def draw_maze(self, canvas, show_visited: bool = False, 
                  show_values: bool = False, show_policy: bool = False):
        """
        Draw the complete maze on canvas.
        
        Args:
            canvas: Tkinter canvas object
            show_visited: Show visited cells
            show_values: Show value heatmap
            show_policy: Show policy arrows
        """
        # Clear canvas
        canvas.delete('all')
        
        # Draw cells
        for row in range(self.height):
            for col in range(self.width):
                self._draw_cell(canvas, row, col, show_visited, show_values)
        
        # Draw special elements
        self._draw_special_elements(canvas)
        
        # Draw policy arrows if enabled
        if show_policy and self.policy_map is not None:
            self._draw_policy_arrows(canvas)
        
        # Draw grid lines
        self._draw_grid(canvas)
    
    def _draw_cell(self, canvas, row: int, col: int, 
                   show_visited: bool, show_values: bool):
        """Draw a single cell."""
        x1, y1, x2, y2 = self.cell_to_canvas(row, col)
        cell_type = self.maze[row, col]
        
        # Determine cell color
        if cell_type == MazeGenerator.WALL:
            color = self.COLORS['wall']
        elif show_visited and (row, col) in self.visited_cells:
            color = self.COLORS['visited']
        elif show_values and self.value_map is not None:
            # Value heatmap
            color = self._get_value_color(row, col)
        else:
            color = self.COLORS['path']
        
        # Draw cell
        canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='')
        
        # Draw value text if enabled
        if show_values and self.value_map is not None:
            if cell_type != MazeGenerator.WALL:
                value = self.value_map.get((row, col), 0.0)
                cx, cy = self.get_cell_center(row, col)
                canvas.create_text(cx, cy, text=f'{value:.1f}',
                                 font=('Arial', 8), fill=self.COLORS['text'])
    
    def _get_value_color(self, row: int, col: int) -> str:
        """Get color for value heatmap."""
        if self.value_map is None:
            return self.COLORS['path']
        
        value = self.value_map.get((row, col), 0.0)
        
        # Normalize value to [0, 1]
        all_values = list(self.value_map.values())
        if not all_values:
            return self.COLORS['path']
        
        min_val, max_val = min(all_values), max(all_values)
        if max_val == min_val:
            normalized = 0.5
        else:
            normalized = (value - min_val) / (max_val - min_val)
        
        # Color gradient: blue (low) -> green (mid) -> red (high)
        if normalized < 0.5:
            # Blue to green
            r = int(50 + normalized * 2 * 100)
            g = int(150 + normalized * 2 * 100)
            b = int(255 - normalized * 2 * 155)
        else:
            # Green to red
            r = int(150 + (normalized - 0.5) * 2 * 105)
            g = int(250 - (normalized - 0.5) * 2 * 150)
            b = int(100 - (normalized - 0.5) * 2 * 100)
        
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def _draw_special_elements(self, canvas):
        """Draw key, door, goal, penalties, charging stations."""
        # Start position
        if 'start' in self.metadata:
            row, col = self.metadata['start']
            self._draw_marker(canvas, row, col, 'S', self.COLORS['start'])
        
        # Key (if not collected)
        if not self.has_key and 'key' in self.metadata:
            row, col = self.metadata['key']
            self._draw_marker(canvas, row, col, 'K', self.COLORS['key'])
        
        # Door
        if 'door' in self.metadata:
            row, col = self.metadata['door']
            door_color = self.COLORS['door_open'] if self.has_key else self.COLORS['door_closed']
            self._draw_marker(canvas, row, col, 'D', door_color)
        
        # Goal
        if 'goal' in self.metadata:
            row, col = self.metadata['goal']
            self._draw_marker(canvas, row, col, 'G', self.COLORS['goal'])
        
        # Penalties
        if 'penalties' in self.metadata:
            for row, col in self.metadata['penalties']:
                self._draw_marker(canvas, row, col, 'P', self.COLORS['penalty'], size=8)
        
        # Charging stations
        if 'charging_stations' in self.metadata:
            for row, col in self.metadata['charging_stations']:
                self._draw_marker(canvas, row, col, 'C', self.COLORS['charging'], size=8)
    
    def _draw_marker(self, canvas, row: int, col: int, 
                    text: str, color: str, size: int = 12):
        """Draw a marker (text) on a cell."""
        if self.maze[row, col] == MazeGenerator.WALL:
            return
        
        cx, cy = self.get_cell_center(row, col)
        
        # Draw background circle
        r = self.cell_size // 3
        canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                          fill=color, outline='')
        
        # Draw text
        canvas.create_text(cx, cy, text=text,
                          font=('Arial', size, 'bold'),
                          fill='white')
    
    def _draw_policy_arrows(self, canvas):
        """Draw policy arrows."""
        if self.policy_map is None:
            return
        
        arrow_length = self.cell_size // 3
        
        for (row, col), action in self.policy_map.items():
            if self.maze[row, col] == MazeGenerator.WALL:
                continue
            
            cx, cy = self.get_cell_center(row, col)
            
            # Action directions: 0=Up, 1=Down, 2=Left, 3=Right
            dx, dy = 0, 0
            if action == 0:  # Up
                dy = -arrow_length
            elif action == 1:  # Down
                dy = arrow_length
            elif action == 2:  # Left
                dx = -arrow_length
            elif action == 3:  # Right
                dx = arrow_length
            
            # Draw arrow
            canvas.create_line(cx, cy, cx + dx, cy + dy,
                             arrow='last', fill=self.COLORS['highlight'],
                             width=2)
    
    def _draw_grid(self, canvas):
        """Draw grid lines."""
        # Vertical lines
        for col in range(self.width + 1):
            x = col * self.cell_size
            canvas.create_line(x, 0, x, self.canvas_height,
                             fill=self.COLORS['grid'], width=1)
        
        # Horizontal lines
        for row in range(self.height + 1):
            y = row * self.cell_size
            canvas.create_line(0, y, self.canvas_width, y,
                             fill=self.COLORS['grid'], width=1)
    
    def draw_agent(self, canvas, row: int, col: int, energy: Optional[int] = None):
        """
        Draw the agent at specified position.
        
        Args:
            canvas: Tkinter canvas
            row, col: Agent position
            energy: Current energy level (optional)
        """
        self.agent_pos = (row, col)
        
        cx, cy = self.get_cell_center(row, col)
        r = self.cell_size // 3
        
        # Draw agent as circle
        canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                          fill=self.COLORS['agent'], outline='white', width=2,
                          tags='agent')
        
        # Draw 'A' text
        canvas.create_text(cx, cy, text='A',
                          font=('Arial', 12, 'bold'),
                          fill='white', tags='agent')
        
        # Draw energy bar if provided
        if energy is not None:
            self._draw_energy_bar(canvas, row, col, energy)
    
    def _draw_energy_bar(self, canvas, row: int, col: int, energy: int):
        """Draw energy bar below agent."""
        x1, y1, x2, y2 = self.cell_to_canvas(row, col)
        
        bar_width = self.cell_size - 4
        bar_height = 4
        bar_x = x1 + 2
        bar_y = y2 - 6
        
        # Background
        canvas.create_rectangle(bar_x, bar_y, bar_x + bar_width, bar_y + bar_height,
                              fill='gray', outline='', tags='agent')
        
        # Energy level
        max_energy = 100  # Assume max
        energy_width = int((energy / max_energy) * bar_width)
        energy_color = '#2ECC71' if energy > 50 else '#E74C3C'
        
        canvas.create_rectangle(bar_x, bar_y, bar_x + energy_width, bar_y + bar_height,
                              fill=energy_color, outline='', tags='agent')
    
    def update_agent(self, canvas, row: int, col: int, energy: Optional[int] = None):
        """Update agent position."""
        # Remove old agent
        canvas.delete('agent')
        
        # Add to visited cells
        self.visited_cells.add((row, col))
        
        # Draw new agent
        self.draw_agent(canvas, row, col, energy)
    
    def set_key_collected(self, collected: bool = True):
        """Update key collection state."""
        self.has_key = collected
    
    def set_value_map(self, value_map: Dict[Tuple[int, int], float]):
        """Set value map for heatmap visualization."""
        self.value_map = value_map
    
    def set_policy_map(self, policy_map: Dict[Tuple[int, int], int]):
        """Set policy map for arrow visualization."""
        self.policy_map = policy_map
    
    def reset_visited(self):
        """Reset visited cells."""
        self.visited_cells.clear()
    
    def reset(self):
        """Reset all state."""
        self.agent_pos = None
        self.has_key = False
        self.visited_cells.clear()
        self.value_map = None
        self.policy_map = None
