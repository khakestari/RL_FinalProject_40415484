"""
Main GUI Application
Interactive visualization for RL agents in maze
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import os
import numpy as np
from pathlib import Path
import time
import threading
import pickle
from typing import Optional, Dict
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.generator import MazeGenerator, generate_target_mazes
from environments.maze import MazeEnvironment
from agents.value_iteration import ValueIteration
from agents.q_learning import QLearning
from agents.sarsa_lambda import SarsaLambda
from gui.renderer import MazeRenderer


class MazeGUI:
    """
    Main GUI application for maze environment.
    """
    
    # Map algorithm display names to model file patterns
    MODEL_PATTERNS = {
        "Value Iteration": ["value_iteration*.pkl"],
        "Q-Learning": ["q_learning*.pkl"],
        "SARSA(λ)": ["sarsa_lambda*.pkl"],
    }
    
    def __init__(self, root):
        """Initialize GUI."""
        self.root = root
        self.root.title("RL Maze Environment - Student ID: 40415484")
        self.root.geometry("1300x850")
        self.root.minsize(1100, 700)
        
        # State
        self.maze = None
        self.metadata = None
        self.env = None
        self.agent = None
        self.renderer = None
        
        self.current_state = None
        self.episode_reward = 0
        self.episode_steps = 0
        self.is_running = False
        self.is_training = False
        self.speed = 3.0  # Steps per second
        
        # Statistics
        self.total_episodes = 0
        self.successful_episodes = 0
        
        # Store mazes for environment switching
        self.source_maze = None
        self.source_metadata = None
        self.similar_maze = None
        self.similar_metadata = None
        self.different_maze = None
        self.different_metadata = None
        
        # Create UI
        self._create_ui()
        
        # Load default maze
        self._load_default_maze()
        
        # Try auto-load model for current algorithm
        self._auto_load_model()
    
    def _create_ui(self):
        """Create user interface."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel: Canvas
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas for maze with scrollbar support
        canvas_frame = ttk.Frame(left_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='white', width=450, height=450)
        self.canvas.pack(expand=True)  # Centers it in canvas_frame without stretching
        
        # Right panel: Controls and info (scrollable)
        right_container = ttk.Frame(main_frame, width=350)
        right_container.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_container.pack_propagate(False)
        
        self.right_canvas = tk.Canvas(right_container, highlightthickness=0)
        self.right_scrollbar = ttk.Scrollbar(right_container, orient="vertical", command=self.right_canvas.yview)
        
        right_frame = ttk.Frame(self.right_canvas)
        
        right_frame.bind(
            "<Configure>",
            lambda e: self.right_canvas.configure(
                scrollregion=self.right_canvas.bbox("all")
            )
        )
        
        self.right_canvas.create_window((0, 0), window=right_frame, anchor="nw", width=330)
        self.right_canvas.configure(yscrollcommand=self.right_scrollbar.set)
        
        self.right_canvas.pack(side="left", fill="both", expand=True)
        self.right_scrollbar.pack(side="right", fill="y")
        
        # Make mousewheel scroll the right canvas
        def _on_mousewheel(event):
            self.right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.right_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # --- Algorithm selection ---
        algo_frame = ttk.LabelFrame(right_frame, text="Algorithm", padding=8)
        algo_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.algo_var = tk.StringVar(value="Q-Learning")
        algorithms = ["Value Iteration", "Q-Learning", "SARSA(λ)"]
        for algo in algorithms:
            ttk.Radiobutton(algo_frame, text=algo, variable=self.algo_var,
                           value=algo, command=self._on_algorithm_change).pack(anchor=tk.W)
        
        # Model buttons row
        model_frame = ttk.Frame(algo_frame)
        model_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(model_frame, text="📂 Load Model", 
                  command=self._load_model).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(model_frame, text="⚡ Auto-Load", 
                  command=self._auto_load_model).pack(side=tk.LEFT, padx=(0, 3))
        
        self.model_label = ttk.Label(algo_frame, text="No model loaded", 
                                     foreground="gray", wraplength=300)
        self.model_label.pack(anchor=tk.W, pady=(3, 0))
        
        # --- Environment selection ---
        env_frame = ttk.LabelFrame(right_frame, text="Environment", padding=8)
        env_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.env_var = tk.StringVar(value="source")
        envs = [("Source (Original)", "source"), 
                ("Similar Target", "similar"), 
                ("Different Target", "different")]
        for text, val in envs:
            ttk.Radiobutton(env_frame, text=text, variable=self.env_var,
                           value=val, command=self._on_environment_change).pack(anchor=tk.W)
        
        # Reward type
        reward_frame = ttk.Frame(env_frame)
        reward_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(reward_frame, text="Reward:").pack(side=tk.LEFT)
        self.reward_var = tk.StringVar(value="sparse")
        ttk.Radiobutton(reward_frame, text="Sparse", variable=self.reward_var,
                       value="sparse", command=self._on_environment_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(reward_frame, text="Shaped", variable=self.reward_var,
                       value="shaped", command=self._on_environment_change).pack(side=tk.LEFT)
        
        # --- Control buttons ---
        control_frame = ttk.LabelFrame(right_frame, text="Controls", padding=8)
        control_frame.pack(fill=tk.X, pady=(0, 8))
        
        # Play/Pause/Reset row
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X)
        
        self.play_btn = ttk.Button(btn_frame, text="▶ Play", command=self._play)
        self.play_btn.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        self.pause_btn = ttk.Button(btn_frame, text="⏸ Pause", command=self._pause,
                                    state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        self.reset_btn = ttk.Button(btn_frame, text="🔄 Reset", command=self._reset)
        self.reset_btn.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        # Step and Train row
        btn_frame2 = ttk.Frame(control_frame)
        btn_frame2.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(btn_frame2, text="⏭ Single Step", 
                  command=self._single_step).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        self.train_btn = ttk.Button(btn_frame2, text="🏋️ Train", 
                                    command=self._start_training)
        self.train_btn.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        # Mode selection
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="evaluate")
        ttk.Radiobutton(mode_frame, text="Evaluate (Greedy)", variable=self.mode_var,
                       value="evaluate").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Train (ε-greedy)", variable=self.mode_var,
                       value="train").pack(side=tk.LEFT)
        
        # Speed control
        speed_frame = ttk.Frame(control_frame)
        speed_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        
        self.speed_var = tk.DoubleVar(value=3.0)
        speed_scale = ttk.Scale(speed_frame, from_=0.5, to=50.0,
                               variable=self.speed_var, orient=tk.HORIZONTAL,
                               command=self._on_speed_change)
        speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.speed_label = ttk.Label(speed_frame, text="3.0x")
        self.speed_label.pack(side=tk.LEFT)
        
        # --- Visualization options ---
        vis_frame = ttk.LabelFrame(right_frame, text="Visualization", padding=8)
        vis_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.show_visited_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis_frame, text="Show visited cells",
                       variable=self.show_visited_var,
                       command=self._redraw).pack(anchor=tk.W)
        
        self.show_values_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(vis_frame, text="Show value heatmap",
                       variable=self.show_values_var,
                       command=self._redraw).pack(anchor=tk.W)
        
        self.show_policy_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(vis_frame, text="Show policy arrows",
                       variable=self.show_policy_var,
                       command=self._redraw).pack(anchor=tk.W)
        
        # --- Episode info ---
        info_frame = ttk.LabelFrame(right_frame, text="Episode Info", padding=8)
        info_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.episode_label = ttk.Label(info_frame, text="Episode: 0")
        self.episode_label.pack(anchor=tk.W)
        
        self.steps_label = ttk.Label(info_frame, text="Steps: 0")
        self.steps_label.pack(anchor=tk.W)
        
        self.reward_label = ttk.Label(info_frame, text="Reward: 0.0")
        self.reward_label.pack(anchor=tk.W)
        
        self.status_label = ttk.Label(info_frame, text="Status: Ready")
        self.status_label.pack(anchor=tk.W)
        
        self.key_status_label = ttk.Label(info_frame, text="Key: ✗ | Door: Closed")
        self.key_status_label.pack(anchor=tk.W)
        
        self.energy_label = ttk.Label(info_frame, text="Energy: --")
        self.energy_label.pack(anchor=tk.W)
        
        # --- Statistics ---
        stats_frame = ttk.LabelFrame(right_frame, text="Statistics", padding=8)
        stats_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.total_ep_label = ttk.Label(stats_frame, text="Total Episodes: 0")
        self.total_ep_label.pack(anchor=tk.W)
        
        self.success_label = ttk.Label(stats_frame, text="Success Rate: 0%")
        self.success_label.pack(anchor=tk.W)
        
        self.training_label = ttk.Label(stats_frame, text="Training: --")
        self.training_label.pack(anchor=tk.W)
    
    def _load_default_maze(self):
        """Load default maze and generate target mazes."""
        try:
            # Try to load saved maze
            self.maze, self.metadata = MazeGenerator.load_maze(
                str(project_root / "environments" / "maps" / "maze_seed8_size15.npz")
            )
        except Exception:
            # Generate new maze
            generator = MazeGenerator(size=15, seed=8)
            self.maze, self.metadata = generator.generate()
            generator.save_maze(self.maze, self.metadata)
        
        # Store as source
        self.source_maze = self.maze.copy()
        self.source_metadata = self.metadata.copy()
        
        # Generate target mazes
        try:
            similar, different, similar_meta, different_meta = generate_target_mazes(
                self.source_maze, self.source_metadata, seed=8
            )
            self.similar_maze = similar
            self.similar_metadata = similar_meta
            self.different_maze = different
            self.different_metadata = different_meta
        except Exception as e:
            print(f"Warning: Could not generate target mazes: {e}")
        
        # Create environment
        self._create_environment()
    
    def _create_environment(self):
        """Create environment from current maze and settings."""
        self.env = MazeEnvironment(
            self.maze, self.metadata, 
            reward_type=self.reward_var.get(), 
            seed=42
        )
        
        # Create renderer
        self.renderer = MazeRenderer(self.maze, self.metadata, cell_size=30)
        
        # Resize canvas
        canvas_width, canvas_height = self.renderer.get_canvas_size()
        self.canvas.config(width=canvas_width, height=canvas_height)
        
        # Draw initial maze
        self._redraw()
    
    def _on_environment_change(self):
        """Handle environment/reward type change."""
        env_type = self.env_var.get()
        
        if env_type == "source":
            self.maze = self.source_maze.copy()
            self.metadata = self.source_metadata.copy()
        elif env_type == "similar":
            if self.similar_maze is not None:
                self.maze = self.similar_maze.copy()
                self.metadata = self.similar_metadata.copy()
            else:
                messagebox.showwarning("Warning", "Similar target maze not available.")
                self.env_var.set("source")
                return
        elif env_type == "different":
            if self.different_maze is not None:
                self.maze = self.different_maze.copy()
                self.metadata = self.different_metadata.copy()
            else:
                messagebox.showwarning("Warning", "Different target maze not available.")
                self.env_var.set("source")
                return
        
        # Reset state
        self._pause()
        self.current_state = None
        self.episode_reward = 0
        self.episode_steps = 0
        self.total_episodes = 0
        self.successful_episodes = 0
        
        # Recreate environment and renderer
        self._create_environment()
        self._update_info()
        self._update_stats()
    
    def _find_model_files(self, algo: str):
        """Find all model files matching the algorithm."""
        models_dir = project_root / "results" / "models"
        if not models_dir.exists():
            return []
        
        import glob
        files = []
        patterns = self.MODEL_PATTERNS.get(algo, [])
        for pattern in patterns:
            files.extend(models_dir.glob(pattern))
        return sorted(files)
    
    def _create_agent_for_algo(self, algo: str):
        """Create an empty agent instance for the given algorithm."""
        if algo == "Value Iteration":
            return ValueIteration(self.env, verbose=False)
        elif algo == "Q-Learning":
            return QLearning(self.env, verbose=False)
        elif algo == "SARSA(λ)":
            return SarsaLambda(self.env, verbose=False)
        return None
    
    def _load_model(self):
        """Load trained model from file dialog."""
        algo = self.algo_var.get()
        
        filepath = filedialog.askopenfilename(
            title="Load Model",
            initialdir=str(project_root / "results" / "models"),
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if filepath:
            self._load_model_from_path(filepath, algo)
    
    def _auto_load_model(self):
        """Auto-load the best available model for current algorithm."""
        algo = self.algo_var.get()
        model_files = self._find_model_files(algo)
        
        if not model_files:
            self.model_label.config(
                text=f"No trained model found for {algo}", 
                foreground="orange"
            )
            return
        
        # Use the first (or most relevant) model file
        filepath = str(model_files[0])
        self._load_model_from_path(filepath, algo)
    
    def _load_model_from_path(self, filepath: str, algo: str):
        """Load a model from a specific file path."""
        try:
            # Create a fresh agent instance
            agent = self._create_agent_for_algo(algo)
            if agent is None:
                messagebox.showerror("Error", f"Unknown algorithm: {algo}")
                return
            
            # Use the agent's own load method which handles the dict format
            agent.load(filepath)
            
            self.agent = agent
            
            # Update value/policy maps if available
            self._update_visualizations()
            
            model_name = Path(filepath).name
            self.model_label.config(text=f"✓ {model_name}", foreground="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.model_label.config(text=f"Load failed", foreground="red")
    
    def _on_algorithm_change(self):
        """Handle algorithm change."""
        self.agent = None
        self.model_label.config(text="No model loaded", foreground="gray")
        
        # Clear visualizations
        if self.renderer:
            self.renderer.set_value_map(None)
            self.renderer.set_policy_map(None)
        self._redraw()
        
        # Try to auto-load model for new algorithm
        self._auto_load_model()
    
    def _update_visualizations(self):
        """Update value and policy visualizations from loaded agent."""
        if self.agent is None or self.renderer is None:
            return
        
        algo = self.algo_var.get()
        
        try:
            if algo == "Value Iteration":
                # Extract V and policy
                value_map = {}
                policy_map = {}
                for state, value in self.agent.V.items():
                    if len(state) >= 2:
                        x, y = state[0], state[1]
                        value_map[(x, y)] = value
                if hasattr(self.agent, 'policy'):
                    for state, action in self.agent.policy.items():
                        if len(state) >= 2:
                            x, y = state[0], state[1]
                            policy_map[(x, y)] = action
                
                self.renderer.set_value_map(value_map)
                self.renderer.set_policy_map(policy_map)
            
            elif algo in ["Q-Learning", "SARSA(λ)"]:
                # Extract max Q-values and policy
                value_map = {}
                policy_map = {}
                
                for state in self.agent.Q.keys():
                    if len(state) >= 2:
                        x, y = state[0], state[1]
                        q_values = self.agent.Q[state]
                        value_map[(x, y)] = float(np.max(q_values))
                        policy_map[(x, y)] = int(np.argmax(q_values))
                
                self.renderer.set_value_map(value_map)
                self.renderer.set_policy_map(policy_map)
        
        except Exception as e:
            print(f"Warning: Could not extract visualizations: {e}")
    
    def _play(self):
        """Start playing episodes."""
        if self.agent is None:
            # Try auto-load first
            self._auto_load_model()
            if self.agent is None:
                messagebox.showwarning(
                    "No Model", 
                    "No trained model found!\n\n"
                    "Options:\n"
                    "1. Click 'Train' to train a new model\n"
                    "2. Click 'Load Model' to load a .pkl file\n"
                    "3. Click 'Auto-Load' to find existing models"
                )
                return
        
        self.is_running = True
        self.play_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        
        self._run_episode()
    
    def _pause(self):
        """Pause execution."""
        self.is_running = False
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
    
    def _reset(self):
        """Reset environment."""
        self.is_running = False
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        
        self.current_state = None
        self.episode_reward = 0
        self.episode_steps = 0
        
        if self.renderer:
            self.renderer.reset()
        self._redraw()
        
        self._update_info()
        self.status_label.config(text="Status: Ready")
        self.key_status_label.config(text="Key: ✗ | Door: Closed")
        self.energy_label.config(text="Energy: --")
    
    def _single_step(self):
        """Execute a single step."""
        if self.agent is None:
            self._auto_load_model()
            if self.agent is None:
                messagebox.showwarning("No Model", "Please train or load a model first!")
                return
        
        if self.current_state is None:
            # Start new episode
            self.current_state = self.env.reset()
            self.episode_reward = 0
            self.episode_steps = 0
            if self.renderer:
                self.renderer.reset_visited()
                self.renderer.set_key_collected(False)
                self.renderer.update_agent(self.canvas, self.current_state[0], self.current_state[1], 
                                         self.current_state[3] if len(self.current_state) > 3 else None)
                self._redraw()
        
        # Get action from agent
        is_greedy = (self.mode_var.get() == "evaluate")
        if hasattr(self.agent, 'get_action'):
            if isinstance(self.agent, ValueIteration):
                action = self.agent.get_action(self.current_state)
            else:
                action = self.agent.get_action(self.current_state, greedy=is_greedy)
        else:
            action = np.random.randint(4)
        
        # Take step
        next_state, reward, done, info = self.env.step(action)
        
        self.episode_reward += reward
        self.episode_steps += 1
        
        # Update state before drawing
        self.current_state = next_state
        
        # Update visualization
        x, y = next_state[0], next_state[1]
        has_key = next_state[2] if len(next_state) > 2 else 0
        energy = next_state[3] if len(next_state) > 3 else None
        
        if self.renderer:
            self.renderer.set_key_collected(bool(has_key))
            self.renderer.update_agent(self.canvas, x, y, energy)
        self._redraw()
        
        # Update key/door/energy info
        door_status = "Open" if self.env.door_opened else "Closed"
        key_status = "✓" if has_key else "✗"
        self.key_status_label.config(text=f"Key: {key_status} | Door: {door_status}")
        if energy is not None:
            self.energy_label.config(text=f"Energy: {energy}/{self.env.max_energy}")
        
        self._update_info()
        
        if done:
            # Episode finished
            self.total_episodes += 1
            if info.get('event') == 'goal_reached':
                self.successful_episodes += 1
                self.status_label.config(text="Status: ✅ SUCCESS!")
            elif info.get('event') == 'out_of_energy':
                self.status_label.config(text="Status: ⚡ Out of Energy")
            elif info.get('event') == 'max_steps_exceeded':
                self.status_label.config(text="Status: ⏰ Max Steps")
            else:
                self.status_label.config(text="Status: ❌ Failed")
            
            self._update_stats()
            self.current_state = None
        else:
            self.status_label.config(text=f"Status: Running... (action: {['↑','↓','←','→'][action]})")
    
    def _run_episode(self):
        """Run episode continuously."""
        if not self.is_running:
            return
        
        self._single_step()
        
        # Schedule next step
        delay = max(20, int(1000 / self.speed))  # Min 20ms delay
        self.root.after(delay, self._run_episode)
    
    def _start_training(self):
        """Start training the selected algorithm in background."""
        if self.is_training:
            messagebox.showinfo("Training", "Training is already in progress!")
            return
        
        algo = self.algo_var.get()
        
        # Confirm training
        episodes = 3000 if algo == "Q-Learning" else 3000
        if algo == "Value Iteration":
            msg = f"Train {algo}?\nThis may take a few minutes."
        else:
            msg = f"Train {algo} for {episodes} episodes?\nThis may take a few minutes."
        
        if not messagebox.askyesno("Train", msg):
            return
        
        self.is_training = True
        self.train_btn.config(state=tk.DISABLED)
        self.training_label.config(text="Training: In progress...")
        
        # Run training in background thread
        train_thread = threading.Thread(target=self._train_agent, args=(algo,), daemon=True)
        train_thread.start()
    
    def _train_agent(self, algo: str):
        """Train agent in background thread."""
        try:
            if algo == "Value Iteration":
                agent = ValueIteration(self.env, gamma=0.99, theta=1e-6, verbose=True)
                stats = agent.train()
                eval_stats = agent.evaluate(num_episodes=50)
                save_path = str(project_root / "results" / "models" / "value_iteration.pkl")
                agent.save(save_path)
                
            elif algo == "Q-Learning":
                agent = QLearning(
                    self.env, alpha=0.1, gamma=0.99,
                    episodes=3000, max_steps=500,
                    verbose=True, log_every=300
                )
                stats = agent.train()
                eval_stats = agent.evaluate(num_episodes=50)
                save_path = str(project_root / "results" / "models" / "q_learning.pkl")
                agent.save(save_path)
                
            elif algo == "SARSA(λ)":
                agent = SarsaLambda(
                    self.env, alpha=0.1, gamma=0.99,
                    lambda_param=0.7,
                    episodes=3000, max_steps=500,
                    verbose=True, log_every=300
                )
                stats = agent.train()
                eval_stats = agent.evaluate(num_episodes=50)
                save_path = str(project_root / "results" / "models" / "sarsa_lambda.pkl")
                agent.save(save_path)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")
            
            # Update GUI from main thread
            self.root.after(0, self._on_training_complete, agent, algo, eval_stats)
            
        except Exception as e:
            self.root.after(0, self._on_training_error, str(e))
    
    def _on_training_complete(self, agent, algo: str, eval_stats: dict):
        """Handle training completion (called on main thread)."""
        self.agent = agent
        self.is_training = False
        self.train_btn.config(state=tk.NORMAL)
        
        success_rate = eval_stats.get('success_rate', 0)
        mean_reward = eval_stats.get('mean_reward', 0)
        
        self.training_label.config(
            text=f"Training: Complete (SR: {success_rate:.0%}, R: {mean_reward:.1f})"
        )
        self.model_label.config(text=f"✓ {algo} (just trained)", foreground="green")
        
        # Update visualizations
        self._update_visualizations()
        self._redraw()
        
        messagebox.showinfo(
            "Training Complete", 
            f"{algo} training finished!\n\n"
            f"Success Rate: {success_rate:.1%}\n"
            f"Mean Reward: {mean_reward:.2f}\n\n"
            f"Model saved. Click Play to watch the agent!"
        )
    
    def _on_training_error(self, error_msg: str):
        """Handle training error."""
        self.is_training = False
        self.train_btn.config(state=tk.NORMAL)
        self.training_label.config(text="Training: Failed")
        messagebox.showerror("Training Error", f"Training failed:\n{error_msg}")
    
    def _on_speed_change(self, value):
        """Handle speed slider change."""
        self.speed = float(value)
        self.speed_label.config(text=f"{self.speed:.1f}x")
    
    def _redraw(self):
        """Redraw the maze."""
        if self.renderer is None:
            return
        
        self.renderer.draw_maze(
            self.canvas,
            show_visited=self.show_visited_var.get(),
            show_values=self.show_values_var.get(),
            show_policy=self.show_policy_var.get()
        )
        
        # Redraw agent if exists
        if self.current_state is not None:
            x, y = self.current_state[0], self.current_state[1]
            has_key = self.current_state[2] if len(self.current_state) > 2 else 0
            energy = self.current_state[3] if len(self.current_state) > 3 else None
            
            self.renderer.set_key_collected(bool(has_key))
            self.renderer.draw_agent(self.canvas, x, y, energy)
    
    def _update_info(self):
        """Update episode info display."""
        self.episode_label.config(text=f"Episode: {self.total_episodes + 1}")
        self.steps_label.config(text=f"Steps: {self.episode_steps}")
        self.reward_label.config(text=f"Reward: {self.episode_reward:.1f}")
    
    def _update_stats(self):
        """Update statistics display."""
        self.total_ep_label.config(text=f"Total Episodes: {self.total_episodes}")
        
        if self.total_episodes > 0:
            success_rate = (self.successful_episodes / self.total_episodes) * 100
            self.success_label.config(text=f"Success Rate: {success_rate:.1f}%")


def main():
    """Run GUI application."""
    root = tk.Tk()
    app = MazeGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
