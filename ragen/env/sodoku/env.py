import gymnasium as gym
import numpy as np
import sys
import os
from typing import List, Any

# If relative import fails, try absolute import
from ragen.env.sodoku.utils import generate_sodoku_puzzle, is_valid_move, is_sodoku_complete, grid_to_string
from ragen.env.base import BaseDiscreteActionEnv
from ragen.env.sodoku.config import SodokuEnvConfig
from ragen.utils import all_seed


class SodokuEnv(BaseDiscreteActionEnv):
    def __init__(self, config=None, **kwargs):
        self.config = config or SodokuEnvConfig()
        self.grid_size = self.config.grid_size
        self.n = self.grid_size * self.grid_size
        self.max_steps = self.config.max_steps
        self.render_mode = self.config.render_mode
        self.GRID_LOOKUP = self.config.grid_lookup

        # Action space: each position can place numbers from 1 to n
        self.ACTION_SPACE = gym.spaces.Discrete(self.n * self.n * self.n)

        # State space: n x n grid, each position can be 0 to n
        self.OBSERVATION_SPACE = gym.spaces.Box(
            low=0, high=self.n, shape=(self.n, self.n), dtype=np.int32
        )

        BaseDiscreteActionEnv.__init__(self)

        # Initialize state
        self.puzzle = None
        self.solution = None
        self.current_grid = None
        self.step_count = 0
        self.fixed_positions = None  # Record positions filled in initial puzzle

    def reset(self, seed=None, mode=None):
        """Reset environment"""
        try:
            with all_seed(seed):
                self.puzzle, self.solution = generate_sodoku_puzzle(
                    grid_size=self.grid_size,
                    seed=seed or self.config.seed,
                    remove_num=self.config.remove_num
                )

            self.current_grid = self.puzzle.copy()
            self.step_count = 0

            # Record positions filled in initial puzzle
            self.fixed_positions = set()
            for i in range(self.n):
                for j in range(self.n):
                    if self.puzzle[i, j] != 0:
                        self.fixed_positions.add((i, j))

            return self.render()
        except (RuntimeError, RuntimeWarning) as e:
            next_seed = abs(hash(str(seed))) % (
                2 ** 32) if seed is not None else None
            return self.reset(next_seed)

    def step(self, action):
        """Execute action"""
        self.step_count += 1
        # print("Step action received:", action)
        # Parse action
        if isinstance(action, np.ndarray):
            action = action.tolist()

        if isinstance(action, (list, tuple)):
            if len(action) != 3:
                # raise ValueError("Action sequence must contain exactly three elements: (row, col, value)")
                return self.render(), -1, False, {
                    "action_is_effective": False,
                    "action_is_valid": False,
                    "success": False,
                    "reason": "invalid_action_format"
                }
            row, col, value = (int(action[0]), int(action[1]), int(action[2]))
        else:
            # row, col, value = self._parse_action(int(action))
            # raise ValueError("Action must be a list or tuple of (row, col, value)")
            return self.render(), -1, False, {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
                "reason": "invalid_action_format"
            }
        # Check if action is valid
        is_valid_code = self._is_valid_action(row, col, value)

        if is_valid_code != 0:
            # If action is invalid, return done=True
            return self.render(), is_valid_code, True, {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
                # "reason": "invalid_action"
            }

        # Execute action
        self.current_grid[row, col] = value

        # Check if complete
        is_complete = is_sodoku_complete(self.current_grid)

        # Calculate reward
        # reward = self._calculate_reward(row, col, value, is_complete)

        # Check if ended
        done = is_complete or self.step_count >= self.max_steps

        return self.render(), 0, done, {
            "action_is_effective": True,
            "action_is_valid": True,
            "success": is_complete,
            "step_count": self.step_count
        }

    def _parse_action(self, action: int) -> tuple:
        """Parse action ID to (row, col, value)"""
        # Action ID = row * n * n + col * n + (value - 1)
        # But need to ensure value is between 1 and n
        row = action // (self.n * self.n)
        col = (action % (self.n * self.n)) // self.n
        value = (action % self.n) + 1
        return row, col, value

    def _is_valid_action(self, row: int, col: int, value: int) -> int:
        """Check if action is valid"""
        # Check if position is within range
        if not (0 <= row < self.n and 0 <= col < self.n and 1 <= value <= self.n):
            return 1

        # Check if trying to modify fixed position
        if (row, col) in self.fixed_positions:
            return 2

        # Check if current position is already filled
        if self.current_grid[row, col] != 0:
            return 3

        # Check sudoku rules
        return 0 if is_valid_move(self.current_grid, row, col, value) else 4

    def _get_error_reason(self, error_code: int) -> str:
        """
        Convert error code to error reason description

        Args:
            error_code: Error code

        Returns:
            str: Error reason description
        """
        error_reasons = {
            -1: "Invalid format: action must contain exactly three elements <action>row col value</action>",
            1: "Position out of range",
            2: "Attempting to modify a fixed position",
            3: "Current position is already filled",
            4: "Violates sudoku rules"
        }
        return error_reasons.get(error_code, "Unknown error")

    def _is_invalid_action(self, action) -> tuple:
        """
        Check if action is invalid (called externally)

        Args:
            action: Action, can be list/tuple (row, col, value) or np.ndarray

        Returns:
            tuple: (is_invalid: bool, error_reason: str, row: int, col: int, value: int)
                   is_invalid=True means action is invalid, error_reason is error reason, row/col/value are parsed values
        """

        if isinstance(action, (list, tuple)):
            if len(action) != 3:
                return True, self._get_error_reason(-1), -1, -1, -1  # Format error
            row, col, value = (int(action[0]), int(action[1]), int(action[2]))
        elif isinstance(action, str):
            # String format, need to split
            parts = action.strip().split()
            if len(parts) != 3:
                return True, self._get_error_reason(-1), -1, -1, -1  # Format error
            try:
                row, col, value = (int(parts[0]), int(parts[1]), int(parts[2]))
            except ValueError:
                return True, self._get_error_reason(-1), -1, -1, -1  # Format error
        else:
            return True, self._get_error_reason(-1), -1, -1, -1  # Format error

        # Check if action is valid
        is_valid_code = self._is_valid_action(row, col, value)
        is_invalid = (is_valid_code != 0)
        error_reason = self._get_error_reason(
            is_valid_code) if is_invalid else ""

        return is_invalid, error_reason, row, col, value

    def safe_step(self, action):
        """
        Safely execute step, if action doesn't meet standards, skip this step instead of interrupting game

        Args:
            action: Action, can be list/tuple (row, col, value) or np.ndarray

        Returns:
            tuple: (obs, reward, done, info)
        """
        # Call check function externally
        is_invalid, error_reason, row, col, value = self._is_invalid_action(
            action)

        if is_invalid:
            # If action is invalid, don't execute step, return current state, but don't interrupt game
            current_obs = self.render()
            current_reward = 0  # Keep current reward
            info = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
                "reason": error_reason,
                "step_count": self.step_count
            }
            # done=False, don't interrupt game
            return current_obs, current_reward, False, info

        # If action is valid, execute step normally
        return self.step([row, col, value])

    def _calculate_reward(self, row: int, col: int, value: int, is_complete: bool) -> float:
        """Calculate reward"""
        if is_complete:
            return 10.0  # Large reward for completing sudoku

        # Check if this action makes sudoku closer to completion
        if is_valid_move(self.current_grid, row, col, value):
            return 1.0  # Small reward for valid action
        else:
            return -1.0  # Penalty for invalid action

    def render(self, mode=None):
        """Render environment state, return environment"""
        render_mode = mode if mode is not None else self.render_mode
        if render_mode == 'text':
            obs_str = grid_to_string(self.current_grid)
            return f"\n{obs_str}\n"
        elif render_mode == 'rgb_array':
            # Can add image rendering logic here
            return self.current_grid
        else:
            raise ValueError(f"Invalid mode: {render_mode}")

    def get_all_actions(self):
        """Get all possible actions"""
        return list(range(self.ACTION_SPACE.n))

    def close(self):
        """Close environment"""
        pass

if __name__ == '__main__':
    # Interactive sudoku game
    print("=== Sudoku Game Setup ===")
    grid_size = 3
    total_cells = (grid_size * grid_size) ** 2
    default_remove_num = int(total_cells * 0.3)
    remove_num = default_remove_num

    try:
        prompt = f"Please enter the number of cells to remove (0-{total_cells}, default {default_remove_num}): "
        num_input = input(prompt).strip()
        if num_input:
            candidate = int(num_input)
            if not (0 <= candidate <= total_cells):
                raise ValueError
            remove_num = candidate
    except (ValueError, EOFError):
        print(f"Invalid input, using default remove count {default_remove_num}")
        remove_num = default_remove_num

    print(f"Final remove count: {remove_num} / {total_cells}")
    print()
    
    config = SodokuEnvConfig(grid_size=grid_size, seed=5797, remove_num=remove_num)
    env = SodokuEnv(config)
    
    print("=== Sudoku Game ===")
    print("Game rules:")
    print(f"- Each row, column, and {env.grid_size}x{env.grid_size} subgrid must contain numbers 1-{env.n}")
    print("- Input format: row col value (e.g., 0 1 2 means fill number 2 at position (0,1))")
    print("- Enter 'quit' to exit game")
    print("- Enter 'reset' to restart")
    print("- Enter 'help' to view help")
    print()
    
    # Reset environment
    print("Initial state:")
    state = env.reset(seed=42)
    print(state)
    print()
    
    step_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input("Please enter (row col value) or command: ").strip()
            
            # Handle special commands
            if user_input.lower() == 'quit':
                print("Game ended!")
                break
            elif user_input.lower() == 'reset':
                print("Restarting game...")
                state = env.reset(seed=42)
                print(state)
                print()
                step_count = 0
                continue
            elif user_input.lower() == 'help':
                print("Help information:")
                print("- Enter three numbers: row col value")
                print("  e.g., 0 1 2 means fill number 2 at position (0,1)")
                print("- Enter 'quit' to exit game")
                print("- Enter 'reset' to restart")
                print("- Enter 'help' to view help")
                print()
                continue
            
            # Parse user input
            parts = user_input.split()
            if len(parts) != 3:
                print("❌ Input format error! Please enter three numbers: row col value")
                continue
            
            try:
                row = int(parts[0])
                col = int(parts[1])
                value = int(parts[2])
            except ValueError:
                print("❌ Input format error! Please enter three integers: row col value")
                continue
            
            # Check input range
            if not (0 <= row < env.n and 0 <= col < env.n and 1 <= value <= env.n):
                print(f"❌ Input out of range! Row and col should be 0-{env.n-1}, value should be 1-{env.n}")
                continue
            
            # Convert to action ID
            # action_id = row * env.n * env.n + col * env.n + (value - 1)
            action = [row, col, value]
            # Execute action
            obs, reward, done, info = env.safe_step(action)
            step_count += 1
            
            # Display results
            if info['action_is_valid']:
                print(f"✅ Success! Filled number {value} at position ({row},{col})")
                print(f"Reward: {reward}")
                print(f"Step count: {step_count}")
                print()
                print("Current state:")
                print(obs)
                print()
                
                if info['success']:
                    print("🎉 Congratulations! Sudoku completed!")
                    print(f"Total steps: {step_count}")
                    break
            else:
                print(f"❌ Invalid action! Filled number {value} at position ({row},{col})")
                print(f"Reason: {info.get('reason', 'Unknown')}")
                print(f"Reward: {reward}")
                print()
                
                if done:
                    print("Game ended!")
                    break
                    
        except KeyboardInterrupt:
            print("\nGame interrupted!")
            break
        except EOFError:
            print("\nDetected non-interactive environment, running demo mode...")
            print("=== Demo Mode ===")
            
            # Demo valid action - fill number 1 at empty position (0,0)
            print("1. Demo valid action - fill number 1 at position (0,0):")
            action_id = 0 * env.n * env.n + 0 * env.n + (1 - 1)
            obs, reward, done, info = env.step(action_id)
            print(f"Result: reward={reward}, valid={info['action_is_valid']}")
            print(f"New state:\n{obs}")
            
            # Demo invalid action - try to modify fixed position (0,1)
            print(f"\n2. Demo invalid action - try to modify fixed position (0,1):")
            action_id = 0 * env.n * env.n + 1 * env.n + (1 - 1)
            obs, reward, done, info = env.step(action_id)
            print(f"Result: reward={reward}, valid={info['action_is_valid']}, reason={info.get('reason', 'Unknown')}")
            
            print("\nDemo completed!")
            break
        except Exception as e:
            print(f"❌ Error occurred: {e}")
            continue
