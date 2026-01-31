from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SudokuEnvConfig:
    """Sudoku environment configuration class"""
    
    # Grid size (3x3, 4x4, 5x5, etc.)
    grid_size: int = 3
    
    # Random seed
    seed: int = 42
    
    # Difficulty level ("easy", "medium", "hard")
    difficulty: str = "easy"
    
    # Directly specify remove count, if specified then ignore difficulty parameter
    remove_num: Optional[int] = 2
    
    # Maximum steps
    max_steps: int = 100
    
    # Render mode
    render_mode: str = 'text'
    
    # Grid lookup table (for rendering)
    grid_lookup: Dict[int, str] = None
    
    # Action lookup table
    action_lookup: Dict[int, str] = None
    
    def __post_init__(self):
        if self.grid_lookup is None:
            self.grid_lookup = {
                0: "_",  # Empty position
                1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 
                6: "6", 7: "7", 8: "8", 9: "9"
            }
        
        if self.action_lookup is None:
            # Action format: (row, col, value)
            self.action_lookup = {}
            action_id = 0
            for row in range(self.grid_size * self.grid_size):
                for col in range(self.grid_size * self.grid_size):
                    for value in range(1, self.grid_size * self.grid_size + 1):
                        self.action_lookup[action_id] = f"({row},{col},{value})"
                        action_id += 1
