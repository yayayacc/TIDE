import numpy as np
import random
from typing import Tuple, List, Optional


def is_valid_move(grid: np.ndarray, row: int, col: int, value: int) -> bool:
    """Check if placing specified value at specified position is valid"""
    grid_size = int(np.sqrt(len(grid)))
    
    # Check row
    if value in grid[row, :]:
        return False
    
    # Check column
    if value in grid[:, col]:
        return False
    
    # Check 3x3 subgrid
    start_row = (row // grid_size) * grid_size
    start_col = (col // grid_size) * grid_size
    subgrid = grid[start_row:start_row + grid_size, start_col:start_col + grid_size]
    if value in subgrid:
        return False
    
    return True


def solve_sudoku(grid: np.ndarray) -> Optional[np.ndarray]:
    """Solve sudoku using backtracking algorithm"""
    grid_size = int(np.sqrt(len(grid)))
    
    def backtrack():
        for row in range(grid_size * grid_size):
            for col in range(grid_size * grid_size):
                if grid[row, col] == 0:  # Empty position
                    for value in range(1, grid_size * grid_size + 1):
                        if is_valid_move(grid, row, col, value):
                            grid[row, col] = value
                            if backtrack():
                                return True
                            grid[row, col] = 0  # Backtrack
                    return False
        return True
    
    # Create copy to avoid modifying original grid
    solution = grid.copy()
    if backtrack():
        return solution
    return None


def generate_sudoku_puzzle(grid_size: int = 3, seed: int = 42, remove_num: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a solvable sudoku puzzle

    Args:
        grid_size: Subgrid size (3=9x9, 2=4x4)
        seed: Random seed
        remove_num: Directly specify remove count (0-n*n), if not specified use default ratio
    """
    random.seed(seed)
    np.random.seed(seed)
    
    n = grid_size * grid_size
    solution = _generate_complete_solution(n)

    # Randomly remove some numbers to create puzzle
    puzzle = solution.copy()

    total_cells = n * n

    if remove_num is None:
        remove_num = int(total_cells * 0.2)
    else:
        remove_num = max(0, min(total_cells, int(remove_num)))
    positions = [(i, j) for i in range(n) for j in range(n)]
    random.shuffle(positions)
    
    for i in range(remove_num):
        row, col = positions[i]
        puzzle[row, col] = 0
    
    return puzzle, solution


def _generate_complete_solution(n: int) -> np.ndarray:
    """Generate complete sudoku solution using backtracking"""
    grid = np.zeros((n, n), dtype=int)

    def backtrack() -> bool:
        for row in range(n):
            for col in range(n):
                if grid[row, col] == 0:
                    candidates = list(range(1, n + 1))
                    random.shuffle(candidates)
                    for value in candidates:
                        if is_valid_move(grid, row, col, value):
                            grid[row, col] = value
                            if backtrack():
                                return True
                            grid[row, col] = 0
                    return False
        return True

    if not backtrack():
        raise RuntimeError("Failed to generate a complete Sudoku solution")

    return grid


def fill_subgrid(grid: np.ndarray, start_row: int, start_col: int):
    """Fill 3x3 subgrid"""
    grid_size = int(np.sqrt(len(grid)))
    numbers = list(range(1, grid_size * grid_size + 1))
    random.shuffle(numbers)
    
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            grid[start_row + i, start_col + j] = numbers[idx]
            idx += 1


def is_sudoku_complete(grid: np.ndarray) -> bool:
    """Check if sudoku is complete and correct"""
    grid_size = int(np.sqrt(len(grid)))
    n = grid_size * grid_size
    
    # Check if there are empty positions
    if np.any(grid == 0):
        return False
    
    # Check each row
    for row in range(n):
        if len(set(grid[row, :])) != n:
            return False
    
    # Check each column
    for col in range(n):
        if len(set(grid[:, col])) != n:
            return False
    
    # Check each subgrid
    for i in range(0, n, grid_size):
        for j in range(0, n, grid_size):
            subgrid = grid[i:i+grid_size, j:j+grid_size].flatten()
            if len(set(subgrid)) != n:
                return False
    
    return True


def grid_to_string(grid: np.ndarray) -> str:
    """Convert sudoku grid to string representation"""
    rows = []
    for row in grid:
        row_str = ''.join(str(cell) if cell != 0 else '_' for cell in row)
        rows.append(row_str)
    return '\n'.join(rows)


def string_to_grid(grid_str: str) -> np.ndarray:
    """Convert string representation to sudoku grid"""
    lines = grid_str.strip().split('\n')
    n = len(lines)
    grid = np.zeros((n, n), dtype=int)
    
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char == '_':
                grid[i, j] = 0
            else:
                grid[i, j] = int(char)
    
    return grid
