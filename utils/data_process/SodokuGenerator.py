import json
import os
import sys
from typing import List, Tuple, Set
import random

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from ragen.env.sudoku.config import SudokuEnvConfig
from ragen.env.sudoku.env import SudokuEnv


class SudokuDataGenerator:
    def __init__(self):
        self.generated_envs: Set[str] = set()
        
    def generate_environments(self, min_level: int, max_level: int, num_attempts: int, grid_size: int=3) -> List[Tuple[int, int]]:
        """
        Generate non-duplicate Sudoku environment data
        
        Args:
            min_level: Minimum difficulty level (remove_num), range 1-3
            max_level: Maximum difficulty level (remove_num), range 1-3
            num_attempts: Number of attempts to generate data
            
        Returns:
            List[Tuple[int, int]]: List of (level, seed)
        """
        generated_data = []
        print(f"Starting to generate environment data...")
        print(f"Grid size: {grid_size}")
        print(f"Difficulty level range: {min_level} - {max_level} (remove_num)")
        print(f"Number of attempts: {num_attempts}")
        
        attempt_count = 0
        while len(generated_data) < num_attempts and attempt_count < num_attempts * 10:  # Prevent infinite loop
            attempt_count += 1
            
            # Randomly select difficulty level (remove_num)
            level = random.randint(min_level, max_level)
            
            # Randomly select seed
            seed = random.randint(1, 10000)
            
            # Create environment
            try:
                env = SudokuEnv(config=SudokuEnvConfig(grid_size=grid_size, seed=seed, remove_num=level))
                state = env.reset(seed=seed)
                
                # Use environment's string representation to check for duplicates
                # render() returns a string format grid
                state_str = str(state) if state is not None else ""
                
                env_str = f"3_{level}_{state_str}"
                
                if env_str not in self.generated_envs:
                    self.generated_envs.add(env_str)
                    generated_data.append((level, seed))
                    print(f"Generated environment {len(generated_data)}: level={level} (remove_num={level}), seed={seed}")
                else:
                    print(f"Environment duplicate, skipping: level={level}, seed={seed}")
                    
            except Exception as e:
                print(f"Failed to create environment: level={level}, seed={seed}, error={e}")
                continue
        
        print(f"Generation complete! Successfully generated {len(generated_data)} non-duplicate environments")
        return generated_data
    
    def save_data(self, data: List[Tuple[int, int]], min_level: int, max_level: int, grid_size:int , output_dir: str = "data/sudoku"):
        """
        Save generated data to JSON file
        
        Args:
            data: Generated data list
            min_level: Minimum difficulty level
            max_level: Maximum difficulty level
            output_dir: Output directory
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        filename = f"{len(data)}_gs{grid_size}_{min_level}_{max_level}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Prepare data format for saving
        save_data = [
            {
                "grid_size": grid_size,
                "level": level,
                "seed": seed,
            }
            for level, seed in data
        ]
        
        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Data saved to: {filepath}")
        return filepath


def main():
    """Main function for interactive parameter input"""
    generator = SudokuDataGenerator()
    
    print("=== Sudoku Environment Data Generator ===")
    
    # Get user input
    try:
        min_level = int(input("Please enter minimum difficulty level (remove_num, range 1-3): "))
        max_level = int(input("Please enter maximum difficulty level (remove_num, range 1-3): "))
        grid_size = int(input("Please enter grid size (default is 3): "))
        num_attempts = int(input("Please enter number of environments to generate: "))
        
        # Validate input
        # if min_level < 1 or min_level > 3:
        #     print("Error: Difficulty level must be between 1-3")
        #     return
        # if max_level < 1 or max_level > 3:
        #     print("Error: Difficulty level must be between 1-3")
        #     return
        if max_level < min_level:
            print("Error: Maximum difficulty level cannot be less than minimum difficulty level")
            return
        if num_attempts <= 0:
            print("Error: Number of generations must be greater than 0")
            return
            
    except ValueError:
        print("Error: Please enter a valid number")
        return
    
    # Generate environment data
    data = generator.generate_environments(min_level=min_level, max_level=max_level, grid_size=grid_size, num_attempts=num_attempts)
    
    if not data:
        print("Failed to generate any environment data")
        return
    
    # Save data
    filepath = generator.save_data(data=data, min_level=min_level, max_level=max_level, grid_size=grid_size )
    
    print(f"\n=== Generation Complete ===")
    print(f"Successfully generated {len(data)} non-duplicate environments")
    print(f"Data saved to: {filepath}")
    
    # Display some statistics
    levels = [item[0] for item in data]
    unique_levels = set(levels)
    print(f"\nStatistics:")
    print(f"- Difficulty level distribution: {dict((level, levels.count(level)) for level in sorted(unique_levels))}")
    print(f"- Average difficulty level: {sum(levels) / len(levels):.2f}")


if __name__ == "__main__":
    main()
