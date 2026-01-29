import json
import os
import sys
from typing import List, Tuple, Set
import random

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from ragen.env.frozen_lake.config import FrozenLakeEnvConfig
from ragen.env.frozen_lake.env import FrozenLakeEnv


class FrozenLakeDataGenerator:
    def __init__(self):
        self.generated_envs: Set[str] = set()
        
    def generate_environments(self, min_grid_size: int, max_grid_size: int, num_attempts: int) -> List[Tuple[int, int]]:
        """
        Generate non-duplicate Frozen Lake environment data
        
        Args:
            min_grid_size: Minimum grid size
            max_grid_size: Maximum grid size
            num_attempts: Number of attempts to generate data
            
        Returns:
            List[Tuple[int, int]]: List of (grid_size, seed)
        """
        generated_data = []
        
        print(f"Starting to generate environment data...")
        print(f"Grid size range: {min_grid_size} - {max_grid_size}")
        print(f"Number of attempts: {num_attempts}")
        
        attempt_count = 0
        while len(generated_data) < num_attempts and attempt_count < num_attempts * 10:  # Prevent infinite loop
            attempt_count += 1
            
            # Randomly select grid size
            grid_size = random.randint(min_grid_size, max_grid_size)
            
            # Randomly select seed
            seed = random.randint(1, 10000)
            
            # Create environment
            try:
                env = FrozenLakeEnv(config=FrozenLakeEnvConfig(size=grid_size, is_slippery=False, map_seed=seed))
                state = env.reset(seed=seed)
                
                # Use environment string representation to check for duplicates
                env_str = f"{grid_size}_{state}"
                
                if env_str not in self.generated_envs:
                    self.generated_envs.add(env_str)
                    generated_data.append((grid_size, seed))
                    print(f"Generated environment {len(generated_data)}: grid_size={grid_size}, seed={seed}")
                else:
                    print(f"Duplicate environment, skipping: grid_size={grid_size}, seed={seed}")
                    
            except Exception as e:
                print(f"Failed to create environment: grid_size={grid_size}, seed={seed}, error={e}")
                continue
        
        print(f"Generation completed! Successfully generated {len(generated_data)} unique environments")
        return generated_data
    
    def save_data(self, data: List[Tuple[int, int]], min_grid_size: int, max_grid_size: int, output_dir: str = "data/frozen_lake"):
        """
        Save generated data to JSON file
        
        Args:
            data: List of generated data
            min_grid_size: Minimum grid size
            max_grid_size: Maximum grid size
            output_dir: Output directory
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        filename = f"{len(data)}_{min_grid_size}_{max_grid_size}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Prepare data format for saving
        save_data =  [
                {
                    "grid_size": grid_size,
                    "seed": seed,
                }
                for grid_size, seed in data
            ]
        
        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Data saved to: {filepath}")
        return filepath


def main():
    """Main function for interactive parameter input"""
    generator = FrozenLakeDataGenerator()
    
    print("=== Frozen Lake Environment Data Generator ===")
    
    # Get user input
    try:
        min_grid_size = int(input("Please enter minimum grid size (recommended 4-8): "))
        max_grid_size = int(input("Please enter maximum grid size (recommended 4-12): "))
        num_attempts = int(input("Please enter number of environments to generate: "))
        
        # Validate input
        if min_grid_size < 4:
            print("Warning: Grid size less than 4 may not be suitable")
        if max_grid_size < min_grid_size:
            print("Error: Maximum grid size cannot be less than minimum grid size")
            return
        if num_attempts <= 0:
            print("Error: Number of generations must be greater than 0")
            return
            
    except ValueError:
        print("Error: Please enter a valid number")
        return
    
    # Generate environment data
    data = generator.generate_environments(min_grid_size, max_grid_size, num_attempts)
    
    if not data:
        print("Failed to generate any environment data")
        return
    
    # Save data
    filepath = generator.save_data(data, min_grid_size, max_grid_size)
    
    print(f"\n=== Generation Completed ===")
    print(f"Successfully generated {len(data)} unique environments")
    print(f"Data saved to: {filepath}")
    
    # Display some statistics
    grid_sizes = [item[0] for item in data]
    unique_sizes = set(grid_sizes)
    print(f"\nStatistics:")
    print(f"- Grid size distribution: {dict((size, grid_sizes.count(size)) for size in sorted(unique_sizes))}")
    print(f"- Average grid size: {sum(grid_sizes) / len(grid_sizes):.2f}")


if __name__ == "__main__":
    main()
