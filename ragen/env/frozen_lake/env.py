import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv
import numpy as np
from ragen.env.frozen_lake.config import FrozenLakeEnvConfig
from ragen.env.frozen_lake.utils import generate_random_map
from ragen.utils import all_seed
from ragen.env.base import BaseDiscreteActionEnv

class FrozenLakeEnv(BaseDiscreteActionEnv, GymFrozenLakeEnv):
    def __init__(self, config: FrozenLakeEnvConfig = FrozenLakeEnvConfig()):
        # Using mappings directly from config
        self.config = config
        self.GRID_LOOKUP = config.grid_lookup
        self.ACTION_LOOKUP = config.action_lookup
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)
        self.render_mode = config.render_mode
        self.action_map = config.action_map
        self.MAP_LOOKUP = config.map_lookup
        random_map = generate_random_map(size=config.size, p=config.p, seed=config.map_seed)
        BaseDiscreteActionEnv.__init__(self)
        GymFrozenLakeEnv.__init__(
            self,
            desc=random_map,
            is_slippery=config.is_slippery,
            render_mode=config.render_mode
        )

    def reset(self, seed=None, mode=None):
        try:
            with all_seed(seed):
                self.config.map_seed = seed
                self.__init__(self.config)   
                GymFrozenLakeEnv.reset(self, seed=seed)
                return self.render()
        except (RuntimeError, RuntimeWarning) as e:
            next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
            return self.reset(next_seed)

    def _will_fall_into_hole(self, action: int) -> bool:
        """
        Check if executing specified action will fall into a hole

        Args:
            action: Action number (1: Left, 2: Down, 3: Right, 4: Up)

        Returns:
            bool: Returns True if will fall into hole, False otherwise
        """
        current_row, current_col = self.player_pos
        # Convert to gymnasium action (0: LEFT, 1: DOWN, 2: RIGHT, 3: UP)
        gym_action = self.action_map[action]

        # Calculate next position (without considering sliding, directly calculate by action)
        next_row, next_col = current_row, current_col
        if gym_action == 0:  # LEFT
            next_col = max(0, current_col - 1)
        elif gym_action == 1:  # DOWN
            next_row = min(len(self.desc) - 1, current_row + 1)
        elif gym_action == 2:  # RIGHT
            next_col = min(self.ncol - 1, current_col + 1)
        elif gym_action == 3:  # UP
            next_row = max(0, current_row - 1)

        # Check if next position is a hole
        return self.desc[next_row, next_col] == b'H'

    def step(self, action: str):
        prev_pos = int(self.s)
        if action.lower() == "left":
            action = 1
        elif action.lower() == "down":
            action = 2
        elif action.lower() == "right":
            action = 3
        elif action.lower() == "up":
            action = 4
        else:
            return self.render(), 0.0, False, {"action_is_effective": False, "action_is_valid": False, "success": False}
        _, reward, done, _, _ = GymFrozenLakeEnv.step(self, self.action_map[action])
        next_obs = self.render()
        info = {"action_is_effective": prev_pos != int(self.s), "action_is_valid": True, "success": self.desc[self.player_pos] == b"G"}

        return next_obs, reward, done, info

    def safe_step(self, action: str):
        """
        Safely execute step, checks if will fall into hole before execution

        Args:
            action: Action string ("left", "down", "right", "up")

        Returns:
            tuple: (obs, reward, done, info)
        """
        # Convert action string to number
        if action.lower() == "left":
            action_num = 1
        elif action.lower() == "down":
            action_num = 2
        elif action.lower() == "right":
            action_num = 3
        elif action.lower() == "up":
            action_num = 4
        else:
            return self.render(), 0.0, False, {"action_is_effective": False, "action_is_valid": False, "success": False}

        # Call check function externally
        if self._will_fall_into_hole(action_num):
            # If will fall into hole, don't execute step, return current state
            current_obs = self.render()
            current_reward = 0  # Keep current reward (usually 0)
            info = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
                "reason": "Will fall into hole"
            }
            return current_obs, current_reward, False, info

        # If won't fall into hole, execute step normally
        return self.step(action)

    def render(self):
        if self.render_mode == 'text':
            room = self.desc.copy()
            # replace the position of start 'S' with 'F', mark the position of the player as 'p'.
            room = np.where(room == b'S', b'F', room)
            room[self.player_pos] = b'P'
            room = np.vectorize(lambda x: self.MAP_LOOKUP[x])(room)
            # add player in hole or player on goal
            room[self.player_pos] = 4 if self.desc[self.player_pos] == b'H' else 5 if self.desc[self.player_pos] == b'G' else 0
            obs_str = '\n'.join(''.join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room)
            return f"\n{obs_str}\n"
        elif self.render_mode == 'rgb_array':
            return self._render_gui('rgb_array')
        else:
            raise ValueError(f"Invalid mode: {self.render_mode}")

    def get_all_actions(self):
        return list([k for k in self.ACTION_LOOKUP.keys()])

    @property
    def player_pos(self):
        return (self.s // self.ncol, self.s % self.ncol)  # (row, col)

    def close(self):
        self.render_cache = None
        super(FrozenLakeEnv, self).close()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    config = FrozenLakeEnvConfig(size=11, p=0.8, is_slippery=False, map_seed=1537)
    env = FrozenLakeEnv(config)
    print(env.reset(seed=1537))
    while True:
        keyboard = input("Enter action: ")
        if keyboard == 'q':
            break
        action = keyboard.lower()
        obs, reward, done, info = env.safe_step(action)
        print("output:",obs, reward, done, info)
    # np_img = env.render('rgb_array')
    # save the image
    # plt.imsave('frozen_lake.png', np_img)
