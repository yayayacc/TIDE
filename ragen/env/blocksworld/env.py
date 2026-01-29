import os
import re
import sys
from typing import Any, Dict, List, Set, Tuple, Optional

from ragen.env.base import BaseDiscreteActionEnv
from ragen.env.blocksworld.config import BlocksworldEnvConfig


class BlocksworldEnv(BaseDiscreteActionEnv):
    """
    Blocksworld gym environment.
    
    """

    def __init__(self, config: Optional[BlocksworldEnvConfig] = None):
        self.config = config or BlocksworldEnvConfig()

        # State
        self.objects: Set[str] = set()
        self.state: Set[str] = set()
        self.goal: Set[str] = set()
        self.holding: Optional[str] = None
        self.step_count: int = 0

        # Initialize parent class
        super().__init__()

    def reset(self, game_file: Dict[str, Any] = None, mode: str = "test") -> Any:
        """Reset environment, initialize objects, initial state and goal according to config's PDDL."""
        # Read PDDL
        pddl_text: str = game_file.get("label", None)
        if pddl_text is None:
            raise ValueError(
                "PDDL text must be provided for BlocksworldEnv reset."
            )
        self.instruction_text: str = game_file.get("query", "")
        if "# Goal State" in self.instruction_text:
            goal_state_index = self.instruction_text.find("# Goal State")
            # Extract content after "# Goal State" (skip "# Goal State" itself)
            goal_content = self.instruction_text[goal_state_index + len("# Goal State"):].strip()
            self.instruction_text = f"Your Goal is: {goal_content}"
        self.level: int = game_file.get("level", -1)
        # Parse PDDL
        self._parse_pddl(pddl_text)

        # Reset step count
        self.step_count = 0

        # If init doesn't explicitly contain holding, default to handempty
        if not any(f.startswith("holding ") for f in self.state):
            self.state.add("handempty")
            self.holding = None
        else:
            # Synchronize holding variable
            for f in self.state:
                if f.startswith("holding "):
                    self.holding = f.split()[1]
                    break

        return self.render()

    def step(self, action: str) -> Tuple[Any, float, bool, Dict]:
        """
        Execute action (string):
          - pickup x
          - putdown x
          - stack x y
          - unstack x y
        Returns: (obs, reward=0.0, done, info)
        """
        self.step_count += 1

        applied = False
        valid_format = True

        try:
            tokens = action.strip().split()
            if not tokens:
                valid_format = False
            else:
                op = tokens[0]
                if op == "pickup":
                    if len(tokens) != 2:
                        valid_format = False
                    else:
                        x = tokens[1]
                        applied, _, _ = self._op_pickup(x)
                elif op == "putdown":
                    if len(tokens) != 2:
                        valid_format = False
                    else:
                        x = tokens[1]
                        applied, _, _ = self._op_putdown(x)
                elif op == "stack":
                    if len(tokens) != 3:
                        valid_format = False
                    else:
                        x, y = tokens[1], tokens[2]
                        applied, _, _ = self._op_stack(x, y)
                elif op == "unstack":
                    if len(tokens) != 3:
                        valid_format = False
                    else:
                        x, y = tokens[1], tokens[2]
                        applied, _, _ = self._op_unstack(x, y)
                else:
                    valid_format = False
        except Exception:
            valid_format = False

        success = self._goal_satisfied()
        done = success or (self.step_count >= self.config.max_steps)

        obs = self.render()
        reward = 0.0  # No reward needed, fixed at 0

        info = {
            "action_is_valid": bool(valid_format and applied),
            "success": success,
                }
        return obs, reward, done, info

    def get_all_actions(self) -> List[str]:
        """List all instantiable actions (does not guarantee current step is executable, only lists domain actions)."""
        acts: List[str] = []
        for x in self.objects:
            acts.append(f"pickup {x}")
            acts.append(f"putdown {x}")
        for x in self.objects:
            for y in self.objects:
                if x != y:
                    acts.append(f"stack {x} {y}")
                    acts.append(f"unstack {x} {y}")
        return acts

    def get_available_actions(self) -> List[str]:
        """Return executable actions in current state (satisfying preconditions)."""
        acts: List[str] = []

        holding_obj: Optional[str] = next(
            (f.split()[1] for f in self.state if f.startswith("holding ")), None
        )

        if holding_obj:
            # putdown x
            acts.append(f"putdown {holding_obj}")
            # stack x y
            x = holding_obj
            for y in self.objects:
                if y != x and f"clear {y}" in self.state:
                    acts.append(f"stack {x} {y}")
        elif "handempty" in self.state:
            # pickup x
            for x in self.objects:
                if f"ontable {x}" in self.state and f"clear {x}" in self.state:
                    acts.append(f"pickup {x}")
            # unstack x y
            for x in self.objects:
                for y in self.objects:
                    if x != y and f"on {x} {y}" in self.state and f"clear {x}" in self.state:
                        acts.append(f"unstack {x} {y}")
        return acts
    # Render
    def render(self) -> str:

        return self._current_state_str()

    def close(self):
        pass

    # --- Internal logic ---

    def _parse_pddl(self, content: str):
        self.objects.clear()
        self.state.clear()
        self.goal.clear()
        self.holding = None

        # objects
        obj_match = re.search(r":objects\s+([^\)]+)\)", content)
        if obj_match:
            self.objects = set(obj_match.group(1).split())

        # init
        init_match = re.search(r":init\s*((?:.|\n)*?)\)\s*\(:goal", content)
        if not init_match:
            init_match = re.search(r":init\s*((?:.|\n)*?)\)", content)
        if init_match:
            inits = re.findall(r"\(([^)]+)\)", init_match.group(1))
            for item in inits:
                self.state.add(item.strip())

        # goal
        goal_match = re.search(r":goal\s*\(and([\s\S]*?)\)\)", content)
        if goal_match:
            goals = re.findall(r"\(([^)]+)\)", goal_match.group(1))
            for item in goals:
                self.goal.add(item.strip())

    def _goal_satisfied(self) -> bool:
        # Keep consistent with original simulator: require state equals goal (not subset)
        return self.goal.issubset(self.state) and self.state.issubset(self.goal)

    def _op_pickup(self, x: str) -> Tuple[bool, int, str]:
        if x not in self.objects:
            return False, -1, f"unknown object {x}"
        if (
            "handempty" in self.state
            and f"ontable {x}" in self.state
            and f"clear {x}" in self.state
        ):
            self.state.remove("handempty")
            self.state.remove(f"ontable {x}")
            self.state.remove(f"clear {x}")
            self.state.add(f"holding {x}")
            self.holding = x
            return True, 1, ""
        return False, 0, "precondition not satisfied"

    def _op_putdown(self, x: str) -> Tuple[bool, int, str]:
        if x not in self.objects:
            return False, -1, f"unknown object {x}"
        if f"holding {x}" in self.state:
            self.state.remove(f"holding {x}")
            self.state.add(f"ontable {x}")
            self.state.add(f"clear {x}")
            self.state.add("handempty")
            self.holding = None
            return True, 1, ""
        return False, 0, "precondition not satisfied"

    def _op_stack(self, x: str, y: str) -> Tuple[bool, int, str]:
        if x not in self.objects or y not in self.objects:
            return False, -1, f"unknown object(s) {x}, {y}"
        if x == y:
            return False, -1, "x and y must be different"
        if f"holding {x}" in self.state and f"clear {y}" in self.state:
            self.state.remove(f"holding {x}")
            self.state.remove(f"clear {y}")
            self.state.add(f"on {x} {y}")
            self.state.add(f"clear {x}")
            self.state.add("handempty")
            self.holding = None
            return True, 1, ""
        return False, 0, "precondition not satisfied"

    def _op_unstack(self, x: str, y: str) -> Tuple[bool, int, str]:
        if x not in self.objects or y not in self.objects:
            return False, -1, f"unknown object(s) {x}, {y}"
        if x == y:
            return False, -1, "x and y must be different"
        if (
            f"on {x} {y}" in self.state
            and f"clear {x}" in self.state
            and "handempty" in self.state
        ):
            self.state.remove(f"on {x} {y}")
            self.state.remove(f"clear {x}")
            self.state.remove("handempty")
            self.state.add(f"holding {x}")
            self.state.add(f"clear {y}")
            self.holding = x
            return True, 1, ""
        # Original simulator returns False here, map it to 0 (not satisfied)
        return False, 0, "precondition not satisfied"

    def _current_state_str(self) -> str:
        lines: List[str] = []
        lines.append(f"I have {len(self.objects)} blocks.")
        ontable, clear, on = [], [], []
        handempty = False
        holding = None

        for fact in self.state:
            parts = fact.split()
            if parts[0] == "ontable":
                ontable.append(parts[1])
            elif parts[0] == "clear":
                clear.append(parts[1])
            elif parts[0] == "on":
                on.append((parts[1], parts[2]))
            elif parts[0] == "handempty":
                handempty = True
            elif parts[0] == "holding":
                holding = parts[1]

        for a, b in on:
            lines.append(f"{a} is on {b}.")
        for b in ontable:
            lines.append(f"{b} is on the table.")
        for b in clear:
            lines.append(f"{b} is clear.")
        if handempty:
            lines.append("My hand is empty.")
        elif holding:
            lines.append(f"I am holding {holding}.")
        return " ".join(lines)

if __name__ == "__main__":
    import json
    data = json.load(open("./data/blocksworld/1200_1_12.json", "r"))
    env = BlocksworldEnv()
    env.reset(data[101])
    print("instruction:", env.instruction_text)
    print("level:", env.level)
    done = False
    while not done:
        print("Current State:")
        print(env.render())
        action=input("Enter action (or 'exit' to quit): ")
        if action.lower() == "exit":
            break
        obs, reward, done, info = env.step(action)
        print(f"Done: {done}, info: {info}\n")