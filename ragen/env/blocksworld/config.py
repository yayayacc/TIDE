from typing import Optional


class BlocksworldEnvConfig:
    def __init__(
        self,
        render_mode: str = "text",
    ):
        # If no PDDL provided, use a simple default problem
        self.max_steps = 100
        self.render_mode = render_mode