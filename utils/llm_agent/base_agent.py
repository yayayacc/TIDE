from typing import Dict, Any, List
class BaseAgent:
    """Abstract base class for LLM agents."""
    def reset(self):
        pass
    def get_next_step(self, prompt: str, observation: str) -> Dict[str, Any]:
        pass
    def close(self):
        pass
    def get_next_step_parallel(self, trajectories) -> List[Dict[str, Any]]:
        pass


