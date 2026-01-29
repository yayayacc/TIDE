
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

@dataclass
class StepMemory:
    
    observation: str = None
    true_state: str = None
    input_state: str = None
    analysis: str = None
    action: str = None
    is_valid: bool = True
    feedback: str = ""
    previous_memory: Optional['StepMemory'] = None
    

class ContextManager:
    def __init__(self,system_prompt: str, instruction_prompt: str,tokenizer, config):
        self.system_prompt: str = system_prompt
        self.instruction_prompt: str = instruction_prompt
        self.history : List[StepMemory] = []
        self.tokenizer = tokenizer
        self.chat_format = config.agent_proxy.chat_format
        self.state = config.agent_proxy.state
        self.history_has_cot = config.agent_proxy.history_has_cot
        self.enable_thinking = config.agent_proxy.enable_thinking
        if self.chat_format == "user_assistant_format_part":
            self.history_window_size = config.agent_proxy.history_window_size

    def format_prompt(self) -> str:
        """
        Format the prompt based on the chat format and history.
        Returns:
            The formatted prompt string.
        """
        if self.chat_format == "default_format":
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.instruction_prompt},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                enable_thinking=False,
                add_generation_prompt=True,
            )

            for memory in self.history[:-1]:
                prompt += "<step>\n"
                if self.state != "no":
                    prompt += f"\t<state>{memory.input_state}</state>\n"
                if self.history_has_cot:
                    prompt += f"\t<analysis>{memory.analysis}</analysis>\n"
                prompt += (
                    f"\t<action>{memory.action}</action>\n</step>\n"
                )
            prompt += "<step>\n"
            if self.state != "no":
                prompt += f"\t<state>{self.history[-1].input_state}</state>\n"

        elif self.chat_format == "user_assistant_format":
            messages = [
                {"role": "system", "content": self.system_prompt},
            ]
            if self.state == "no":
                messages.append(
                    {"role": "user", "content": self.instruction_prompt}
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": f"{self.instruction_prompt}\nCurrent state: <state>{self.history[0].input_state}</state>",
                    }
                )
            if len(self.history) == 1:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    enable_thinking=self.enable_thinking,
                    add_generation_prompt=True,
                )
                return prompt
            agent_msg = ""
            if self.history_has_cot:
                agent_msg += f"<analysis> {self.history[0].analysis} </analysis>\n"
            agent_msg += f"<action> {self.history[0].action} </action>\n"
            messages.append({"role": "assistant", "content": agent_msg})
            for memory in self.history[1:-1]:
                user_content = ""
                if self.state != "no": 
                    user_content = f"<state>{memory.input_state}</state>"
                messages.append({"role": "user", "content": user_content})
                agent_msg = ""
                if self.history_has_cot:
                    agent_msg += f"<analysis> {memory.analysis} </analysis>\n"
                agent_msg += f"<action> {memory.action} </action>\n"
                messages.append({"role": "assistant", "content": agent_msg})
            last_user_content = ""
            if self.state != "no":
                last_user_content = f"<state>{self.history[-1].input_state}</state>"
            messages.append({"role": "user", "content": last_user_content})
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                enable_thinking=self.enable_thinking,
                add_generation_prompt=True,
            )
                
        elif self.chat_format == "user_assistant_format_part":
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.instruction_prompt},
            ]
            start_index = max(0, len(self.history) - 1 - self.history_window_size)
            process_history = self.history[start_index:]
            if len(process_history) == 1:
                if self.state != "no":
                    messages[1]["content"] += f"\ncurrent state: <state>{process_history[0].input_state}</state>"
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    enable_thinking=self.enable_thinking,
                    add_generation_prompt=True,
                )
                return prompt

            # Truncate history based on history_window_size
            # window_size controls the number of assistant replies to keep
            # The actual history length participating in prompt construction is window_size + 1 (including the current one to be generated)
            
            # First user-assistant conversation turn
            first_memory = process_history[0]
            if self.state != "no":
                 messages[1]["content"] += f"\n<state>{first_memory.input_state}</state>"
            
            agent_msg = ""
            if self.history_has_cot:
                agent_msg += f"<analysis> {first_memory.analysis} </analysis>\n"
            agent_msg += f"<action> {first_memory.action} </action>\n"
            messages.append({"role": "assistant", "content": agent_msg})

            # Intermediate history records
            for memory in process_history[1:-1]:
                user_content = ""
                if self.state != "no":
                    user_content = f"<state>{memory.input_state}</state>"
                messages.append({"role": "user", "content": user_content})
                
                agent_msg = ""
                if self.history_has_cot:
                    agent_msg += f"<analysis> {memory.analysis} </analysis>\n"
                agent_msg += f"<action> {memory.action} </action>\n"
                messages.append({"role": "assistant", "content": agent_msg})

            # Last user input
            last_user_content = ""
            if self.state != "no":
                last_user_content = f"current state: <state>{process_history[-1].input_state}</state>"
            messages.append({"role": "user", "content": last_user_content})

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                enable_thinking=self.enable_thinking,
                add_generation_prompt=True,
            )

        return prompt

    def format_messages(self) -> List[Dict[str, str]]:
        if self.chat_format == "default_format":
            raise NotImplementedError("format_messages not implemented for default_format")

        elif self.chat_format == "user_assistant_format":
            messages = [
                {"role": "system", "content": self.system_prompt},
            ]
            if self.state == "no":
                messages.append(
                    {"role": "user", "content": self.instruction_prompt}
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": f"{self.instruction_prompt}\n<state>{self.history[0].input_state}</state>",
                    }
                )
            if len(self.history) == 1:

                return messages
            agent_msg = ""
            if self.history_has_cot:
                agent_msg += f"<analysis> {self.history[0].analysis} </analysis>\n"
            agent_msg += f"<action> {self.history[0].action} </action>\n"
            messages.append({"role": "assistant", "content": agent_msg})
            for memory in self.history[1:-1]:
                user_content = ""
                if self.state != "no": 
                    user_content = f"<state>{memory.input_state}</state>"
                messages.append({"role": "user", "content": user_content})
                agent_msg = ""
                if self.history_has_cot:
                    agent_msg += f"<analysis> {memory.analysis} </analysis>\n"
                agent_msg += f"<action> {memory.action} </action>\n"
                messages.append({"role": "assistant", "content": agent_msg})
            last_user_content = "" 
            if self.state != "no":
                last_user_content = f"<state>{self.history[-1].input_state}</state>"
            messages.append({"role": "user", "content": last_user_content})
        
        elif self.chat_format == "user_assistant_format_part":
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.instruction_prompt},
            ]
            start_index = max(0, len(self.history) - 1 - self.history_window_size)
            process_history = self.history[start_index:]
            if len(process_history) == 1:
                if self.state != "no":
                    messages[1]["content"] += f"\ncurrent state:<state>{process_history[0].input_state}</state>"
                return messages

            # Truncate history based on history_window_size
            # window_size controls the number of assistant replies to keep
            # The actual history length participating in prompt construction is window_size + 1 (including the current one to be generated)
            
            # First user-assistant conversation turn
            first_memory = process_history[0]
            if self.state != "no":
                 messages[1]["content"] += f"\n<state>{first_memory.input_state}</state>"
            
            agent_msg = ""
            if self.history_has_cot:
                agent_msg += f"<analysis> {first_memory.analysis} </analysis>\n"
            agent_msg += f"<action> {first_memory.action} </action>\n"
            messages.append({"role": "assistant", "content": agent_msg})

            # Intermediate history records
            for memory in process_history[1:-1]:
                user_content = ""
                if self.state != "no":
                    user_content = f"<state>{memory.input_state}</state>"
                messages.append({"role": "user", "content": user_content})
                
                agent_msg = ""
                if self.history_has_cot:
                    agent_msg += f"<analysis> {memory.analysis} </analysis>\n"
                agent_msg += f"<action> {memory.action} </action>\n"
                messages.append({"role": "assistant", "content": agent_msg})

            # Last user input
            last_user_content = ""
            if self.state != "no":
                last_user_content = f"current state:<state>{process_history[-1].input_state}</state>"
            messages.append({"role": "user", "content": last_user_content})
            
        return messages
    
    def check_loop(self) -> Dict[str, Any]:
        """check if the agent the loop step in history actions"""
        general_loop_log = []
        specific_loop_log = []

        def _normalize(s):
            if s is None:
                return ""
            if not isinstance(s, str):
                try:
                    s = str(s)
                except Exception:
                    return ""
            return " ".join(s.strip().split()).lower()

        for step_idx in range(1, len(self.history)):
            cur_memory = self.history[step_idx]
            prev_memory = self.history[step_idx - 1]

            cur_action = _normalize(cur_memory.action)
            prev_action = _normalize(prev_memory.action)
            cur_obs = _normalize(cur_memory.observation)
            prev_obs = _normalize(prev_memory.observation)

            if (
                cur_action == prev_action
                and cur_obs == prev_obs
                and cur_action != "do_nothing"
            ):
                specific_loop_log.append(
                    {
                        "step": step_idx,
                        "action": cur_action,
                    }
                )
        return specific_loop_log

    def internal_modeling_prompt(self) -> str:
        """Generate internal modeling prompt based on history."""
        messages = [
                {"role": "system", "content": self.system_prompt},
            ]
        user_msg = self.instruction_prompt + "\n# History Steps:\n"
        if self.chat_format == "default_format" or self.chat_format == "user_assistant_format":
            for memory in self.history[:-1]:
                user_msg += f"<step>\n"
                if self.state != "no":
                    user_msg += f"\t<state>{memory.input_state}</state>\n"
                if self.history_has_cot:
                    user_msg += f"\t<analysis>{memory.analysis}</analysis>\n"
                user_msg += f"\t<action>{memory.action}</action>\n</step>\n"

        user_msg += "# Based on the above history actions, predict the current state.\n You should follow the format:\n<state>...</state>\n"
        messages.append({"role": "user", "content": user_msg})
        prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                enable_thinking=self.enable_thinking,
                add_generation_prompt=True,
        )
        return prompt

    def internal_modeling_prompt(self) -> str:
        """Generate internal modeling prompt based on history."""
        messages = self.internal_modeling_messages()
        prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                enable_thinking=self.enable_thinking,
                add_generation_prompt=True,
        )
        return prompt

    def internal_modeling_messages(self) -> List[Dict[str, str]]:
        """Generate internal modeling messages based on history."""
        messages = [
                {"role": "system", "content": self.system_prompt},
            ]
        user_msg = self.instruction_prompt + "\n# History Steps:\n"
        
        history_to_process = self.history[:-1]
        if self.chat_format == "user_assistant_format_part":
            # Also follows history_window_size logic, but applies to self.history[:-1]
            start_index = max(0, len(history_to_process) - self.history_window_size)
            history_to_process = history_to_process[start_index:]

        for memory in history_to_process:
            user_msg += f"<step>\n"
            if self.state != "no":
                user_msg += f"\t<state>{memory.input_state}</state>\n"
            if self.history_has_cot:
                user_msg += f"\t<analysis>{memory.analysis}</analysis>\n"
            user_msg += f"\t<action>{memory.action}</action>\n</step>\n"

        user_msg += "# Based on the above history actions, predict the current state.\n You should follow the format:\n<state>...</state>\n"
        messages.append({"role": "user", "content": user_msg})
        return messages

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("xxx")

    # --- Test examples start ---
    class MockConfig:
        def __init__(self, chat_format, state, history_has_cot, enable_thinking, history_window_size):
            self.agent_proxy = self
            self.chat_format = chat_format
            self.state = state
            self.history_has_cot = history_has_cot
            self.enable_thinking = enable_thinking
            self.history_window_size = history_window_size

        def get(self, key, default):
            if key == "history_window_size":
                return self.history_window_size
            return default

    # 1. Initialize
    system_prompt = "You are a helpful assistant."
    instruction_prompt = "Complete the following task."
    
    # 2. Simulate history records
    history = [
        StepMemory(input_state="State 0", analysis="Analysis 0", action="Action 0"),
        StepMemory(input_state="State 1", analysis="Analysis 1", action="Action 1"),
        StepMemory(input_state="State 2", analysis="Analysis 2", action="Action 2"),
        StepMemory(input_state="State 3", analysis="Analysis 3", action="Action 3"),
    ]

    # 3. Test different history_window_size values
    window_sizes_to_test = [2, 1, 0]

    for size in window_sizes_to_test:
        print(f"--- Testing with history_window_size = {size} ---")
        config = MockConfig(
            chat_format="user_assistant_format_part",
            state="yes",
            history_has_cot=True,
            enable_thinking=True,
            history_window_size=size
        )
        ctx_manager = ContextManager(system_prompt, instruction_prompt, tokenizer, config)
        ctx_manager.history = history
        
        formatted_prompt = ctx_manager.format_prompt()
        print("Formatted Prompt:")
        print(formatted_prompt)
        print("-" * 50 + "\n")
        print("internal_modeling_prompt:")
        internal_prompt = ctx_manager.internal_modeling_prompt()
        print(internal_prompt)
        print("=" * 80 + "\n")
      