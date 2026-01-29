from utils.llm_agent.base_agent import BaseAgent
from utils.llm_agent.ctx_manager import ContextManager, StepMemory
from utils.TaskRunner.TaskRunner import TrajectoryInfo
from vllm import LLM, CompletionOutput, SamplingParams, RequestOutput
from typing import Dict, Any, List
from transformers import AutoTokenizer
import re
from bisect import bisect_right
from utils.func import get_logp_distribution, calculate_entropy
import numpy as np
import os


class VLLMAgent(BaseAgent):
    def __init__(self, config, aim_gpus: str = "0", llm: LLM = None) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = aim_gpus
        if llm is not None:
            self.llm = llm
        else:
            # Construct LLM parameters on demand
            llm_kwargs = dict(
                model=config.vllm_agent.model_path,
                tensor_parallel_size=config.vllm_agent.tp_size,
                pipeline_parallel_size=config.vllm_agent.pp_size,
                max_model_len=config.agent_proxy.max_model_len,
                gpu_memory_utilization=config.vllm_agent.max_gpu_mem_util,
                trust_remote_code=True,
                enforce_eager=config.vllm_agent.enforce_eager,
            )
            # Only set this parameter when max_logprobs > 0
            if getattr(config.vllm_agent, "max_logprobs", 0) > 0:
                llm_kwargs["max_logprobs"] = config.vllm_agent.max_logprobs

            self.llm = LLM(**llm_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.vllm_agent.model_path, trust_remote_code=True
        )

        # Construct SamplingParams parameters on demand
        sampling_kwargs = dict(
            n=config.agent_proxy.step_rollout_n,
            temperature=config.agent_proxy.temperature,
            top_p=config.agent_proxy.top_p,
            max_tokens=config.agent_proxy.max_step_len,
            include_stop_str_in_output=config.vllm_agent.include_stop_str_in_output,
        )
        # Only set logprobs when max_logprobs > 0
        if getattr(config.vllm_agent, "max_logprobs", 0) > 0:
            sampling_kwargs["logprobs"] = config.vllm_agent.max_logprobs

        self.sampling_param = SamplingParams(**sampling_kwargs)

        if config.agent_proxy.chat_format == "default_format":
            self.sampling_param.stop = [config.agent_proxy.stop]
        self.task: str = config.task
        self.use_vllm_tqdm: bool = config.vllm_agent.use_vllm_tqdm
        self.enable_thinking: bool = config.agent_proxy.enable_thinking
        self.max_logprobs: int = config.vllm_agent.max_logprobs
        self.chat_format: str = (
            config.agent_proxy.chat_format
        )  # default_format, user_assistant_format, user_history_format
        self.history_has_cot: bool = (
            config.agent_proxy.history_has_cot
        )  # whether load each step cot in prompt from history
        self.stop_on_error: bool = config.agent_proxy.stop_on_error
        self.step_rollout_n: int = config.agent_proxy.step_rollout_n

    def get_next_step(self, traj: TrajectoryInfo) -> Dict[str, Any]:

        prompt = traj.ctx_manager.format_prompt()
        outputs: List[RequestOutput] = self.llm.generate(
            prompts=prompt,
            sampling_params=self.sampling_param,
            use_tqdm=self.use_vllm_tqdm,
        )
        # step rollout_n
        rollout_steps = []
        if self.step_rollout_n > 1:
            for choice in outputs[0].outputs:
                step_info = self._parse_response(choice)
                rollout_steps.append(step_info)
            action_space_entropy = self._cal_action_space_entropy(rollout_steps)
            # TODO: if use tts ,can choose best one
            step_info = rollout_steps[0]
        else:
            step_info = self._parse_response(outputs[0].outputs[0])
            rollout_steps.append(step_info)
            action_space_entropy = 0

        return {
            "model_input": prompt,
            **step_info,
            "action_space_entropy": action_space_entropy,
            "rollout_steps": rollout_steps,
        }

    def _parse_response(self, output: CompletionOutput) -> Dict[str, Any]:
        analysis_pattern = r"<analysis>(.*?)</analysis>"
        action_pattern = r"<action>(.*?)</action>"
        full_text = output.text
        output_token_ids = output.token_ids
        pieces: List[str] = [
            self.tokenizer.decode(
                [tid],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for tid in output_token_ids
        ]
        if self.max_logprobs > 0:
            token_logp_dist = get_logp_distribution(
                output, max_logprobs=self.max_logprobs
            )
        generated_token_ids = output.token_ids

        # 1) First match think/analysis, then search for action after it, to avoid matching action inside think
        search_start_pos = 0
        if self.enable_thinking:
            think_pattern = r"<think>(.*?)</think>"
            think_match = re.search(think_pattern, full_text, re.DOTALL)
            if think_match:
                search_start_pos = think_match.end()

        analysis_match = re.search(
            analysis_pattern, full_text[search_start_pos:], re.DOTALL
        )

        analysis_text = "DO_NOTHING"
        action_text = "DO_NOTHING"
        analysis_span = None  # (start_char, end_char) for inner content
        action_span = (
            None  # (start_char, end_char) for inner content (absolute coordinates)
        )

        if analysis_match:
            analysis_text = analysis_match.group(1).strip()
            analysis_span = (
                search_start_pos + analysis_match.start(1),
                search_start_pos + analysis_match.end(1),
            )

            # Only search for action after think/analysis ends
            abs_analysis_end = search_start_pos + analysis_match.end()
            post_text = full_text[abs_analysis_end:]
            action_match_after = re.search(action_pattern, post_text, re.DOTALL)
            if action_match_after:
                action_text = action_match_after.group(1).strip()
                # Convert relative position to absolute character span
                abs_start = abs_analysis_end + action_match_after.start(1)
                abs_end = abs_analysis_end + action_match_after.end(1)
                action_span = (abs_start, abs_end)
        else:
            # No think, then globally search for first action
            action_match_any = re.search(
                action_pattern, full_text[search_start_pos:], re.DOTALL
            )
            if action_match_any:
                action_text = action_match_any.group(1).strip()
                action_span = (
                    search_start_pos + action_match_any.start(1),
                    search_start_pos + action_match_any.end(1),
                )

        # 2) Calculate token-level entropy (using approximate mapping from absolute character span to token)
        analysis_token_entropy = None
        action_token_entropy = None

        if analysis_span is not None and analysis_text:
            analysis_start, analysis_end = analysis_span
            analysis_start_token = len(
                self.tokenizer.encode(
                    full_text[:analysis_start], add_special_tokens=False
                )
            )
            analysis_end_token = len(
                self.tokenizer.encode(
                    full_text[:analysis_end], add_special_tokens=False
                )
            )
            if analysis_end_token <= len(generated_token_ids) and self.max_logprobs > 0:
                analysis_token_logprob_dist = token_logp_dist[
                    analysis_start_token:analysis_end_token
                ]
                analysis_token_entropy = calculate_entropy(
                    analysis_token_logprob_dist
                )

        if action_span is not None and action_text:
            action_start, action_end = action_span
            action_start_token = len(
                self.tokenizer.encode(
                    full_text[:action_start], add_special_tokens=False
                )
            )
            action_end_token = len(
                self.tokenizer.encode(
                    full_text[:action_end], add_special_tokens=False
                )
            )
            if action_end_token <= len(generated_token_ids) and self.max_logprobs > 0:
                action_token_logprob_dist = token_logp_dist[
                    action_start_token:action_end_token
                ]
                action_token_entropy = calculate_entropy(
                    action_token_logprob_dist
                )

        token_entropy_stats = {
            "action_stats": {
                "mean": (
                    np.mean(action_token_entropy)
                    if action_token_entropy
                    else None
                ),
                "var": (
                    np.var(action_token_entropy)
                    if action_token_entropy
                    else None
                ),
                "max": (
                    max(action_token_entropy) if action_token_entropy else None
                ),
                "min": (
                    min(action_token_entropy) if action_token_entropy else None
                ),
                "raw": action_token_entropy if action_token_entropy else None,
            },
            "analysis_stats": {
                "mean": (
                    np.mean(analysis_token_entropy)
                    if analysis_token_entropy
                    else None
                ),
                "var": (
                    np.var(analysis_token_entropy)
                    if analysis_token_entropy
                    else None
                ),
                "max": (
                    max(analysis_token_entropy)
                    if analysis_token_entropy
                    else None
                ),
                "min": (
                    min(analysis_token_entropy)
                    if analysis_token_entropy
                    else None
                ),
                "raw": (
                    analysis_token_entropy if analysis_token_entropy else None
                ),
            },
        }

        clean_analysis = (
            analysis_text.strip().lower()
            if isinstance(analysis_text, str)
            else ""
        )
        clean_action = (
            action_text.strip().lower() if isinstance(action_text, str) else ""
        )
        return {
            "analysis": clean_analysis or "do_nothing",
            "action": clean_action or "do_nothing",
            "response": output.text,
            "tokens": pieces,  # String fragment corresponding to each token
            "token_ids": output_token_ids,
            "token_entropy_stats": token_entropy_stats,
        }

    # def _parse_response_v2(self, output: CompletionOutput) -> Dict[str, Any]:
    #     analysis_pattern = r"<analysis>(.*?)</analysis>"
    #     action_pattern = r"<action>(.*?)</action>"

    #     output_token_ids = output.token_ids
    #     # Decode token by token precisely, avoid displacement caused by whitespace cleanup
    #     pieces: List[str] = [
    #         self.tokenizer.decode(
    #             [tid],
    #             skip_special_tokens=False,
    #             clean_up_tokenization_spaces=False,
    #         )
    #         for tid in output_token_ids
    #     ]
    #     text = "".join(pieces)

    #     # Build cumulative character end position array for fast char -> token mapping
    #     cum_ends: List[int] = []
    #     total = 0
    #     for p in pieces:
    #         total += len(p)
    #         cum_ends.append(total)

    #     def char_to_token_idx(pos: int) -> int:
    #         """
    #         Given character position pos (0-based), return token index i (0-based) where it belongs.
    #         If pos is in the fragment of the i-th token, return i.
    #         """
    #         return bisect_right(cum_ends, pos)

    #     def span_to_token_span(start: int, end: int) -> tuple[int, int]:
    #         """
    #         Convert character interval [start, end) to token interval [ti_start, ti_end)
    #         Return (ti, ti) for empty interval
    #         """
    #         if start == end:
    #             ti = (
    #                 char_to_token_idx(start)
    #                 if start < len(text)
    #                 else len(cum_ends)
    #             )
    #             return (ti, ti)
    #         ti_start = char_to_token_idx(start)
    #         ti_end = char_to_token_idx(end - 1) + 1
    #         return (ti_start, ti_end)

    #     def match_to_info(m: re.Match) -> Dict[str, Any]:
    #         tag_start, tag_end = m.span(0)
    #         inner_start, inner_end = m.span(1)

    #         tag_tok_span = span_to_token_span(tag_start, tag_end)
    #         inner_tok_span = span_to_token_span(inner_start, inner_end)

    #         return {
    #             "text": m.group(1),
    #             "char_span": (tag_start, tag_end),
    #             "inner_char_span": (inner_start, inner_end),
    #             "token_span": tag_tok_span,  # Token range including tags [start, end)
    #             "inner_token_span": inner_tok_span,  # Token range for content only [start, end)
    #         }

    #     # Use same strategy as _parse_response: if think/analysis exists, only search for action after it
    #     analysis_re = re.compile(analysis_pattern, re.DOTALL | re.IGNORECASE)
    #     action_re = re.compile(action_pattern, re.DOTALL | re.IGNORECASE)

    #     analysis_m = analysis_re.search(text)
    #     if analysis_m:
    #         action_m = action_re.search(text, analysis_m.end())
    #     else:
    #         action_m = action_re.search(text)

    #     analysis_info = match_to_info(analysis_m) if analysis_m else None
    #     action_info = match_to_info(action_m) if action_m else None

    #     token_logp_dist = get_logp_distribution(
    #         output, max_logprobs=self.max_logprobs
    #     )
    #     if analysis_info:
    #         analysis_token_logp_dist = token_logp_dist[
    #             analysis_info["token_span"][0] : analysis_info["token_span"][1]
    #         ]
    #         analysis_token_entropy: List[float] = calculate_entropy(
    #             analysis_token_logp_dist
    #         )
    #     if action_info:
    #         action_token_logp_dist = token_logp_dist[
    #             action_info["token_span"][0] : action_info["token_span"][1]
    #         ]
    #         action_token_entropy: List[float] = calculate_entropy(
    #             action_token_logp_dist
    #         )

    #     token_logprob_stats = {
    #         "action_stats": {
    #             "mean": np.mean(action_token_entropy) if action_info else None,
    #             "var": np.var(action_token_entropy) if action_info else None,
    #             "max": max(action_token_entropy) if action_info else None,
    #             "min": min(action_token_entropy) if action_info else None,
    #             "raw": action_token_entropy if action_info else None,
    #         },
    #         "analysis_stats": {
    #             "mean": (
    #                 np.mean(analysis_token_entropy) if analysis_info else None
    #             ),
    #             "var": (
    #                 np.var(analysis_token_entropy) if analysis_info else None
    #             ),
    #             "max": max(analysis_token_entropy) if analysis_info else None,
    #             "min": min(analysis_token_entropy) if analysis_info else None,
    #             "raw": analysis_token_entropy if analysis_info else None,
    #         },
    #     }

    #     clean_analysis = (
    #         analysis_info["text"].strip() if analysis_info else "DO_NOTHING"
    #     )
    #     clean_action = (
    #         action_info["text"].strip() if action_info else "DO_NOTHING"
    #     )
    #     return {
    #         "response": output.text,
    #         "tokens": pieces,  # String fragment corresponding to each token
    #         "token_ids": output_token_ids,
    #         "analysis": clean_analysis,
    #         "action": clean_action,
    #         "token_entropy_stats": token_logprob_stats,
    #     }

    def _cal_action_space_entropy(
        self, rollout_steps: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate the entropy of the action space based on the rollout_steps.
        Args:
            rollout_steps (List[Dict[str, Any]]): The list of rollout steps containing action information.
        Returns:
            The entropy of the action space.
        """
        action_counts = {}
        for step in rollout_steps:
            if step["action"] in action_counts:
                action_counts[step["action"]] += 1
            else:
                action_counts[step["action"]] = 1
        total_actions = len(rollout_steps)
        action_probs = np.array(
            [np.log(count / total_actions) for count in action_counts.values()]
        )
        entropy = calculate_entropy(action_probs.reshape(1, -1))
        return entropy[0]

    def get_internal_state(self) -> str:
        """Get the internal state representation of the agent."""
        # For VLLM agent, we currently do not have a specific internal state representation.
        # This function can be customized based on the task requirements.
        raise NotImplementedError

    def get_next_step_parallel(
        self, trajectories: List[TrajectoryInfo]
    ) -> List[Dict[str, Any]]:
        # Create a prompt for each active trajectory
        # len(prompts) == len(self.parallel_history)
        prompts = [traj.ctx_manager.format_prompt() for traj in trajectories]

        # self.sampling_param.n is already set to self.step_rollout_n
        # vLLM will generate n independent completions for each prompt in the prompts list
        outputs: List[RequestOutput] = self.llm.generate(
            prompts=prompts,
            sampling_params=self.sampling_param,
            use_tqdm=self.use_vllm_tqdm,
        )

        # Length of outputs equals length of prompts
        # Each output object contains n completions (choices)
        batch_results = []
        for i, request_output in enumerate(outputs):
            rollout_steps = []
            # request_output.outputs contains n rollout results for a single prompt
            for choice in request_output.outputs:
                step_info = self._parse_response(choice)
                rollout_steps.append(step_info)
            # TODO : if use tts ,can choose best one
            step_info = rollout_steps[0]  # Select first one as main output
            action_space_entropy = self._cal_action_space_entropy(rollout_steps)

            batch_results.append(
                {
                    "model_input": prompts[i],
                    "action_space_entropy": action_space_entropy,
                    **step_info,
                    "rollout_steps": rollout_steps,
                }
            )

        return batch_results

    def close(self):
        del self.llm
