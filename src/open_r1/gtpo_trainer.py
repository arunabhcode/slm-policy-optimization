import os
import textwrap
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable, Optional, Sized, Union

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed, pad_across_processes
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl import GRPOTrainer
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_decorator
from trl.import_utils import is_rich_available, is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class GTPOTrainer(GRPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def entropy_from_logits(self, logits: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
        """
        Compute the Shannon entropy (in nats) for each row of *logits* in a memory-efficient way.

        Instead of materializing the full softmax for all rows at once, the logits are flattened to shape (N, num_classes),
        where N is the product of all leading dimensions. Computation is then performed in chunks of size `chunk_size`
        along this flattened dimension, reducing peak memory usage. The result is reshaped back to match the input's
        leading dimensions.

        Args:
            logits (`torch.Tensor`):
                Logits tensor of shape `(..., num_classes)`. Entropy is taken along the last axis; all leading dimensions
                are preserved in the output.
            chunk_size (`int`, *optional*, defaults to `128`):
                Number of rows from the flattened logits to process per iteration. Smaller values reduce memory usage at
                the cost of more iterations.

        Returns:
            `torch.Tensor`:
                Entropy values with shape `logits.shape[:-1]`.
        """
        original_shape = logits.shape[:-1]  # all dims except num_classes
        num_classes = logits.shape[-1]

        # Flatten all leading dimensions into one
        flat_logits = logits.reshape(-1, num_classes)

        entropies = []
        for chunk in flat_logits.split(chunk_size, dim=0):
            logps = F.log_softmax(chunk, dim=-1)
            chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
            entropies.append(chunk_entropy)

        entropies = torch.cat(entropies, dim=0)
        return entropies.reshape(original_shape)


    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text))
                all_outputs = self.llm.generate(
                    ordered_set_of_prompts, sampling_params=self.sampling_params, use_tqdm=False
                )
                completion_ids = []
                for outputs in all_outputs:
                    for output in outputs.outputs:
                        completion_ids.append(output.token_ids)
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            logits = self.model(prompt_completion_ids, attention_mask=attention_mask).logits
            
            # Shift for next-token prediction
            shifted_logits = logits[:, :-1, :]
            shifted_labels = prompt_completion_ids[:, 1:]
            
            # Extract only the completion token logits for entropy
            completion_logits = shifted_logits[:, -logits_to_keep:, :]
            entropies = self.entropy_from_logits(completion_logits)

            # Get standard logps (if needed for the first iteration)
            if self.num_iterations > 1:
                per_token_logps = selective_log_softmax(shifted_logits, shifted_labels)
                old_per_token_logps = per_token_logps[:, -logits_to_keep:]
            else:
                old_per_token_logps = None

            # Get reference logps (if KL penalty is enabled)
            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super(GRPOTrainer, self)._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        
        entropies_padded = pad_across_processes(entropies, dim=1, pad_index=0.0)
        completion_mask_padded = pad_across_processes(completion_mask, dim=1, pad_index=0)
        global_entropies = gather(entropies_padded)
        global_mask = gather(completion_mask_padded)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)


        advantages = self._get_gtpo_advantages(
            rewards=rewards,
            entropies=global_entropies,
            mask=global_mask
        )

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        reward_per_func = rewards_per_func.mean(0)

        
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()
            rewards_per_func_list = rewards_per_func.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards_to_log,
                    }
                    for i, reward_func in enumerate(self.reward_funcs):
                        if isinstance(reward_func, nn.Module):
                            reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                        else:
                            reward_func_name = reward_func.__name__
                            
                        # Extract the i-th column from our rewards list
                        table[f"reward_{reward_func_name}"] = [row[i] for row in rewards_per_func_list]
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
    

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        log_ratio = per_token_logps - old_per_token_logps
        log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        log_importance_weights = log_importance_weights.unsqueeze(-1)
        coef_1 = torch.exp(log_importance_weights)

        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss

    def _get_gtpo_advantages(self, rewards, entropies, mask, success_threshold=0.5):
        """
        Calculates detached advantages for GTPO.
        """
        with torch.no_grad():
            grouped_rewards = rewards.view(-1, self.num_generations) # (B, G)
            grouped_entropies = entropies.view(-1, self.num_generations, entropies.size(-1)) # (B, G, T)
            grouped_mask = mask.view(-1, self.num_generations, mask.size(-1)) # (B, G, T)

            # 1. Safely compute mean entropy ignoring padding tokens
            masked_entropies = grouped_entropies * grouped_mask
            valid_token_counts = grouped_mask.sum(dim=2).clamp(min=1.0)
            grouped_mean_entropies = masked_entropies.sum(dim=2) / valid_token_counts # (B, G)
            
            success_mask = (grouped_rewards > success_threshold).float() # (B, G)
            safe_mask_val = -10000.0

            pos_logits = grouped_mean_entropies.masked_fill(success_mask == 0, safe_mask_val)
            pos_weights = torch.softmax(pos_logits, dim=1) # (B, G)

            neg_logits = (1.0 / grouped_mean_entropies.clamp(min=1e-3)).masked_fill(success_mask == 1, safe_mask_val)
            neg_weights = torch.softmax(neg_logits, dim=1) # (B, G)

            positive_rewards = 0.5 * grouped_rewards + 0.5 * pos_weights * success_mask.sum(dim=1, keepdim=True)
            negative_rewards = -0.5 + 0.5 * neg_weights * (1 - success_mask).sum(dim=1, keepdim=True)

            # 2. Bulletproof variance calculations to prevent sqrt(negative) edge cases
            pos_mean = positive_rewards.mean(dim=1, keepdim=True) # (B, 1)
            pos_var = positive_rewards.var(dim=1, keepdim=True, unbiased=False) # (B, 1)
            pos_std = torch.sqrt(torch.clamp(pos_var, min=0.0) + 1e-8) # (B, 1)
            positive_advantages = (positive_rewards - pos_mean) / (pos_std + 1e-4) # (B, G)

            neg_mean = negative_rewards.mean(dim=1, keepdim=True) # (B, 1)
            neg_var = negative_rewards.var(dim=1, keepdim=True, unbiased=False) # (B, 1)
            neg_std = torch.sqrt(torch.clamp(neg_var, min=0.0) + 1e-8) # (B, 1)
            negative_advantages = (negative_rewards - neg_mean) / (neg_std + 1e-4) # (B, G)

            gtpo_advantages = (success_mask * positive_advantages) + ((1 - success_mask) * negative_advantages) # (B, G)
            
            # flatten to align with sequence-level coef_1
            gtpo_advantages = gtpo_advantages.view(-1, 1) # (B*G, 1)

        return gtpo_advantages