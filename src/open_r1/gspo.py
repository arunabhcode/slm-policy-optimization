#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM
import torch
import numpy as np
import copy
import os
import json
import transformers

from open_r1.utils.model_utils import get_tokenizer
from open_r1.rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
)


class GSPOTrainer:
    def __init__(
        self,
        config,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        introspect=None,
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.introspect = introspect
        if self.tokenizer is not None:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.use_vllm = getattr(config, "use_vllm", True)

        # Device assignment
        self.train_device = self.config.train_device
        self.vllm_device = self.config.vllm_device
        self.ref_device = self.config.ref_device

        REWARD_FUNCS_REGISTRY = {
            "accuracy": accuracy_reward,
            "format": format_reward,
            "reasoning_steps": reasoning_steps_reward,
            "cosine": get_cosine_scaled_reward(
                min_value_wrong=config.cosine_min_value_wrong,
                max_value_wrong=config.cosine_max_value_wrong,
                min_value_correct=config.cosine_min_value_correct,
                max_value_correct=config.cosine_max_value_correct,
                max_len=config.cosine_max_len,
            ),
            "repetition_penalty": get_repetition_penalty_reward(
                ngram_size=config.repetition_n_grams,
                max_penalty=config.repetition_max_penalty,
            ),
            "length": len_reward,
            "code": code_reward,
            "code_format": get_code_format_reward(language=config.code_language),
            "tag_count": tag_count_reward,
        }
        self.reward_funcs = [
            REWARD_FUNCS_REGISTRY[func] for func in config.reward_funcs
        ]
        self.reward_weights = (
            config.reward_weights
            if config.reward_weights
            else [1.0] * len(self.reward_funcs)
        )

        def custom_collate(features):
            batch = {}
            for key in features[0].keys():
                batch[key] = [f[key] for f in features]
            return batch

        if self.train_dataset is not None:
            self.dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.config.per_device_train_batch_size,
                shuffle=True,
                collate_fn=custom_collate,
            )

        # Initialize vLLM for generation FIRST
        if self.use_vllm:
            self.init_vllm()

        # Initialize policy model for training SECOND
        self.init_policy_model()

        # Initialize optimizer and scheduler
        self.init_optimizer()

        # Track training state
        self.global_step = 0
        self.epoch = 0

    def init_vllm(self):
        """Initialize vLLM engine for generation"""
        print(
            f"Initializing vLLM with model: {self.config.model_name_or_path} on {self.vllm_device}"
        )

        self.llm = LLM(
            model=self.config.model_name_or_path,
            revision=self.config.model_revision,
            trust_remote_code=self.config.trust_remote_code,
            dtype=self.config.torch_dtype,
            gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            max_model_len=self.config.vllm_max_model_len,
            enforce_eager=self.config.vllm_enforce_eager,
            tensor_parallel_size=self.config.vllm_tensor_parallel_size,
        )

    def init_policy_model(self):
        """Initialize the policy model for training, ref model for KL"""
        print(f"Loading policy model and ref model on {self.train_device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            revision=self.config.model_revision,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            attn_implementation=self.config.attn_implementation,
        ).to(self.train_device)

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=self.config.gradient_checkpointing_kwargs
            )

        self.device = self.train_device
        self.model.train()

        # Reference model — place on ref device
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            revision=self.config.model_revision,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            attn_implementation=self.config.attn_implementation,
        ).to(self.ref_device)
        self.ref_model.eval()

    def init_optimizer(self):
        """Initialize optimizer and learning rate scheduler"""
        if self.train_dataset is not None:
            steps_per_epoch = (
                len(self.dataloader) // self.config.gradient_accumulation_steps
            )
            total_steps = min(
                self.config.max_steps, steps_per_epoch * self.config.num_train_epochs
            )

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

            num_warmup_steps = int(total_steps * self.config.warmup_ratio)
            scheduler_kwargs = (
                self.config.lr_scheduler_kwargs.copy()
                if self.config.lr_scheduler_kwargs
                else {}
            )

            if "min_lr_rate" in scheduler_kwargs:
                ratio = scheduler_kwargs.pop("min_lr_rate")
                scheduler_kwargs["min_lr"] = self.config.learning_rate * ratio

            self.scheduler = transformers.get_scheduler(
                name=self.config.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                scheduler_specific_kwargs=scheduler_kwargs,
            )

            print(
                f"Optimizer initialized: {total_steps} total steps, {num_warmup_steps} warmup steps"
            )

    def pytorch_to_vllm_weights(self):
        """Resync pytorch weights back to VLLM safely (assumes TP=1)."""
        with torch.no_grad():
            cpu_weights = [
                (name, param.data.cpu())
                for name, param in self.model.named_parameters()
            ]
            vllm_model = (
                self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            )
            vllm_model.load_weights(cpu_weights)
            del cpu_weights
            torch.cuda.empty_cache()

    def generate_with_model(self, prompts, num_generations):
        """Fallback generation without vLLM, now correctly extracting logprobs."""
        all_completions = []
        all_completion_ids = []
        all_prompt_ids = []
        all_prompts = []
        all_logprobs = []

        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True
            )
            if isinstance(p, (list, tuple))
            else p
            for p in prompts
        ]

        self.model.eval()

        for gen_idx in range(num_generations):
            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_completion_length,
                    temperature=self.config.temperature,
                    do_sample=True,
                    top_k=0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,  # Required for transition scores
                )

            # Compute actual sequence logprobs
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            generated_ids = outputs.sequences

            for i in range(len(formatted_prompts)):
                prompt_len = inputs.input_ids.shape[1]
                prompt_tokens = inputs.input_ids[i].tolist()
                comp_ids = generated_ids[i, prompt_len:].tolist()
                comp_scores = transition_scores[i].tolist()

                clean_ids = []
                clean_logprobs = []
                for tid, tscore in zip(comp_ids, comp_scores):
                    if (
                        tid == self.tokenizer.eos_token_id
                        or tid == self.tokenizer.pad_token_id
                    ):
                        break
                    clean_ids.append(tid)
                    clean_logprobs.append(tscore)

                completion_text = self.tokenizer.decode(
                    clean_ids, skip_special_tokens=False
                )

                all_completions.append(completion_text)
                all_completion_ids.append(tuple(clean_ids))
                all_prompt_ids.append(tuple(prompt_tokens))
                all_prompts.append(formatted_prompts[i])
                all_logprobs.append(clean_logprobs)

            del outputs, generated_ids, transition_scores, inputs
            torch.cuda.empty_cache()

        self.model.train()

        return {
            "completions": all_completions,
            "completion_ids": all_completion_ids,
            "prompt_ids": all_prompt_ids,
            "prompts": all_prompts,
            "logprobs": all_logprobs,
        }

    def generate_rollouts(self, prompts, num_generations):
        """Generate completions for the given prompts."""
        if self.use_vllm:
            return self._generate_rollouts_vllm(prompts, num_generations)
        else:
            return self.generate_with_model(prompts, num_generations)

    def _generate_rollouts_vllm(self, prompts, num_generations):
        """Generate completions using vLLM for the given prompts."""
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_completion_length,
            n=num_generations,
            logprobs=1,  # Request logprobs from vLLM
        )

        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True
            )
            if isinstance(p, (list, tuple))
            else p
            for p in prompts
        ]

        outputs = self.llm.generate(
            formatted_prompts, sampling_params=sampling_params, use_tqdm=False
        )

        all_completions = []
        all_completion_ids = []
        all_prompt_ids = []
        all_prompts = []
        all_logprobs = []

        for idx, output in enumerate(outputs):
            for completion_output in output.outputs:
                all_completions.append(completion_output.text)
                all_completion_ids.append(completion_output.token_ids)
                all_prompt_ids.append(output.prompt_token_ids)
                all_prompts.append(formatted_prompts[idx])

                # Extract the logprob for each generated token
                tok_logprobs = [
                    step_dict[tok_id].logprob if tok_id in step_dict else -100.0
                    for step_dict, tok_id in zip(
                        completion_output.logprobs, completion_output.token_ids
                    )
                ]
                all_logprobs.append(tok_logprobs)

        return {
            "completions": all_completions,
            "completion_ids": all_completion_ids,
            "prompt_ids": all_prompt_ids,
            "prompts": all_prompts,
            "logprobs": all_logprobs,
        }

    def compute_rewards(self, completions, solutions):
        """Compute rewards for each completion"""
        total_rewards = np.zeros(len(completions))
        formatted_completions = [
            [{"role": "assistant", "content": c}] for c in completions
        ]
        for func, weight in zip(self.reward_funcs, self.reward_weights):
            func_rewards = func(completions=formatted_completions, solution=solutions)
            total_rewards += weight * np.array(func_rewards)
        return total_rewards.tolist()

    def compute_advantages(self, rewards, num_generations):
        """Compute advantages using group normalization."""
        rewards = np.array(rewards).reshape(-1, num_generations)
        mean = rewards.mean(axis=1, keepdims=True)
        std = rewards.std(axis=1, keepdims=True) + 1e-8
        advantages = (rewards - mean) / std
        return advantages.flatten()

    def compute_policy_loss(self, rollouts, advantages):
        """Compute the policy gradient loss for GRPO securely and efficiently."""

        prompt_ids = [
            torch.tensor(ids, dtype=torch.long) for ids in rollouts["prompt_ids"]
        ]
        completion_ids = [
            torch.tensor(ids, dtype=torch.long) for ids in rollouts["completion_ids"]
        ]
        batch_size = len(completion_ids)

        max_len = max(
            (len(pids) + len(cids)) for pids, cids in zip(prompt_ids, completion_ids)
        )

        padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        valid_mask = torch.zeros(batch_size, max_len - 1, dtype=torch.long)
        seq_lengths = torch.zeros(batch_size, dtype=torch.float32)

        for i in range(batch_size):
            conversation = torch.cat([prompt_ids[i], completion_ids[i]])
            seq_len = len(conversation)
            prompt_len = len(prompt_ids[i])

            offset = max_len - seq_len
            padded_ids[i, offset:] = conversation
            attention_mask[i, offset:] = 1

            # 1 for completion tokens, 0 for prompt/padding
            valid_mask[i, offset + prompt_len - 1 : offset + seq_len - 1] = 1
            seq_lengths[i] = len(completion_ids[i])

        # Clamp seq lengths to avoid Division by Zero
        seq_lengths = torch.clamp(seq_lengths, min=1.0)

        output_device = self.device
        valid_mask = valid_mask.to(output_device)
        seq_lengths = seq_lengths.to(output_device)
        advantages_tensor = torch.tensor(
            advantages, dtype=torch.float32, device=output_device
        )

        # Micro-batching policy forward pass to avoid OOM
        micro_batch_size = max(1, batch_size // 2)
        log_probs_chunks = []

        for mb_start in range(0, batch_size, micro_batch_size):
            mb_end = min(mb_start + micro_batch_size, batch_size)
            mb_ids = padded_ids[mb_start:mb_end].to(self.device)
            mb_mask = attention_mask[mb_start:mb_end].to(self.device)

            logits = self.model(input_ids=mb_ids, attention_mask=mb_mask).logits
            mb_log_probs = self.get_logprobs(logits, mb_ids)
            log_probs_chunks.append(mb_log_probs)

            del logits, mb_ids, mb_mask
            torch.cuda.empty_cache()

        log_probs = torch.cat(log_probs_chunks, dim=0)

        # Micro-batching reference model
        ref_device = self.ref_device
        ref_padded_ids = padded_ids.detach().clone()
        ref_attention_mask = attention_mask.detach().clone()
        ref_log_probs_chunks = []

        with torch.no_grad():
            for mb_start in range(0, batch_size, micro_batch_size):
                mb_end = min(mb_start + micro_batch_size, batch_size)
                ref_mb_ids = ref_padded_ids[mb_start:mb_end].to(ref_device)
                ref_mb_mask = ref_attention_mask[mb_start:mb_end].to(ref_device)

                ref_mb_logits = self.ref_model(
                    input_ids=ref_mb_ids, attention_mask=ref_mb_mask
                ).logits
                ref_mb_log_probs = self.get_logprobs(ref_mb_logits, ref_mb_ids)
                ref_log_probs_chunks.append(ref_mb_log_probs.to(output_device))

                del ref_mb_logits, ref_mb_ids, ref_mb_mask
                torch.cuda.synchronize(ref_device)
                torch.cuda.empty_cache()

            ref_log_probs = torch.cat(ref_log_probs_chunks, dim=0)
            del ref_padded_ids, ref_attention_mask

        # Compute per-token KL before averaging
        per_token_kl = (
            torch.exp(ref_log_probs - log_probs.detach())
            - (ref_log_probs - log_probs.detach())
            - 1.0
        )
        seq_kl = (per_token_kl * valid_mask).sum(dim=1) / seq_lengths

        # Calculate current sequence log probabilities
        seq_log_probs = (log_probs * valid_mask).sum(dim=1) / seq_lengths

        # Enforce that rollouts MUST contain true generation logprobs
        if rollouts.get("logprobs") is None or rollouts["logprobs"][0] is None:
            raise ValueError(
                "Fatal: Generation logprobs are missing. PPO/GRPO clipping requires true old logprobs to function."
            )

        # Map the true generated logprobs to the exact same valid_mask window
        old_logprobs_list = [
            torch.tensor(lp, dtype=torch.float32) for lp in rollouts["logprobs"]
        ]
        padded_old_logprobs = torch.zeros(
            batch_size, max_len, dtype=torch.float32, device=output_device
        )

        for i in range(batch_size):
            seq_len = len(prompt_ids[i]) + len(completion_ids[i])
            prompt_len = len(prompt_ids[i])
            offset = max_len - seq_len

            padded_old_logprobs[i, offset + prompt_len - 1 : offset + seq_len - 1] = (
                old_logprobs_list[i].to(output_device)
            )

        old_seq_log_probs = (padded_old_logprobs * valid_mask).sum(dim=1) / seq_lengths

        # Calculate the true sequence ratio
        seq_ratio = torch.exp(seq_log_probs - old_seq_log_probs)

        surr1 = seq_ratio * advantages_tensor
        surr2 = (
            torch.clamp(seq_ratio, 1.0 - self.config.epsilon, 1.0 + self.config.epsilon)
            * advantages_tensor
        )
        policy_loss = -torch.min(surr1, surr2)

        # Combine losses
        combined_loss = (policy_loss + self.config.beta * seq_kl).mean()

        return combined_loss

    def train(self, resume_from_checkpoint=None):
        """Main training loop"""
        if not len(self.train_dataset):
            raise ValueError("train_dataset has length 0")

        if resume_from_checkpoint is not None:
            state_path = os.path.join(resume_from_checkpoint, "trainer_state.pt")
            if os.path.exists(state_path):
                print(f"Resuming from checkpoint: {resume_from_checkpoint}")
                state = torch.load(state_path, map_location=self.device)
                self.optimizer.load_state_dict(state["optimizer"])
                self.scheduler.load_state_dict(state["scheduler"])
                self.global_step = state["global_step"]
                self.epoch = state.get("epoch", 0)

                # Load state dict into the EXISTING model object to preserve optimizer hooks
                state_dict = AutoModelForCausalLM.from_pretrained(
                    resume_from_checkpoint,
                    torch_dtype=getattr(torch, self.config.torch_dtype),
                    attn_implementation=self.config.attn_implementation,
                ).state_dict()

                self.model.load_state_dict(state_dict)
                del state_dict
                torch.cuda.empty_cache()

                self.pytorch_to_vllm_weights()
                print(f"Resumed at step {self.global_step}")

        print(f"Starting training for {self.config.num_train_epochs} epochs")
        total_loss = 0.0

        # Respect the loaded epoch on resume
        for epoch in range(self.epoch, self.config.num_train_epochs):
            self.epoch = epoch
            print(f"Epoch {epoch+1}/{self.config.num_train_epochs}")

            epoch_iterator = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")

            for step, batch in enumerate(epoch_iterator):
                prompts = batch["prompt"]
                with torch.no_grad():
                    rollouts = self.generate_rollouts(
                        prompts, self.config.num_train_generations
                    )

                solutions = [
                    sol
                    for sol in batch["solution"]
                    for _ in range(self.config.num_train_generations)
                ]

                rewards = self.compute_rewards(rollouts["completions"], solutions)

                sample_prompt = prompts[0] if prompts else None
                sample_completion = (
                    rollouts["completions"][0][:] if rollouts["completions"] else None
                )
                sample_reward = rewards[0] if rewards else None

                advantages = self.compute_advantages(
                    rewards, self.config.num_train_generations
                )

                loss = self.compute_policy_loss(rollouts, advantages)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                total_loss += loss.item()

                del loss, rollouts
                torch.cuda.empty_cache()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.pytorch_to_vllm_weights()
                    self.global_step += 1

                    if self.global_step % self.config.logging_steps == 0:
                        # Fixed loss average logging math
                        avg_loss = total_loss / self.config.logging_steps
                        avg_reward = np.mean(rewards)
                        lr = self.scheduler.get_last_lr()[0]

                        epoch_iterator.set_postfix(
                            {
                                "loss": f"{avg_loss:.4f}",
                                "reward": f"{avg_reward:.4f}",
                                "lr": f"{lr:.2e}",
                            }
                        )

                        if self.introspect is not None:
                            self.introspect.log_scalar_dict(
                                {
                                    "train/loss": avg_loss,
                                    "train/reward_mean": avg_reward,
                                    "train/learning_rate": lr,
                                    "train/epoch": epoch,
                                    "train/global_step": self.global_step,
                                }
                            )

                        total_loss = 0.0

                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()

                if self.global_step >= self.config.max_steps:
                    break
            if self.global_step >= self.config.max_steps:
                break

        metrics = {"train_loss": total_loss, "final_step": self.global_step}
        return type("TrainOutput", (), {"metrics": metrics})()

    def save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(
            self.config.output_dir, f"checkpoint-{self.global_step}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"\nSaving checkpoint to {checkpoint_dir}")
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "epoch": self.epoch,
            },
            os.path.join(checkpoint_dir, "trainer_state.pt"),
        )

    def evaluate(self):
        """Evaluate the model"""
        if self.eval_dataset is None:
            return {
                "eval/reward_mean": 0.0,
                "eval/reward_std": 0.0,
                "eval/reward_min": 0.0,
                "eval/reward_max": 0.0,
                "eval/num_samples": 0,
            }

        def custom_collate(features):
            batch = {}
            for key in features[0].keys():
                batch[key] = [f[key] for f in features]
            return batch

        eval_dataloader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.config.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=custom_collate,
        )

        all_rewards = []
        all_completions = []
        all_prompts = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                prompts = batch["prompt"]
                rollouts = self.generate_rollouts(
                    prompts, self.config.num_eval_generations
                )
                solutions = [
                    sol
                    for sol in batch["solution"]
                    for _ in range(self.config.num_eval_generations)
                ]
                rewards = self.compute_rewards(rollouts["completions"], solutions)

                all_rewards.extend(rewards)
                all_completions.extend(rollouts["completions"])
                all_prompts.extend(rollouts["prompts"])

        metrics = {
            "eval/reward_mean": float(np.mean(all_rewards)),
            "eval/reward_std": float(np.std(all_rewards)),
            "eval/reward_min": float(np.min(all_rewards)),
            "eval/reward_max": float(np.max(all_rewards)),
            "eval/num_samples": len(all_rewards),
        }

        if self.introspect is not None:
            self.introspect.log_scalar_dict(metrics)
            self.introspect.log_completions_table(
                prompts=all_prompts, completions=all_completions, rewards=all_rewards
            )

        return metrics

    def save_model(self):
        """Save the final trained model"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        print(f"Saving final model to {self.config.output_dir}")
        self.model.save_pretrained(self.config.output_dir)
        config_dict = {
            k: v for k, v in self.config.__dict__.items() if not k.startswith("_")
        }
        with open(
            os.path.join(self.config.output_dir, "training_config.json"), "w"
        ) as f:
            json.dump(config_dict, f, indent=2, default=str)

    def log_metrics(self, split, metrics):
        print(f"[{split}] {metrics}")

    def save_metrics(self, split, metrics):
        pass

    def save_state(self):
        """Save full trainer state"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        state_path = os.path.join(self.config.output_dir, "trainer_state.pt")
        torch.save(
            {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            state_path,
        )
        print(f"Trainer state saved to {state_path}")

    def get_logprobs(self, logits, input_ids):
        """Compute per-token log probabilities from logits."""
        logits = logits[:, :-1, :]  # (B, T-1, V)
        target_ids = input_ids[:, 1:]  # (B, T-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        per_token_logps = log_probs.gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        return per_token_logps
