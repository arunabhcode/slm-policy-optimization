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
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.use_vllm = getattr(config, "use_vllm", True)

        # Device assignment
        if self.use_vllm:
            self.vllm_device = "cuda:0"
            self.train_device = "cuda:1"
        else:
            self.train_device = "cuda:0"
            self.vllm_device = "cuda:0"

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
        self.reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in config.reward_funcs]
        self.reward_weights = (
            config.reward_weights if config.reward_weights else [1.0] * len(self.reward_funcs)
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

        # Initialize vLLM for generation FIRST (on GPU 0)
        if self.use_vllm:
            self.init_vllm()

        # Initialize policy model for training SECOND (on GPU 1)
        self.init_policy_model()

        # Initialize optimizer and scheduler
        self.init_optimizer()

        # Track training state
        self.global_step = 0
        self.epoch = 0

    def init_vllm(self):
        """Initialize vLLM engine for generation on GPU 0"""
        print(f"Initializing vLLM with model: {self.config.model_name_or_path} on {self.vllm_device}")

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
        if self.use_vllm:
            print(f"Loading policy model and ref model on {self.train_device}")
        else:
            print(f"Loading policy model and ref model on {self.train_device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            revision=self.config.model_revision,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            attn_implementation=self.config.attn_implementation,
        ).to(self.train_device)

        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=self.config.gradient_checkpointing_kwargs
            )

        self.device = self.train_device
        self.model.train()

        # Reference model — place on training device to avoid conflicts with vLLM on GPU 0
        ref_device = self.train_device
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            revision=self.config.model_revision,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            attn_implementation=self.config.attn_implementation,
        ).to(ref_device)
        self.ref_model.eval()

    def init_optimizer(self):
        """Initialize optimizer and learning rate scheduler"""
        # Calculate total training steps
        if self.train_dataset is not None:
            steps_per_epoch = (
                len(self.dataloader) // self.config.gradient_accumulation_steps
            )
            total_steps = min(
                self.config.max_steps, steps_per_epoch * self.config.num_train_epochs
            )

            # Initialize optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

            # Initialize scheduler
            num_warmup_steps = int(total_steps * self.config.warmup_ratio)
            scheduler_kwargs = self.config.lr_scheduler_kwargs.copy() if self.config.lr_scheduler_kwargs else {}
            
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
        """Resync pytorch weights back to VLLM using the V0 engine's model_executor."""
        with torch.no_grad():
            # 1. Extract weights to the CPU bridge
            cpu_weights = [(name, param.data.cpu()) for name, param in self.model.named_parameters()]

            # 2. Access the newly unhidden V0 model executor
            vllm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            vllm_model.load_weights(cpu_weights)

            # 3. Cleanup
            del cpu_weights
            torch.cuda.empty_cache()

    def generate_with_model(self, prompts):
        """
        Generate completions using HuggingFace model.generate() instead of vLLM.
        """
        all_completions = []
        all_completion_ids = []
        all_logprobs = []
        all_prompts = []

        if isinstance(prompts[0], (list, tuple)):
            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
                for prompt in prompts
            ]
        else:
            formatted_prompts = list(prompts)

        self.model.eval()

        for gen_idx in range(self.config.num_generations):
            # Tokenize all prompts together (left-padded)
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
                )

            generated_ids = outputs

            for i in range(len(formatted_prompts)):
                comp_ids = generated_ids[i, inputs.input_ids.shape[1]:].tolist()

                # Strip trailing pad/eos tokens
                clean_ids = []
                for tid in comp_ids:
                    if tid == self.tokenizer.eos_token_id or tid == self.tokenizer.pad_token_id:
                        break
                    clean_ids.append(tid)

                completion_text = self.tokenizer.decode(clean_ids, skip_special_tokens=False)

                all_completions.append(completion_text)
                all_completion_ids.append(tuple(clean_ids))
                all_logprobs.append(None)  # Logprobs recomputed in compute_policy_loss
                all_prompts.append(formatted_prompts[i])

            del outputs, generated_ids, inputs
            torch.cuda.empty_cache()

        self.model.train()

        return {
            "completions": all_completions,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "prompts": all_prompts,
        }

    def generate_rollouts(self, prompts):
        """
        Generate completions for the given prompts.
        """
        if self.use_vllm:
            return self._generate_rollouts_vllm(prompts)
        else:
            return self.generate_with_model(prompts)

    def _generate_rollouts_vllm(self, prompts):
        """
        Generate completions using vLLM for the given prompts.
        """
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_completion_length,
            n=self.config.num_generations,
            logprobs=1,
        )

        if isinstance(prompts[0], (list, tuple)):
            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
                for prompt in prompts
            ]
        else:
            formatted_prompts = list(prompts)

        outputs = self.llm.generate(
            formatted_prompts, sampling_params=sampling_params, use_tqdm=False
        )

        # Extract results
        all_completions = []
        all_completion_ids = []
        all_logprobs = []
        all_prompts = []

        for idx, output in enumerate(outputs):
            for completion_output in output.outputs:
                all_completions.append(completion_output.text)
                all_completion_ids.append(completion_output.token_ids)
                all_logprobs.append(completion_output.logprobs)
                all_prompts.append(formatted_prompts[idx])

        return {
            "completions": all_completions,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "prompts": all_prompts,
        }

    def compute_rewards(self, completions, solutions):
        """Compute rewards for each completion using the configured reward functions"""
        total_rewards = np.zeros(len(completions))
        formatted_completions = [
            [{"role": "assistant", "content": c}] for c in completions
        ]
        for func, weight in zip(self.reward_funcs, self.reward_weights):
            func_rewards = func(
                completions=formatted_completions,
                solution=solutions,
            )
            total_rewards += weight * np.array(func_rewards)

        return total_rewards.tolist()

    def compute_advantages(self, rewards, num_generations):
        """
        Compute advantages using group normalization.
        """
        rewards = np.array(rewards).reshape(-1, num_generations)
        mean = rewards.mean(axis=1, keepdims=True)
        std = rewards.std(axis=1, keepdims=True) + 1e-8
        advantages = (rewards - mean) / std
        return advantages.flatten()

    def compute_policy_loss(self, rollouts, advantages):
        """
        Compute the policy gradient loss for GRPO.
        """
        # Get completion token IDs and convert to tensors

        prompt_ids = [
            self.tokenizer(
                p, return_tensors="pt", padding=False, add_special_tokens=False
            ).input_ids.squeeze(0)
            for p in rollouts["prompts"]
        ]

        completion_ids = [
            torch.tensor(ids, dtype=torch.long) for ids in rollouts["completion_ids"]
        ]

        # Get batch size from the number of completions
        batch_size = len(completion_ids)

        # Pad sequences to same length
        max_len = max(
            (len(pids) + len(cids)) for pids, cids in zip(prompt_ids, completion_ids)
        )

        padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        valid_mask = torch.zeros(batch_size, max_len - 1, dtype=torch.long)
        seq_lengths = torch.zeros(batch_size, dtype=torch.float32)

        for i in range(batch_size):
            # Concatenate prompt and completion
            conversation = torch.cat([prompt_ids[i], completion_ids[i]])
            seq_len = len(conversation)
            prompt_len = len(prompt_ids[i])

            # Left-pad: place content at the end of the sequence
            offset = max_len - seq_len
            padded_ids[i, offset:] = conversation
            attention_mask[i, offset:] = 1

            # Valid mask: 1 for completion tokens, 0 for prompt and padding
            valid_mask[i, offset + prompt_len - 1 : offset + seq_len - 1] = 1
            seq_lengths[i] = len(
                completion_ids[i]
            )  # Length of the generated completion

        # Move input tensors to the training device
        padded_ids = padded_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Get current policy log probabilities
        logits = self.model(input_ids=padded_ids, attention_mask=attention_mask).logits

        output_device = logits.device
        valid_mask = valid_mask.to(output_device)
        seq_lengths = seq_lengths.to(output_device)
        padded_ids_out = padded_ids.to(output_device)
        advantages_tensor = torch.tensor(
            advantages, dtype=torch.float32, device=output_device
        )

        log_probs = self.get_logprobs(logits, padded_ids_out)
        del logits  # Free policy logits immediately
        torch.cuda.empty_cache()

        # Synchronize CUDA before ref model forward to prevent race conditions
        # with gradient checkpointing holding references to the same tensors
        torch.cuda.synchronize(self.device)

        # Get reference policy log probabilities in micro-batches to reduce peak memory
        # Clone tensors to avoid CUDA memory conflicts with gradient checkpointing
        ref_padded_ids = padded_ids.detach().clone()
        ref_attention_mask = attention_mask.detach().clone()

        ref_device = self.train_device
        ref_micro_batch_size = max(1, batch_size // 2)  # Process in 2 chunks
        ref_log_probs_chunks = []

        with torch.no_grad():
            for mb_start in range(0, batch_size, ref_micro_batch_size):
                mb_end = min(mb_start + ref_micro_batch_size, batch_size)
                ref_mb_ids = ref_padded_ids[mb_start:mb_end].to(ref_device)
                ref_mb_mask = ref_attention_mask[mb_start:mb_end].to(ref_device)

                ref_mb_logits = self.ref_model(
                    input_ids=ref_mb_ids, attention_mask=ref_mb_mask
                ).logits

                ref_mb_log_probs = self.get_logprobs(ref_mb_logits, ref_mb_ids)
                ref_log_probs_chunks.append(ref_mb_log_probs.to(output_device))

                del ref_mb_logits, ref_mb_ids, ref_mb_mask
                torch.cuda.empty_cache()

            ref_log_probs = torch.cat(ref_log_probs_chunks, dim=0)
            del ref_log_probs_chunks, ref_padded_ids, ref_attention_mask

        # Calculate Sequence-level log probabilities
        # Sum log probs over completion tokens only, then divide by completion length
        seq_log_probs = (log_probs * valid_mask).sum(dim=1) / seq_lengths
        ref_seq_log_probs = (ref_log_probs * valid_mask).sum(dim=1) / seq_lengths

        del ref_log_probs, log_probs  # Free these before loss computation

        # For a single forward pass per rollout, old_seq_log_probs is the detached current seq_log_probs
        old_seq_log_probs = seq_log_probs.detach()

        # Calculate GSPO Surrogate Loss (Sequence level)
        seq_ratio = torch.exp(seq_log_probs - old_seq_log_probs)

        surr1 = seq_ratio * advantages_tensor
        surr2 = (
            torch.clamp(seq_ratio, 1.0 - self.config.epsilon, 1.0 + self.config.epsilon)
            * advantages_tensor
        )
        policy_loss = -torch.min(surr1, surr2)

        # Calculate Sequence-level KL divergence penalty
        seq_kl = (
            torch.exp(ref_seq_log_probs - seq_log_probs)
            - (ref_seq_log_probs - seq_log_probs)
            - 1.0
        )

        # 6. Combine losses and average over the batch
        combined_loss = (policy_loss + self.config.beta * seq_kl).mean()

        return combined_loss

    def train(self, resume_from_checkpoint=None):
        """
        Main training loop: rollout -> rewards -> advantages -> policy update
        """
        if not len(self.train_dataset):
            raise ValueError("train_dataset has length 0")

        # Resume from checkpoint if specified
        if resume_from_checkpoint is not None:
            state_path = os.path.join(resume_from_checkpoint, "trainer_state.pt")
            if os.path.exists(state_path):
                print(f"Resuming from checkpoint: {resume_from_checkpoint}")
                state = torch.load(state_path, map_location=self.device)
                self.optimizer.load_state_dict(state["optimizer"])
                self.scheduler.load_state_dict(state["scheduler"])
                self.global_step = state["global_step"]
                self.epoch = state.get("epoch", 0)

                self.model = AutoModelForCausalLM.from_pretrained(
                    resume_from_checkpoint,
                    torch_dtype=getattr(torch, self.config.torch_dtype),
                    attn_implementation=self.config.attn_implementation,
                ).to(self.train_device)
                self.model.train()
                self.pytorch_to_vllm_weights()
                print(f"Resumed at step {self.global_step}")

        print(f"Starting training for {self.config.num_train_epochs} epochs")
        print(f"Max steps: {self.config.max_steps}")
        print(f"Batch size: {self.config.per_device_train_batch_size}")
        print(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        print(f"Generations per prompt: {self.config.num_generations}")

        total_loss = 0.0

        for epoch in range(self.config.num_train_epochs):
            self.epoch = epoch
            print(f"Epoch {epoch+1}/{self.config.num_train_epochs}")

            epoch_iterator = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")

            for step, batch in enumerate(epoch_iterator):
                # 1. Generate rollouts using vLLM
                prompts = batch["prompt"]
                with torch.no_grad():
                    rollouts = self.generate_rollouts(prompts)

                solutions = [
                    sol
                    for sol in batch["solution"]
                    for _ in range(self.config.num_generations)
                ]

                # 2. Compute rewards
                rewards = self.compute_rewards(rollouts["completions"], solutions)

                # 3. Calculate advantages using group normalization
                advantages = self.compute_advantages(
                    rewards, self.config.num_generations
                )

                # 4. Compute policy loss
                loss = self.compute_policy_loss(rollouts, advantages)

                # Scale loss by gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                total_loss += loss.item()

                # Free loss tensor and clear cache between steps
                del loss, rollouts
                torch.cuda.empty_cache()

                # 5. Update policy if we've accumulated enough gradients
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.pytorch_to_vllm_weights()

                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = total_loss / (
                            self.config.logging_steps
                            * self.config.gradient_accumulation_steps
                        )
                        avg_reward = np.mean(rewards)
                        lr = self.scheduler.get_last_lr()[0]

                        epoch_iterator.set_postfix(
                            {
                                "loss": f"{avg_loss:.4f}",
                                "reward": f"{avg_reward:.4f}",
                                "lr": f"{lr:.2e}",
                            }
                        )

                        if (
                            self.config.log_completions
                            and self.global_step % (self.config.logging_steps * 10) == 0
                        ):
                            print(f"Prompt: {prompts[0]}")
                            print(f"Completion: {rollouts['completions'][0][:200]}")
                            print(f"Reward: {rewards[0]:.4f}")

                        total_loss = 0.0

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()

                if self.global_step >= self.config.max_steps:
                    break

            if self.global_step >= self.config.max_steps:
                break

        metrics = {
            "train_loss": total_loss,
            "final_step": self.global_step,
        }

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

        # Save optimizer and scheduler state
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
        """Evaluate the model on eval_dataset by generating completions and computing rewards."""

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

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                prompts = batch["prompt"]
                rollouts = self.generate_rollouts(prompts)

                solutions = [
                    sol
                    for sol in batch["solution"]
                    for _ in range(self.config.num_generations)
                ]

                rewards = self.compute_rewards(rollouts["completions"], solutions)
                all_rewards.extend(rewards)
                all_completions.extend(rollouts["completions"])

        metrics = {
            "eval_reward_mean": float(np.mean(all_rewards)),
            "eval_reward_std": float(np.std(all_rewards)),
            "eval_reward_min": float(np.min(all_rewards)),
            "eval_reward_max": float(np.max(all_rewards)),
            "eval_num_samples": len(all_rewards),
        }

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
        """Save full trainer state (optimizer, scheduler, step) for checkpoint resumption."""
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
        """
        Compute per-token log probabilities from logits.

        Args:
            logits: (batch_size, seq_len, vocab_size) model output logits
            input_ids: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len - 1) per-token log probabilities
        """
        # Shift: logits[t] predicts input_ids[t+1]
        logits = logits[:, :-1, :]  # (B, T-1, V)
        target_ids = input_ids[:, 1:]  # (B, T-1)

        log_probs = torch.log_softmax(logits, dim=-1)
        # Gather the log prob of the actual token
        per_token_logps = log_probs.gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)

        return per_token_logps
