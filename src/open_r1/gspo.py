#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM
import torch
import numpy as np
import copy

from open_r1.utils.model_utils import get_tokenizer


class GSPOTrainer:
    def __init__(
        self,
        config,
        train_dataset=None,
        tokenizer=None,
        reward_funcs=None,
        reward_weights=None,
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs
        self.reward_weights = (
            reward_weights if reward_weights else [1.0] * len(reward_funcs)
        )

        self.dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
        )

        # Initialize vLLM for generation
        self.init_vllm()

        # Initialize policy model for training
        self.init_policy_model()

        # Initialize optimizer and scheduler
        self.init_optimizer()

        # Track training state
        self.global_step = 0
        self.epoch = 0

    def init_vllm(self):
        """Initialize vLLM engine for generation"""
        print(f"Initializing vLLM with model: {self.config.model_name_or_path}")

        self.llm = LLM(
            model=self.config.model_name_or_path,
            revision=self.config.model_revision,
            trust_remote_code=self.config.trust_remote_code,
            dtype=self.config.torch_dtype,
            gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            max_model_len=self.config.vllm_max_model_len,
            enforce_eager=self.config.vllm_enforce_eager,
            # For colocated mode, use external launcher
            # distributed_executor_backend="external_launcher",
        )

        # Load tokenizer if not provided
        self.tokenizer = get_tokenizer(self.config)

    def init_policy_model(self):
        """Initialize the policy model for training"""
        print(f"Loading policy model: {self.config.model_name_or_path}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            revision=self.config.model_revision,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            attn_implementation=self.config.attn_implementation,
        )

        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=self.config.gradient_checkpointing_kwargs
            )

        # Move to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.train()
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()

    def init_optimizer(self):
        """Initialize optimizer and learning rate scheduler"""
        # Calculate total training steps
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
        self.scheduler = torch.optim.get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
            scheduler_specific_kwargs=self.config.lr_scheduler_kwargs,
        )

        print(
            f"Optimizer initialized: {total_steps} total steps, {num_warmup_steps} warmup steps"
        )

    def pytorch_to_vllm_weights(self):
        pass

    def generate_rollouts(self, prompts):
        """
        Generate completions using vLLM for the given prompts.
        """
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_completion_length,
            n=self.config.num_generations,  # Generate multiple completions per prompt
            logprobs=1,  # Return logprobs for importance sampling
        )

        if isinstance(prompts[0], list):
            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
                for prompt in prompts
            ]
        else:
            formatted_prompts = prompts

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

    def compute_rewards(self, rollouts, completions):
        total_rewards = np.zeros(len(completions))
        total_rewards = torch.zeros(len(completions))

        for func, weight in zip(self.reward_funcs, self.reward_weights):
            # Call reward function with appropriate kwargs
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

        old_seq_log_probs_list = []
        ref_seq_log_probs_list = []
        valid_masks_list = []
        inputs_list = []

        prompt_ids = self.tokenizer(
            rollouts["prompts"],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        )

        completion_ids = [
            torch.tensor(ids, dtype=torch.long) for ids in rollouts["completion_ids"]
        ]

        # Pad sequences to same length
        max_len = max(
            (len(pids) + len(cids)) for pids, cids in zip(prompt_ids, completion_ids)
        )

        padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        action_mask = torch.zeros(
            batch_size, max_len - 1, dtype=torch.long
        )  # -1 for shifted logits
        seq_lengths = torch.zeros(batch_size, dtype=torch.float32)

        for i in range(batch_size):
            # Concatenate prompt and completion
            conversation = torch.cat([prompt_ids[i], completion_ids[i]])
            seq_len = len(conversation)
            prompt_len = len(prompt_ids[i])

            padded_ids[i, :seq_len] = conversation
            attention_mask[i, :seq_len] = 1

            # Action mask: 1 for completion tokens, 0 for prompt and padding
            action_mask[i, prompt_len - 1 : seq_len - 1] = 1
            seq_lengths[i] = len(
                completion_ids[i]
            )  # Length of the generated completion

        # Move tensors to device
        padded_ids = padded_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        action_mask = action_mask.to(self.device)
        seq_lengths = seq_lengths.to(self.device)
        advantages_tensor = torch.tensor(
            advantages, dtype=torch.float32, device=self.device
        )

        # 1. Get current policy log probabilities
        logits = self.model(input_ids=padded_ids, attention_mask=attention_mask).logits
        log_probs = self._get_per_token_logps(logits, padded_ids)

        # 2. Get reference policy log probabilities
        with torch.no_grad():
            ref_logits = self.ref_model(
                input_ids=padded_ids, attention_mask=attention_mask
            ).logits
            ref_log_probs = self._get_per_token_logps(ref_logits, padded_ids)

        # 3. Calculate Sequence-level log probabilities
        # Sum log probs over completion tokens only, then divide by completion length
        seq_log_probs = (log_probs * action_mask).sum(dim=1) / seq_lengths
        ref_seq_log_probs = (ref_log_probs * action_mask).sum(dim=1) / seq_lengths

        # For a single forward pass per rollout, old_seq_log_probs is the detached current seq_log_probs
        old_seq_log_probs = seq_log_probs.detach()

        # 4. Calculate GSPO Surrogate Loss (Sequence level)
        seq_ratio = torch.exp(seq_log_probs - old_seq_log_probs)

        surr1 = seq_ratio * advantages_tensor
        surr2 = torch.clamp(seq_ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2)

        # 5. Calculate Sequence-level KL divergence penalty
        seq_kl = (
            torch.exp(ref_seq_log_probs - seq_log_probs)
            - (ref_seq_log_probs - seq_log_probs)
            - 1.0
        )

        # 6. Combine losses and average over the batch
        combined_loss = (policy_loss + beta * seq_kl).mean()

        return combined_loss

    def train(self, resume_from_checkpoint=None):
        """
        Main training loop: rollout -> rewards -> advantages -> policy update
        """
        if not len(self.train_dataset):
            raise ValueError("train_dataset has length 0")

        print(f"Starting training for {self.config.num_train_epochs} epochs")
        print(f"Max steps: {self.config.max_steps}")
        print(f"Batch size: {self.config.per_device_train_batch_size}")
        print(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        print(f"Generations per prompt: {self.config.num_generations}")

        total_loss = 0.0

        for epoch in range(self.config.num_train_epochs):
            print(f"Epoch {epoch+1}/{self.config.num_train_epochs}")

            epoch_iterator = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")

            for step, batch in enumerate(epoch_iterator):
                # 1. Generate rollouts using vLLM
                prompts = batch["prompt"]
                with torch.no_grad():
                    rollouts = self.generate_rollouts(prompts)

                # 2. Compute rewards
                rewards = self.compute_rewards(
                    rollouts["prompts"], rollouts["completions"]
                )

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

                        if (
                            self.config.log_completions
                            and self.global_step % (self.config.logging_steps * 10) == 0
                        ):
                            print(f"\n--- Sample completion ---")
                            print(f"Prompt: {prompts[0]}")
                            print(f"Completion: {rollouts['completions'][0][:200]}...")
                            print(f"Reward: {rewards[0]:.4f}")

                        total_loss = 0.0

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()

                if self.global_step >= self.config.max_steps:
                    break

        metrics = {
            "train_loss": total_loss,
            "final_step": self.global_step,
        }

        return type("TrainOutput", (), {"metrics": metrics})()

    def save_checkpoint(self):
        """Save model checkpoint"""
        # checkpoint_dir = os.path.join(
        #     self.config.output_dir, f"checkpoint-{self.global_step}"
        # )
        # os.makedirs(checkpoint_dir, exist_ok=True)

        # print(f"\nSaving checkpoint to {checkpoint_dir}")
        # self.model.save_pretrained(checkpoint_dir)
        # self.tokenizer.save_pretrained(checkpoint_dir)

        # # Save optimizer and scheduler state
        # torch.save(
        #     {
        #         "optimizer": self.optimizer.state_dict(),
        #         "scheduler": self.scheduler.state_dict(),
        #         "global_step": self.global_step,
        #         "epoch": self.epoch,
        #     },
        #     os.path.join(checkpoint_dir, "trainer_state.pt"),
        # )

    def evaluate(self):
        # TODO
        return {}

    def save_model(self, output_dir=None):
        """Save the final trained model"""
        save_dir = output_dir or self.config.output_dir
        os.makedirs(save_dir, exist_ok=True)

        print(f"Saving final model to {save_dir}")
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # Save config
        import json

        config_dict = {
            k: v for k, v in self.config.__dict__.items() if not k.startswith("_")
        }
        with open(os.path.join(save_dir, "training_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    def log_metrics(self, split, metrics):
        print(f"[{split}] {metrics}")

    def save_metrics(self, split, metrics):
        pass

    def save_state(self):
        pass
