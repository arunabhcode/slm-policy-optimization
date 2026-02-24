#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch

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
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Initialize scheduler
        num_warmup_steps = int(total_steps * self.config.warmup_ratio)
        self.scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
            scheduler_specific_kwargs=self.config.lr_scheduler_kwargs,
        )

        print(
            f"Optimizer initialized: {total_steps} total steps, {num_warmup_steps} warmup steps"
        )

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

        # Apply chat template if prompts are conversations
        if isinstance(prompts[0], list):
            # Prompts are in conversation format
            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
                for prompt in prompts
            ]
        else:
            formatted_prompts = prompts

        # Generate with vLLM
        outputs = self.llm.generate(
            formatted_prompts, sampling_params=sampling_params, use_tqdm=False
        )

        # Extract results
        all_completions = []
        all_completion_ids = []
        all_logprobs = []

        for output in outputs:
            for completion_output in output.outputs:
                all_completions.append(completion_output.text)
                all_completion_ids.append(completion_output.token_ids)
                all_logprobs.append(completion_output.logprobs)

        return {
            "completions": all_completions,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "prompts": formatted_prompts,
        }

    def compute_rewards(self, rollouts, completions):
        total_rewards = np.zeros(len(formatted_completions))
        total_rewards = torch.zeros(len(completions))

        for func, weight in zip(self.reward_funcs, self.reward_weights):
            # Call reward function with appropriate kwargs
            func_rewards = func(
                completions=formatted_completions,
                solution=solutions,
            )
            total_rewards += weight * np.array(func_rewards)

        return total_rewards.tolist()

        """Run all reward functions and sum their scores."""
        for func in self.reward_funcs:
            # Assuming reward function returns a list/tensor of floats
            scores = func(prompts, completions)
            total_rewards += torch.tensor(scores, dtype=torch.float32)
        return total_rewards

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
        completion_ids = [
            torch.tensor(ids, dtype=torch.long) for ids in rollouts["completion_ids"]
        ]

        # Pad sequences to same length
        max_len = max(len(ids) for ids in completion_ids)
        padded_ids = torch.zeros(len(completion_ids), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(completion_ids), max_len, dtype=torch.long)

        for i, ids in enumerate(completion_ids):
            padded_ids[i, : len(ids)] = ids
            attention_mask[i, : len(ids)] = 1

        # Move to device
        padded_ids = padded_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(
            self.device
        )

        # Forward pass through policy model
        outputs = self.model(input_ids=padded_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # Shift for next token prediction

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Get log probs of actual tokens
        target_ids = padded_ids[:, 1:]  # Shift targets
        target_log_probs = torch.gather(
            log_probs, dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Mask padding tokens
        mask = attention_mask[:, 1:].float()
        target_log_probs = target_log_probs * mask

        # Sum log probs per sequence
        sequence_log_probs = target_log_probs.sum(dim=-1)

        # Policy gradient loss: -advantages * log_probs
        loss = -(advantages_tensor * sequence_log_probs).mean()

        return loss

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
