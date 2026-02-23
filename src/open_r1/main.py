# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import builtins
import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import torch
import transformers
import yaml
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from gspo import GSPOTrainer

# Compatibility shim: rewards.py evaluates `AsyncSandbox` in type annotations at import time.
if not hasattr(builtins, "AsyncSandbox"):
    builtins.AsyncSandbox = object  # type: ignore[attr-defined]

# from open_r1.configs import GRPOConfig
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

from open_r1.config import GSPOConfig
# from open_r1.utils import get_tokenizer
# from open_r1.utils.callbacks import get_callbacks
# from open_r1.utils.wandb_logging import init_wandb_training


logger = logging.getLogger(__name__)


def main(config):
    # Set seed for reproducibility
    set_seed(config.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.info(f"Config parameters {config}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(config.output_dir):
        last_checkpoint = get_last_checkpoint(config.output_dir)
    if last_checkpoint is not None and config.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # if "wandb" in config.report_to:
    # init_wandb_training(config)

    # Load the dataset
    dataset = load_dataset(config.dataset_name, name=config.dataset_config)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(config.model_args, config.training_args)

    # Get reward functions
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
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=config.code_language),
        "tag_count": tag_count_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in config.reward_funcs]

    # Format into conversation
    def make_conversation(example):
        prompt = []

        if config.system_prompt is not None:
            prompt.append({"role": "system", "content": config.system_prompt})

        prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GSPOTrainer(
        config=config,
    )
    print("Trainer dict:", trainer.__dict__)
    exit(0)
    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if config.resume_from_checkpoint:
        checkpoint = config.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[config.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(config.output_dir)
    logger.info(f"Model saved to {config.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": config.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(config.output_dir)

    ##########
    # Evaluate
    ##########
    if config.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[config.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    print("Hello GSPO!")
    yaml_path = "/home/arunabh/slm-policy-optimization/recipes/gspo.yaml"
    # yaml_path = sys.argv[1]
    GSPOConfig = GSPOConfig.from_yaml(yaml_path)
    main(GSPOConfig)
