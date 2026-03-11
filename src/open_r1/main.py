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

import argparse
import logging
import os
import sys

import datasets
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from gspo import GSPOTrainer
import torch

from open_r1.config import GSPOConfig
from open_r1.introspect import Introspect
from open_r1.utils.model_utils import get_tokenizer


logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.multiprocessing.set_start_method('spawn', force=True)
os.environ["VLLM_USE_V1"] = "0"


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
    # logger.info(f"Config parameters {config}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(config.output_dir):
        last_checkpoint = get_last_checkpoint(config.output_dir)
    if last_checkpoint is not None and config.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    introspect = None
    if "wandb" in config.report_to:
        config_dict = {
            k: v for k, v in config.__dict__.items() if not k.startswith("_")
        }
        introspect = Introspect()
        introspect.initialize(entity_name="vrshy-stanford", project_name="slm-policy", config_dict=config_dict)

    # Load the dataset
    dataset = load_dataset(config.dataset_name, name=config.dataset_config)
    dataset[config.dataset_train_split] = dataset[config.dataset_train_split].select(range(4))
    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(config)

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
    eval_dataset = (
        dataset[config.dataset_test_split].select(range(4))
        if config.do_eval and config.dataset_test_split in dataset
        else None
    )
    trainer = GSPOTrainer(
        config=config,
        train_dataset=dataset[config.dataset_train_split],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        introspect=introspect,
    )

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
    trainer.save_model()
    logger.info(f"Model saved to {config.output_dir}")

    ##########
    # Evaluate
    ##########
    if config.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        if config.dataset_test_split in dataset:
            metrics["eval_samples"] = len(dataset[config.dataset_test_split])
        else:
            metrics["eval_samples"] = metrics.get("eval/num_samples", 0)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if introspect is not None:
        introspect.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with GSPO from a YAML config.")
    parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML configuration file (e.g. recipes/gspo.yaml)",
    )
    args = parser.parse_args()
    config = GSPOConfig.from_yaml(args.config)
    main(config)
