# coding=utf-8

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GSPOConfig:
    model_name_or_path: str
    model_revision: str
    torch_dtype: str
    attn_implementation: str

    dataset_name: str
    dataset_config: str | None
    system_prompt: str
    trust_remote_code: bool
    chat_template: str

    vllm_enforce_eager: bool
    vllm_gpu_memory_utilization: float
    vllm_max_model_len: int
    do_eval: bool
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    gradient_checkpointing_kwargs: dict[str, Any]
    learning_rate: float
    log_completions: bool
    logging_steps: int
    lr_scheduler_type: str
    lr_scheduler_kwargs: dict[str, Any]
    max_completion_length: int
    max_steps: int
    num_generations: int
    num_train_epochs: int
    output_dir: str
    per_device_eval_batch_size: int
    per_device_train_batch_size: int
    report_to: list[str]
    reward_funcs: list[str]
    reward_weights: list[float]
    save_steps: int
    seed: int
    temperature: float
    warmup_ratio: float
    resume_from_checkpoint: bool

    cosine_min_value_wrong: float
    cosine_max_value_wrong: float
    cosine_min_value_correct: float
    cosine_max_value_correct: float
    cosine_max_len: int
    repetition_n_grams: int
    repetition_max_penalty: float
    code_language: str
    dataset_train_split: str

    epsilon: float
    beta: float

    auto_set_chat_template: bool = False
    vllm_mode: str = "server"  # "server" or "colocate"
    vllm_tensor_parallel_size: int = 1
    use_vllm: bool = True

    @classmethod
    def from_yaml(cls, path) -> "GSPOConfig":
        with Path(path).expanduser().open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
