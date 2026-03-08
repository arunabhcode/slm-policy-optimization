"""
Modal cloud launcher for SLM Policy Optimization training (main.py).

Usage from src/open_r1:
    uv run modal run train_modal.py --config ../../recipes/gspo.yaml
"""

import modal
import os

PROJECT_ROOT = "../.."

# Define the Modal image, using a CUDA devel base image so runtime CUDA ops can compile
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .add_local_dir(
        PROJECT_ROOT,
        remote_path="/workspace/slm-policy-optimization",
        ignore=[
            ".venv",
            ".git",
            "**/__pycache__",
            ".ruff_cache",
            "*.egg-info",
            "build",
            "dist",
        ],
        copy=True,
    )
    .workdir("/workspace/slm-policy-optimization")
    .run_commands("uv sync --all-extras")
    .env({"PATH": "/workspace/slm-policy-optimization/.venv/bin:$PATH"})
)

app = modal.App(
    name="slm-policy-optimization",
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"]),
    ],
)

GPU_CONFIG = "A100-80GB"
TIMEOUT_SECONDS = 24 * 60 * 60  # 24 hours

VOLUME = modal.Volume.from_name("slm-policy-checkpoints", create_if_missing=True)
CHECKPOINT_DIR = "/checkpoints"


@app.function(
    gpu=GPU_CONFIG,
    timeout=TIMEOUT_SECONDS,
    volumes={CHECKPOINT_DIR: VOLUME},
)
def train(config_path: str):
    """Run the training job on Modal."""
    import sys
    import yaml

    os.chdir("/workspace/slm-policy-optimization")

    # Add both src and src/open_r1 to the Python path
    sys.path.insert(0, "/workspace/slm-policy-optimization/src")
    sys.path.insert(0, "/workspace/slm-policy-optimization/src/open_r1")

    # Patch output_dir to use the persistent volume
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    original_output_dir = config_data.get("output_dir", "output")
    patched_output_dir = os.path.join(
        CHECKPOINT_DIR, os.path.basename(original_output_dir)
    )
    config_data["output_dir"] = patched_output_dir

    patched_config_path = "/tmp/patched_config.yaml"
    with open(patched_config_path, "w") as f:
        yaml.safe_dump(config_data, f)

    from open_r1.config import GSPOConfig
    from open_r1.main import main

    config = GSPOConfig.from_yaml(patched_config_path)
    main(config)

    VOLUME.commit()
    print(f"Training complete. Checkpoints saved to volume at: {patched_output_dir}")


@app.local_entrypoint()
def entrypoint(config: str):
    """Local entrypoint that dispatches training to Modal.

    Args:
        config: Path to the YAML configuration file.
    """
    if not os.path.exists(config):
        raise FileNotFoundError(f"Config file not found: {config}")

    # Resolve the config path relative to the project root for the remote container
    local_project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
    abs_config = os.path.abspath(config)
    remote_config_path = os.path.relpath(abs_config, local_project_root)

    print(f"Launching training on Modal with remote config path: {remote_config_path}")
    train.remote(config_path=remote_config_path)
    print("Training job completed.")
