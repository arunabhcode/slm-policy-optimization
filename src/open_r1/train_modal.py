"""
Modal cloud launcher for SLM Policy Optimization training (main.py).

Usage:
    modal run train_modal.py --config recipes/gspo.yaml
"""

import modal
import os

# Define the Modal image, installing deps from pyproject.toml
image = (
    modal.Image.debian_slim(python_version="3.12")
    .copy_local_dir(".", remote_path="/workspace/slm-policy-optimization")
    .pip_install_from_pyproject(
        "/workspace/slm-policy-optimization/pyproject.toml",
    )
    .run_commands(
        "pip install -e /workspace/slm-policy-optimization",
    )
)

app = modal.App(
    name="slm-policy-optimization",
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"]),
    ],
)

GPU_CONFIG = modal.gpu.A100(count=1, size="80GB")
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
    sys.path.insert(0, "/workspace/slm-policy-optimization/src")

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
    from open_r1.main import main  # fixed import

    config = GSPOConfig.from_yaml(patched_config_path)
    main(config)

    VOLUME.commit()

    print(f"Training complete. Checkpoints saved to volume at: {patched_output_dir}")


@app.local_entrypoint()
def entrypoint(config: str = "recipes/gspo.yaml"):
    """Local entrypoint that dispatches training to Modal.

    Args:
        config: Path to the YAML configuration file.
    """
    if not os.path.exists(config):
        raise FileNotFoundError(f"Config file not found: {config}")

    print(f"Launching training on Modal with config: {config}")
    train.remote(config_path=config)
    print("Training job completed.")
