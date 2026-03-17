# Policy Optimization using GRPO-S

## Installation

### Prerequisites
Install `uv` for managing virtual environments:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Set up a virtual environment with Python 3.12:
```bash
uv venv --python 3.12
source .venv/bin/activate
```

### Dependencies
Install project dependencies:
```bash
uv sync
```

> **Note**: Ensure your lockfile/environment resolves to a PyTorch version compatible with `vLLM` (the project previously used PyTorch `v2.5.1`).

### Authentication
Log in to Hugging Face and Weights & Biases:
```bash
huggingface-cli login
wandb login
```

## Training

Train models using a YAML config with 3 GPUs:
```bash
python src/open_r1/gtpo.py
```

## Evaluation

Evaluate models using `lighteval` with custom tasks in `src/open_r1/evaluate.py`. For single-GPU setups:
```bash

# Example: AIME 2024
TASK=aime24
lighteval vllm "$MODEL_ARGS" "custom|$TASK|0|0" \
  --custom-tasks src/open_r1/evaluate.py \
  --use-chat-template \
  --output-dir "$OUTPUT_DIR"
```
