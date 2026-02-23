import torch


class GSPOTrainer:
    def __init__(
        self,
        config,
    ):
        self.config = config

        # TODO: init model/optimizer/scheduler/ref_model as needed

    def train(self, resume_from_checkpoint=None):
        # TODO: rollout -> rewards -> advantages -> policy update
        # return an object or dict similar to HF trainer output
        return {"train_loss": 0.0}

    def evaluate(self):
        # TODO
        return {}

    def save_model(self, output_dir=None):
        # TODO
        pass

    def log_metrics(self, split, metrics):
        print(f"[{split}] {metrics}")

    def save_metrics(self, split, metrics):
        pass

    def save_state(self):
        pass

    def push_to_hub(self, **kwargs):
        pass
