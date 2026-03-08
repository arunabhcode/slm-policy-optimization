#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import wandb
from logger import print
import numpy as np

PROJECT_NAME = "interview-co"


class Introspect:
    """
    Class for introspecting the model.
    """

    def __init__(self):
        """
        Initialize the introspection class.
        """
        wandb.login()
        print("Introspect initialized")

    def initialize(self, config_dict=None):
        """
        Initialize the introspection process.
        """
        wandb.init(
            project=PROJECT_NAME,
            config=config_dict,
            mode="online",  # this is online even for local w&b server even though messaging says cloud sync
        )

    def finalize(self):
        """
        Finalize the introspection process.
        """
        wandb.finish()

    def log_accuracy(self, accuracy):
        """
        Log the accuracy to Weights & Biases.
        """
        wandb.log({"accuracy": accuracy})

    def log_training_loss(self, training_loss):
        """
        Log the training_loss): to Weights & Biases.
        """
        wandb.log({"training_loss:": training_loss})

    def log_test_loss(self, test_loss):
        """
        Log the test_loss to Weights & Biases.
        """
        wandb.log({"test_loss": test_loss})

    def log_image_predictions(self, images, predictions, labels):
        """
        Log image predictions to Weights & Biases.
        """
        table = wandb.Table(columns=["Image", "Prediction", "Label"])
        for image, pred, label in zip(images, predictions, labels):
            table.add_data(
                wandb.Image(image.cpu().numpy() * 255), pred.cpu(), label.cpu()
            )
        wandb.log({"predictions": table}, commit=False)

    def log_model_summary(self, model):
        """
        Log the model summary to Weights & Biases.
        """
        wandb.watch(model, log="all", log_graph=True)

    def log_scalar_dict(self, scalar_dict):
        """
        Log a dictionary of scalar values to Weights & Biases.
        """
        wandb.log(scalar_dict)

    def log_feature_maps(self, feature_maps_dict):
        """
        Log model internal feature maps (activations).
        Logs the map for the first item in the batch, averaged over channels.
        Assumes feature_maps_dict values are tensors (B, C, H, W).
        """
        log_data = {}
        for layer_name, feature_map in feature_maps_dict.items():
            if feature_map is None or feature_map.ndim < 4 or feature_map.shape[0] == 0:
                print(f"Skipping feature map {layer_name}: Invalid shape or empty")
                continue

            # Use the first image in the batch
            first_map = feature_map[0]  # Shape: C, H, W

            # Average across channels
            avg_map = first_map.mean(dim=0)  # Shape: H, W

            # Normalize to [0, 1] for visualization
            min_val, max_val = avg_map.min(), avg_map.max()
            if max_val > min_val:
                normalized_map = (avg_map - min_val) / (max_val - min_val)
            else:
                normalized_map = avg_map  # Avoid division by zero if flat

            normalized_map_np = normalized_map.cpu().numpy()

            log_data[f"feature_map/{layer_name}"] = wandb.Image(normalized_map_np)

        if log_data:
            wandb.log(log_data)

    def log_attention_maps(self, attention_maps_dict):
        """
        Log model attention maps.
        Logs the map for the first item in the batch and the first head.
        Assumes attention_maps_dict values are tensors (e.g., B, Heads, SeqLen, SeqLen or B, Heads, H, W).
        """
        log_data = {}
        for map_name, attention_map in attention_maps_dict.items():
            if (
                attention_map is None
                or attention_map.ndim < 3
                or attention_map.shape[0] == 0
            ):  # Needs at least B, H, W or B, S, S
                print(f"Skipping attention map {map_name}: Invalid shape or empty")
                continue

            # Use the first item in the batch and the first head (if applicable)
            first_item_map = attention_map[0]
            if first_item_map.ndim > 2:  # e.g., Heads, H, W or Heads, S, S
                first_head_map = first_item_map[0]
            else:  # e.g., H, W or S, S
                first_head_map = first_item_map

            # Normalize to [0, 1] for visualization
            min_val, max_val = first_head_map.min(), first_head_map.max()
            if max_val > min_val:
                normalized_map = (first_head_map - min_val) / (max_val - min_val)
            else:
                normalized_map = first_head_map  # Avoid division by zero

            normalized_map_np = normalized_map.cpu().numpy()

            log_data[f"attention_map/{map_name}"] = wandb.Image(normalized_map_np)
            # Consider using heatmaps or allowing selection of specific heads

        if log_data:
            wandb.log(log_data)
