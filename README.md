# SLM Policy Optimization

> Based on **Paper**: [Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't](https://arxiv.org/abs/2503.16219)  

## Overview

This repository accompanies our research on applying reinforcement learning algorithms to improve reasoning capabilities in small language models (1.5B parameters). We investigate what training strategies, datasets, and hyperparameters yield the best reasoning performance at minimal cost.

## Repo Map

> **Note**: The main training and evaluation code lives on other branches. This branch serves as the project landing page and documentation hub.

### Branches

| Branch | Description |
|:---|---|
| `main` | Project landing page |
| `arunabh/gspo` | GSPO (Group Sequence Policy Optimization) training & evaluation code |
| `tatiana/grpo-s` | GRPO-S (Sequence level GRPO variant with dynamic entropy weighting) training & evaluation code |

## Datasets

| Dataset | Link |
|---|---|
| open-rs | [🤗 knoveleng/open-rs](https://huggingface.co/datasets/knoveleng/open-rs) |


