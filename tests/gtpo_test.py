import torch
import pytest
from unittest.mock import Mock


from open_r1.grpo import GTPOTrainer

def test_standard_mixed_batch():
    """Test 1: Normal conditions. Half successes, half failures."""
    B, G, S = 2, 4, 32
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0,  1.0, 1.0, 0.0, 0.0])
    entropies = torch.rand(B * G, S)
    
    mock_self = Mock()
    mock_self.num_generations = G
    
    adv = GTPOTrainer._get_gtpo_advantages(mock_self, rewards, entropies)
    
    assert adv.shape == (B * G, 1), "Advantage shape is incorrect. View(-1, 1) is likely missing."
    assert not torch.isnan(adv).any(), "NaNs detected in advantages."

def test_all_success_batch():
    """Test 2: Ensures positive advantages don't explode when variance is zero."""
    B, G, S = 1, 2, 32
    rewards = torch.tensor([1.0, 1.0])
    entropies = torch.rand(B * G, S)
    
    mock_self = Mock()
    mock_self.num_generations = G
    
    adv = GTPOTrainer._get_gtpo_advantages(mock_self, rewards, entropies)
    
    assert not torch.isnan(adv).any(), "NaNs detected when all generations succeed."

def test_all_failure_batch():
    """Test 3: Ensures negative advantages don't explode when variance is zero."""
    B, G, S = 1, 2, 32
    rewards = torch.tensor([0.0, 0.0])
    entropies = torch.rand(B * G, S)
    
    mock_self = Mock()
    mock_self.num_generations = G
    
    adv = GTPOTrainer._get_gtpo_advantages(mock_self, rewards, entropies)
    
    assert not torch.isnan(adv).any(), "NaNs detected when all generations fail."

def test_nan_padding_handling():
    """Test 4: Validates that padding tokens (represented as NaNs) are ignored gracefully."""
    B, G, S = 1, 2, 4
    rewards = torch.tensor([1.0, 0.0])
    entropies = torch.tensor([
        [0.5, 0.4, float('nan'), float('nan')],
        [0.6, 0.7, float('nan'), float('nan')]
    ])
    
    mock_self = Mock()
    mock_self.num_generations = G
    
    adv = GTPOTrainer._get_gtpo_advantages(mock_self, rewards, entropies)
    
    assert not torch.isnan(adv).any(), "Function failed to handle NaN padding in entropies."

def test_fp16_extreme_low_entropy():
    """Test 5: The FP16 Overflow Trap. Simulates a highly confident model."""
    B, G, S = 1, 2, 32
    rewards = torch.tensor([0.0, 0.0], dtype=torch.float16)
    
    # Extremely low entropy, cast to FP16
    entropies = torch.full((B * G, S), 1e-6, dtype=torch.float16)
    
    mock_self = Mock()
    mock_self.num_generations = G
    
    adv = GTPOTrainer._get_gtpo_advantages(mock_self, rewards, entropies)
    
    assert not torch.isnan(adv).any(), "FP16 Overflow detected in advantages! Clamping failed."