from dataclasses import dataclass
from typing import Callable, List

import torch

@dataclass(frozen=True)
class PPOConfig:
    num_mini_batches: int
    clip_coef: float
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float
    num_epochs: int = 1
    clip_value_loss: bool = False
    adaptive_entropy: bool = True

@dataclass(frozen=True)
class HivemindConfig:
    """Configuration for hivemind architecture components"""
    # Ant configuration
    num_ants: int  # Number of ants per simulation
    ant_obs_dim: int  # Dimension of local observation per ant
    ant_mlp_hidden: int  # Hidden dimension in ant MLP
    
    # Communication parameters
    command_dim: int  # Dimension of command from hivemind to ants
    message_dim: int  # Dimension of message from ants to hivemind
    
    # Attention parameters
    attn_heads: int  # Number of attention heads 
    attn_output_dim: int  # Output dimension of attention
    
    # LSTM parameters 
    lstm_hidden_dim: int  # Hidden dimension of LSTM
    lstm_layers: int  # Number of LSTM layers

@dataclass(frozen=True)
class TrainConfig:
    num_updates: int
    steps_per_update: int
    num_bptt_chunks: int
    lr: float
    gamma: float
    ppo: PPOConfig
    hivemind: HivemindConfig  # Hivemind configuration
    gae_lambda: float = 1.0
    normalize_advantages: bool = True
    normalize_values : bool = True
    value_normalizer_decay : float = 0.99999
    mixed_precision : bool = False

    def __repr__(self):
        rep = "TrainConfig:"

        for k, v in self.__dict__.items():
            if k == 'ppo':
                rep += f"\n  ppo:"
                for ppo_k, ppo_v in self.ppo.__dict__.items():
                    rep += f"\n    {ppo_k}: {ppo_v}"
            elif k == 'hivemind':
                rep += f"\n  hivemind:"
                for hive_k, hive_v in self.hivemind.__dict__.items():
                    rep += f"\n    {hive_k}: {hive_v}"
            else:
                rep += f"\n  {k}: {v}" 

        return rep

@dataclass(frozen=True)
class SimInterface:
    step: Callable
    obs: List[torch.Tensor]  # First element should be [batch_size, num_ants, ant_obs_dim]
    actions: torch.Tensor    # Shape [batch_size, num_ants, action_components]
    dones: torch.Tensor      # Shape [batch_size, 1]
    rewards: torch.Tensor    # Shape [batch_size, 1] - global reward for the hivemind
