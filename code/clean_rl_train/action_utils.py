import torch
import numpy as np
from typing import Tuple, List
import torch.distributions as dist

# Action space dimensions
MOVE_AMOUNT_BINS = 4  # [0, 3]
MOVE_ANGLE_BINS = 8   # [0, 7]
ROTATE_BINS = 5       # [-2, 2]
GRAB_BINS = 2         # [0, 1]

# Action space offsets (for converting to/from flat indices)
MOVE_ANGLE_OFFSET = MOVE_AMOUNT_BINS
ROTATE_OFFSET = MOVE_ANGLE_OFFSET * MOVE_ANGLE_BINS
GRAB_OFFSET = ROTATE_OFFSET * ROTATE_BINS
TOTAL_ACTIONS = MOVE_AMOUNT_BINS * MOVE_ANGLE_BINS * ROTATE_BINS * GRAB_BINS

def get_action_dims() -> dict:
    """Return a dictionary with the dimensions of each action component."""
    return {
        'move_amount': MOVE_AMOUNT_BINS,
        'move_angle': MOVE_ANGLE_BINS,
        'rotate': ROTATE_BINS,
        'grab': GRAB_BINS,
        'total': TOTAL_ACTIONS
    }

def split_action_logits(logits: torch.Tensor) -> dict:
    """
    Split the flat action logits into separate components.
    
    Args:
        logits: Tensor of shape [batch_size, TOTAL_ACTIONS]
        
    Returns:
        Dictionary with separated logits for each action component
    """
    batch_size = logits.shape[0]
    
    # Reshape to separate the action components
    logits = logits.view(batch_size, MOVE_AMOUNT_BINS, MOVE_ANGLE_BINS, ROTATE_BINS, GRAB_BINS)
    
    # Sum over other dimensions to get logits for each component
    move_amount_logits = logits.sum(dim=(2, 3, 4))  # [batch_size, MOVE_AMOUNT_BINS]
    move_angle_logits = logits.sum(dim=(1, 3, 4))  # [batch_size, MOVE_ANGLE_BINS]
    rotate_logits = logits.sum(dim=(1, 2, 4))      # [batch_size, ROTATE_BINS]
    grab_logits = logits.sum(dim=(1, 2, 3))        # [batch_size, GRAB_BINS]
    
    return {
        'move_amount': move_amount_logits,
        'move_angle': move_angle_logits,
        'rotate': rotate_logits,
        'grab': grab_logits
    }

def sample_actions(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample actions from the action logits.
    
    Args:
        logits: Tensor of shape [batch_size, TOTAL_ACTIONS]
        
    Returns:
        Tuple of (actions, log_probs) where:
        - actions: Tensor of shape [batch_size, 4] containing the sampled actions
        - log_probs: Tensor of shape [batch_size] containing the log probabilities of the sampled actions
    """
    action_probs = torch.softmax(logits, dim=-1)
    action_dists = dist.Categorical(probs=action_probs)
    flat_actions = action_dists.sample()
    log_probs = action_dists.log_prob(flat_actions)
    
    # Convert flat indices to multi-discrete actions
    actions = flat_to_multi_discrete(flat_actions)
    
    return actions, log_probs

def flat_to_multi_discrete(flat_actions: torch.Tensor) -> torch.Tensor:
    """
    Convert flat action indices to multi-discrete actions.
    
    Args:
        flat_actions: Tensor of shape [batch_size] containing flat action indices
        
    Returns:
        Tensor of shape [batch_size, 4] containing the multi-discrete actions
    """
    grab = flat_actions % GRAB_BINS
    rotate = (flat_actions // GRAB_BINS) % ROTATE_BINS - 2  # Convert to [-2, 2]
    move_angle = (flat_actions // (GRAB_BINS * ROTATE_BINS)) % MOVE_ANGLE_BINS
    move_amount = (flat_actions // (GRAB_BINS * ROTATE_BINS * MOVE_ANGLE_BINS)) % MOVE_AMOUNT_BINS
    
    return torch.stack([move_amount, move_angle, rotate, grab], dim=-1)

def log_prob_actions(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log probability of the given actions under the current policy.
    
    Args:
        logits: Tensor of shape [batch_size, TOTAL_ACTIONS] containing action logits
        actions: Tensor of shape [batch_size, 4] containing multi-discrete actions
        
    Returns:
        Tensor of shape [batch_size] containing the log probabilities
    """
    # Convert multi-discrete actions to flat indices
    flat_actions = multi_discrete_to_flat(actions)
    
    # Calculate log probabilities
    action_probs = torch.softmax(logits, dim=-1)
    action_dists = dist.Categorical(probs=action_probs)
    
    return action_dists.log_prob(flat_actions)

def multi_discrete_to_flat(actions: torch.Tensor) -> torch.Tensor:
    """
    Convert multi-discrete actions to flat indices.
    
    Args:
        actions: Tensor of shape [batch_size, 4] containing multi-discrete actions
        
    Returns:
        Tensor of shape [batch_size] containing flat action indices
    """
    move_amount = actions[..., 0].long()
    move_angle = actions[..., 1].long()
    rotate = (actions[..., 2].long() + 2) % ROTATE_BINS  # Convert from [-2, 2] to [0, 4]
    grab = actions[..., 3].long()
    
    return ((move_amount * MOVE_ANGLE_BINS + move_angle) * ROTATE_BINS + rotate) * GRAB_BINS + grab

def entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculate the entropy of the action distribution.
    
    Args:
        logits: Tensor of shape [batch_size, TOTAL_ACTIONS] containing action logits
        
    Returns:
        Tensor of shape [batch_size] containing the entropy
    """
    action_probs = torch.softmax(logits, dim=-1)
    action_dists = dist.Categorical(probs=action_probs)
    return action_dists.entropy()
