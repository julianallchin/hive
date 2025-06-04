import torch
import math
from typing import Tuple, List

def setup_obs(sim, device: torch.device) -> torch.Tensor:
    """
    Process observations from Madrona environment.
    
    Args:
        sim: Madrona SimManager instance
        device: Device to place tensors on
        
    Returns:
        torch.Tensor: Processed observations of shape (num_worlds * num_agents, obs_dim)
    """
    # Get tensors from the simulation
    self_obs_tensor = sim.self_observation_tensor().to_torch()
    lidar_tensor = sim.lidar_tensor().to_torch()
    steps_remaining_tensor = sim.steps_remaining_tensor().to_torch()
    
    # Get batch dimensions (N: num_worlds, A: num_agents)
    N, A = self_obs_tensor.shape[0:2]
    batch_size = N * A
    
    # Reshape and combine observations
    obs_tensors = [
        self_obs_tensor.view(batch_size, *self_obs_tensor.shape[2:]),
        lidar_tensor.view(batch_size, *lidar_tensor.shape[2:]),
        steps_remaining_tensor.view(batch_size, *steps_remaining_tensor.shape[2:]),
    ]

    num_obs_features = 0
    for tensor in obs_tensors:
        num_obs_features += math.prod(tensor.shape[1:])
    
    return obs_tensors, num_obs_features

def get_dones(sim, device: torch.device) -> torch.Tensor:
    """
    Get done flags from the environment.
    
    Args:
        sim: Madrona SimManager instance
        device: Device to place tensors on
        
    Returns:
        torch.Tensor: Done flags of shape (num_worlds * num_agents,)
    """
    return sim.done_tensor().to_torch().view(-1).to(device)

def get_rewards(sim, device: torch.device) -> torch.Tensor:
    """
    Get rewards from the environment.
    
    Args:
        sim: Madrona SimManager instance
        device: Device to place tensors on
        
    Returns:
        torch.Tensor: Rewards of shape (num_worlds * num_agents,)
    """
    return sim.reward_tensor().to_torch().view(-1).to(device)

def get_actions(sim, device: torch.device) -> torch.Tensor:
    """
    Get action tensor from the environment.
    
    Args:
        sim: Madrona SimManager instance
        device: Device to place tensors on
        
    Returns:
        torch.Tensor: Action tensor
    """
    return sim.action_tensor().to_torch().to(device)

def get_obs_dim(sim) -> int:
    """
    Calculate the total observation dimension.
    
    Args:
        sim: Madrona SimManager instance
        
    Returns:
        int: Total observation dimension
    """
    # Get a sample observation to determine dimensions
    sample_obs = setup_obs(sim, torch.device('cpu'))
    return sample_obs.shape[-1]