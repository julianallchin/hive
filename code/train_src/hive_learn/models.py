import torch
import torch.nn as nn
import torch.nn.functional as F

from .action import DiscreteActionDistributions
from .actor_critic import ActorCritic, DiscreteActor, Critic

class MLP(nn.Module):
    def __init__(self, dims):
        """Enhanced MLP with flexible architecture
        
        Args:
            dims: List of dimensions for the entire network, including input and output dims
                  e.g. [64, 128, 64, 32] for input_dim=64, two hidden layers (128, 64), and output_dim=32
                  Must have at least length 2 (for input and output dimensions)
        """
        super().__init__()
        
        # Validate input
        if not isinstance(dims, (list, tuple)) or len(dims) < 2:
            raise ValueError("dims must be a list/tuple with at least 2 dimensions (input and output)")
        
        # Build network with variable width layers
        layers = []
        
        # Create layers between each pair of dimensions
        for i in range(len(dims) - 1):
            # Linear transformation
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # Add normalization and activation to all layers
            layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(nn.ReLU())
            
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                # Use kaiming normal initialization for all layers
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, inputs):
        return self.net(inputs)

class LinearLayerDiscreteActor(DiscreteActor):
    def __init__(self, actions_num_buckets, in_channels):
        total_action_dim = sum(actions_num_buckets)
        impl = nn.Linear(in_channels, total_action_dim)

        super().__init__(actions_num_buckets, impl)

        nn.init.orthogonal_(self.impl.weight, gain=0.01)
        nn.init.constant_(self.impl.bias, 0)

class LinearLayerCritic(Critic):
    def __init__(self, in_channels):
        super().__init__(nn.Linear(in_channels, 1))

        nn.init.orthogonal_(self.impl.weight)
        nn.init.constant_(self.impl.bias, 0)
