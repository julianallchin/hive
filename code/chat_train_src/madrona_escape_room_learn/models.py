import torch
import torch.nn as nn
import torch.nn.functional as F

from .action import DiscreteActionDistributions
from .actor_critic import ActorCritic, DiscreteActor, Critic

class MLP(nn.Module):
    def __init__(self, input_dim, num_channels, num_layers):
        super().__init__()

        layers = [
            nn.Linear(input_dim, num_channels),
            nn.LayerNorm(num_channels),
            nn.ReLU(),
        ]
        for i in range(num_layers - 1):
            layers.append(nn.Linear(num_channels, num_channels))
            layers.append(nn.LayerNorm(num_channels))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, inputs):
        return self.net(inputs)


class AntMLP(nn.Module):
    def __init__(self, local_obs_dim, command_dim, hidden_dims, 
                 num_phys_action_logits, message_dim):
        """Enhanced AntMLP with variable number of hidden layers
        
        Args:
            local_obs_dim: Dimension of local observation input
            command_dim: Dimension of command input from hivemind
            hidden_dims: List of hidden layer dimensions (supports variable depth)
            num_phys_action_logits: Number of physical action logits to output
            message_dim: Dimension of message to send to hivemind
        """
        super().__init__()
        
        # Validate inputs
        if not isinstance(hidden_dims, (list, tuple)) or len(hidden_dims) < 1:
            raise ValueError("hidden_dims must be a list/tuple with at least one dimension")
            
        # Input dimension is combined local observation and command
        input_dim = local_obs_dim + command_dim
        
        # Build variable-depth network architecture
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # First layer from inputs to first hidden layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layer_norms.append(nn.LayerNorm(hidden_dims[0]))
        
        # Additional hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layer_norms.append(nn.LayerNorm(hidden_dims[i + 1]))
        
        # Output heads - connect to the last hidden layer
        self.action_head = nn.Linear(hidden_dims[-1], num_phys_action_logits)
        self.message_head = nn.Linear(hidden_dims[-1], message_dim)

        # Initialize weights for hidden layers
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
            if layer.bias is not None: 
                nn.init.constant_(layer.bias, val=0)
        
        # Action outputs should have small initialization
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.constant_(self.action_head.bias, 0)
        
        # Message outputs should have middle-range initialization
        nn.init.orthogonal_(self.message_head.weight, gain=0.1)
        nn.init.constant_(self.message_head.bias, 0)

    def forward(self, local_obs, command):
        # local_obs: [*, local_obs_dim]
        # command: [*, command_dim]
        
        # Concatenate observation and command inputs
        x = torch.cat([local_obs, command], dim=-1)
        
        # Process through variable-depth MLP
        for i, (layer, layer_norm) in enumerate(zip(self.layers, self.layer_norms)):
            x = F.relu(layer_norm(layer(x)))
        
        # Generate outputs
        phys_action_logits = self.action_head(x)
        message_to_hivemind = self.message_head(x)
        
        return phys_action_logits, message_to_hivemind


class HivemindAttention(nn.Module):
    def __init__(self, ant_message_dim, num_heads, attention_output_dim, dropout=0.1):
        super().__init__()
        # Using nn.MultiheadAttention for the attention mechanism
        self.mha = nn.MultiheadAttention(
            embed_dim=ant_message_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Optional projection if dimensions don't match
        self.fc_out = nn.Linear(ant_message_dim, attention_output_dim) if ant_message_dim != attention_output_dim else nn.Identity()

    def forward(self, ant_messages_batch):
        # ant_messages_batch: [batch_sims, num_ants, ant_message_dim]
        
        # Apply self-attention over ant messages
        attn_output, _ = self.mha(ant_messages_batch, ant_messages_batch, ant_messages_batch)
        
        # Global message: average over the "num_ants" dimension to get one vector per sim
        global_message_latent = torch.mean(attn_output, dim=1)  # [batch_sims, ant_message_dim]
        
        # Project to final output dimension if needed
        global_message = self.fc_out(global_message_latent) # [batch_sims, attention_output_dim]
        
        return global_message


class IdentityNet(nn.Module):
    """Simple identity network that returns input as output
    Used for the actor's impl when the backbone already provides action logits"""
    def forward(self, x):
        return x


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
