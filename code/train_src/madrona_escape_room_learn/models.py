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
    def __init__(self, local_obs_dim, command_dim, mlp_hidden_dim, 
                 num_phys_action_logits, message_dim):
        super().__init__()
        # Neural network layers for ant's processing
        self.fc1 = nn.Linear(local_obs_dim + command_dim, mlp_hidden_dim)
        self.ln1 = nn.LayerNorm(mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.ln2 = nn.LayerNorm(mlp_hidden_dim)
        
        # Output heads
        self.action_head = nn.Linear(mlp_hidden_dim, num_phys_action_logits)
        self.message_head = nn.Linear(mlp_hidden_dim, message_dim)

        # Initialize weights
        for layer in [self.fc1, self.fc2]:
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
        
        # Process through MLP
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        
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
