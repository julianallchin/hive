import torch
import torch.nn as nn
import torch.nn.functional as F

from .action import DiscreteActionDistributions
from .actor_critic import ActorCritic, DiscreteActor, MultiAgentCritic, Critic

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

class MultiAgentSharedMLP(nn.Module):
    def __init__(self, input_dim_per_agent, num_channels_per_agent, num_layers):
        super().__init__()
        self.mlp = MLP(input_dim_per_agent, num_channels_per_agent, num_layers)
        self.input_dim_per_agent = input_dim_per_agent

    # inputs: [N * M, A * -1]
    def forward(self, inputs):
        unflattened_inputs = inputs.view(inputs.shape[0], -1, self.input_dim_per_agent)
        action_logits = self.mlp(unflattened_inputs)
        return action_logits.view(inputs.shape[0], -1) # [N * M, A * -1]


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

class MultiAgentLinearLayerCritic(MultiAgentCritic):
    def __init__(self, num_agents, in_channels_per_agent):
        super().__init__(nn.Linear(in_channels_per_agent * num_agents, 1))

        nn.init.orthogonal_(self.impl.weight)
        nn.init.constant_(self.impl.bias, 0)

class AttentionEncoder(nn.Module):
    def __init__(self, input_dim_per_agent, num_channels_per_agent, num_layers, output_dim):
        super().__init__()
        self.input_dim_per_agent = input_dim_per_agent
        self.num_channels_per_agent = num_channels_per_agent
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        self.query = nn.Parameter(torch.zeros(1, 1, num_channels_per_agent))
        self.mlp = MLP(input_dim_per_agent, num_channels_per_agent, num_layers)
        self.attn = nn.MultiheadAttention(num_channels_per_agent, num_heads=1, batch_first=True)
        self.mlp_out = nn.Linear(num_channels_per_agent, output_dim)

    def forward(self, inputs):
        # inputs: [N * M, A * -1]
        assert (len(inputs.shape) == 2)
        
        unflattened_inputs = inputs.view(inputs.shape[0], -1, self.input_dim_per_agent)
        mlp_out = self.mlp(unflattened_inputs)
        attn_out, _ = self.attn(self.query.expand(inputs.shape[0], -1, -1), mlp_out, mlp_out)
        flat_attn_out = attn_out.view(inputs.shape[0], -1)
        return self.mlp_out(flat_attn_out)
        