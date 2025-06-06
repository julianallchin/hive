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

class MultiAgentMLP(nn.Module):
    def __init__(self, num_agents, input_dim_per_agent, num_channels_per_agent, num_layers):
        super().__init__()
        self.mlp = MLP(input_dim_per_agent * num_agents, num_channels_per_agent * num_agents, num_layers)
        self.num_agents = num_agents

    # inputs: [N * M, A, -1]
    def forward(self, inputs):
        flat_inputs = inputs.view(inputs.shape[0], -1)
        flat_outputs = self.mlp(flat_inputs)
        return flat_outputs.view(*inputs.shape[0:2], -1)


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