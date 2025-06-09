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

class MultiAgentLinearDiscreteActor(nn.Module):
    def __init__(self, actions_num_buckets, in_channels_per_agent):
        super().__init__()
        total_action_dim = sum(actions_num_buckets)
        self.in_channels_per_agent = in_channels_per_agent
        self.actions_num_buckets = actions_num_buckets
        self.impl = nn.Linear(in_channels_per_agent, total_action_dim)

        nn.init.orthogonal_(self.impl.weight, gain=0.01)
        nn.init.constant_(self.impl.bias, 0)

    def forward(self, features_in):
        assert(len(features_in.shape) == 2)
        assert(features_in.shape[1] % self.in_channels_per_agent == 0)
        num_agents = features_in.shape[1] // self.in_channels_per_agent

        features_in_per_agent = features_in.view(features_in.shape[0], num_agents, self.in_channels_per_agent) # [N * M, A, in_channels_per_agent]
        action_logits = self.impl(features_in_per_agent) # [N * M, A, total_action_dim]
        flat_action_logits = action_logits.view(action_logits.shape[0], -1) # [N * M, A * total_action_dim]
        assert(flat_action_logits.shape[1] == num_agents * sum(self.actions_num_buckets))
        return DiscreteActionDistributions( # actions_num_buckets is a list of how many options per action
                num_agents * self.actions_num_buckets, logits=flat_action_logits)

        

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
    def __init__(self, input_dim_per_agent, msg_dim, mlp1_num_layers, mlp2_num_layers, output_dim):
        super().__init__()
        self.input_dim_per_agent = input_dim_per_agent
        self.msg_dim = msg_dim
        self.mlp1_num_layers = mlp1_num_layers
        self.mlp2_num_layers = mlp2_num_layers
        self.output_dim = output_dim
        
        self.query = nn.Parameter(torch.zeros(1, 1, msg_dim))
        self.mlp1 = MLP(input_dim_per_agent, msg_dim, mlp1_num_layers)
        self.attn = nn.MultiheadAttention(msg_dim, num_heads=1, batch_first=True)
        self.mlp2 = MLP(msg_dim, output_dim, mlp2_num_layers)

    def forward(self, inputs):
        # inputs: [N * M, A * -1]
        assert(len(inputs.shape) == 2)
        assert(inputs.shape[1] % self.input_dim_per_agent == 0)
        unflattened_inputs = inputs.view(inputs.shape[0], -1, self.input_dim_per_agent)
        msg = self.mlp1(unflattened_inputs)
        pooled_msg, _ = self.attn(self.query.expand(inputs.shape[0], -1, -1), msg, msg)
        flat_pooled_msg = pooled_msg.view(inputs.shape[0], -1)
        ret = self.mlp2(flat_pooled_msg) # [N * M, output_dim]
        # assert(ret.shape[1] == self.output_dim)
        # assert(len(ret.shape) == 2)
        return ret
        
class DictatorAttentionActorEncoder(nn.Module):
    def __init__(self, obs_per_agent, agent_msg_dim, ant_msg_mlp_num_layers, command_mlp_num_layers, ant_action_mlp_num_layers, command_dim, action_logits_dim):
        super().__init__()
        self.obs_per_agent = obs_per_agent
        self.agent_msg_dim = agent_msg_dim
        self.ant_msg_mlp_num_layers = ant_msg_mlp_num_layers
        self.command_mlp_num_layers = command_mlp_num_layers
        self.ant_action_mlp_num_layers = ant_action_mlp_num_layers
        self.command_dim = command_dim
        self.action_logits_dim = action_logits_dim
        
        self.msg_to_command = AttentionEncoder(obs_per_agent, agent_msg_dim, ant_msg_mlp_num_layers, command_mlp_num_layers, command_dim)
        self.command_and_obs_to_action_logits = MLP(
            input_dim = command_dim + obs_per_agent,
            num_channels = action_logits_dim,
            num_layers = ant_action_mlp_num_layers
        )
    
    def forward(self, obs):

        # obs: [N * M, A * -1]
        # assert(len(obs.shape) == 2)
        # assert(obs.shape[1] % self.obs_per_agent == 0)
        num_agents = obs.shape[1] // self.obs_per_agent

        command = self.msg_to_command(obs) # [N * M, command_dim]
        # assert(command.shape[1] == self.command_dim)
        # assert(len(command.shape) == 2)
        
        expanded_command = command.view(command.shape[0], 1, command.shape[1]).expand(-1, num_agents, -1) # [N*M, A, command_dim]
        obs_by_agent = obs.view(obs.shape[0], num_agents, self.obs_per_agent) # [N * M, A, obs_per_agent]
        flattened_ant_input = torch.cat([expanded_command, obs_by_agent], dim=-1) # [N * M, A, command_dim + obs_per_agent]
        # assert(len(flattened_ant_input.shape) == 3)
        action_logits = self.command_and_obs_to_action_logits(flattened_ant_input) # [N * M, A, action_logits_dim]
        # assert(action_logits.shape[1] == num_agents)
        # assert(action_logits.shape[2] == self.action_logits_dim)
        # assert(len(action_logits.shape) == 3)
        return action_logits.view(obs.shape[0], -1) # [N * M, A * action_logits_dim]