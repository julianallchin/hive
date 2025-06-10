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
        # assert(len(features_in.shape) == 2)
        # assert(features_in.shape[1] % self.in_channels_per_agent == 0)
        num_agents = features_in.shape[1] // self.in_channels_per_agent

        features_in_per_agent = features_in.view(features_in.shape[0], num_agents, self.in_channels_per_agent) # [N * M, A, in_channels_per_agent]
        action_logits = self.impl(features_in_per_agent) # [N * M, A, total_action_dim]
        flat_action_logits = action_logits.view(action_logits.shape[0], -1) # [N * M, A * total_action_dim]
        # assert(flat_action_logits.shape[1] == num_agents * sum(self.actions_num_buckets))
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
    def __init__(self,
                 input_dim_per_agent,
                 msg_dim,
                 mlp1_num_layers,
                 num_attn_heads,
                 mlp2_num_layers,
                 output_dim):
        super().__init__()
        self.input_dim_per_agent = input_dim_per_agent
        self.msg_dim = msg_dim
        self.mlp1_num_layers = mlp1_num_layers
        self.mlp2_num_layers = mlp2_num_layers
        self.output_dim = output_dim
        
        self.mlp1 = MLP(input_dim_per_agent, msg_dim, mlp1_num_layers)
        self.query = nn.Parameter(torch.rand(1, 1, msg_dim))
        self.attn = nn.MultiheadAttention(msg_dim, num_heads=num_attn_heads, batch_first=True)
        self.mlp2 = MLP(msg_dim, output_dim, mlp2_num_layers)

    def forward(self, inputs):
        # inputs: [N * M, A * -1]
        # assert(len(inputs.shape) == 2)
        # assert(inputs.shape[1] % self.input_dim_per_agent == 0)
        unflattened_inputs = inputs.view(inputs.shape[0], -1, self.input_dim_per_agent)
        msg = self.mlp1(unflattened_inputs)
        pooled_msg, _ = self.attn(self.query.expand(inputs.shape[0], -1, -1), msg, msg)
        flat_pooled_msg = pooled_msg.view(inputs.shape[0], -1)
        ret = self.mlp2(flat_pooled_msg) # [N * M, output_dim]
        # assert(ret.shape[1] == self.output_dim)
        # assert(len(ret.shape) == 2)
        return ret
        
class DictatorAttentionActorEncoder(nn.Module):
    def __init__(self, obs_per_agent, agent_msg_dim, agent_msg_mlp_num_layers, command_mlp_num_layers, num_attn_heads, agent_action_mlp_num_layers, command_dim, action_logits_dim):
        super().__init__()
        self.obs_per_agent = obs_per_agent
        self.agent_msg_dim = agent_msg_dim
        self.agent_msg_mlp_num_layers = agent_msg_mlp_num_layers
        self.command_mlp_num_layers = command_mlp_num_layers
        self.agent_action_mlp_num_layers = agent_action_mlp_num_layers
        self.command_dim = command_dim
        self.action_logits_dim = action_logits_dim
        
        self.msg_to_command = AttentionEncoder(
            input_dim_per_agent=obs_per_agent,
            msg_dim=agent_msg_dim,
            num_attn_heads=num_attn_heads,
            mlp1_num_layers=agent_msg_mlp_num_layers,
            mlp2_num_layers=command_mlp_num_layers,
            output_dim=command_dim
        )
        self.command_and_obs_to_action_logits = MLP(
            input_dim = command_dim + obs_per_agent,
            num_channels = action_logits_dim,
            num_layers = agent_action_mlp_num_layers
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
        flattened_agent_input = torch.cat([expanded_command, obs_by_agent], dim=-1) # [N * M, A, command_dim + obs_per_agent]
        # assert(len(flattened_agent_input.shape) == 3)
        action_logits = self.command_and_obs_to_action_logits(flattened_agent_input) # [N * M, A, action_logits_dim]
        # assert(action_logits.shape[1] == num_agents)
        # assert(action_logits.shape[2] == self.action_logits_dim)
        # assert(len(action_logits.shape) == 3)
        return action_logits.view(obs.shape[0], -1) # [N * M, A * action_logits_dim]


class RecurrentAttentionActorEncoder(nn.Module):
    def __init__(self,
                 obs_per_agent,
                 agent_msg_dim,
                 agent_mlp_num_layers,
                 lstm_hidden_size,
                 num_attn_heads,
                 pooled_msg_dim,
                 pooled_msg_mlp_num_layers,
                 agent_action_mlp_num_layers,
                 command_mlp_num_layers,
                 command_dim,
                 action_logits_dim):
        super().__init__()
        self.obs_per_agent = obs_per_agent
        self.agent_msg_dim = agent_msg_dim
        self.agent_mlp_num_layers = agent_mlp_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.pooled_msg_dim = pooled_msg_dim
        self.pooled_msg_mlp_num_layers = pooled_msg_mlp_num_layers
        self.agent_action_mlp_num_layers = agent_action_mlp_num_layers
        self.command_dim = command_dim
        self.action_logits_dim = action_logits_dim
        self.command_mlp_num_layers = command_mlp_num_layers

        
        self.hidden_shape = (2, 1, self.lstm_hidden_size)

        self.agent_mlp = MLP(
            input_dim = obs_per_agent + command_dim,
            num_channels = action_logits_dim + agent_msg_dim,
            num_layers = agent_mlp_num_layers
        )

        self.query = nn.Parameter(torch.rand(1, 1, agent_msg_dim))
        self.attn = nn.MultiheadAttention(agent_msg_dim, num_heads=num_attn_heads, batch_first=True)

        self.pooled_msg_mlp = MLP(
            input_dim = agent_msg_dim,
            num_channels = pooled_msg_dim,
            num_layers = pooled_msg_mlp_num_layers
        )

        self.lstm = nn.LSTM(
            input_size = pooled_msg_dim,
            hidden_size = lstm_hidden_size,
            batch_first = True
        )

        self.command_mlp = MLP(
            input_dim = lstm_hidden_size,
            num_channels = command_dim,
            num_layers = command_mlp_num_layers
        )

    def forward(self, obs, rnn_states):        
        # # obs: [N * M, A * -1]
        # assert(len(obs.shape) == 2)
        # assert(obs.shape[1] % self.obs_per_agent == 0)
        num_agents = obs.shape[1] // self.obs_per_agent

        h_curr = rnn_states[0, ...] # [num_layers, N * M, lstm_hidden_size]
        c_curr = rnn_states[1, ...] # [num_layers, N * M, lstm_hidden_size]

        # assert(h_curr.shape[0] == 1)
        # assert(c_curr.shape[0] == 1)

        # assert(h_curr.shape[1] == obs.shape[0])
        # assert(c_curr.shape[1] == obs.shape[0])

        # agents use the command from the previous step to theoretically enable parallel operation
        prev_command = self.command_mlp(h_curr[-1, ...]) # [N * M, command_dim]. we use hidden state of last layer.
        # assert(prev_command.shape[0] == obs.shape[0])
        # assert(prev_command.shape[1] == self.command_dim)
        # assert(len(prev_command.shape) == 2)
        expanded_prev_command = prev_command.unsqueeze(1).expand(-1, num_agents, -1) # [N*M, A, command_dim]

        # agents: obs + command -> action_logits + msg
        obs_by_agent = obs.view(obs.shape[0], num_agents, self.obs_per_agent) # [N * M, A, obs_per_agent]
        agent_input = torch.cat([expanded_prev_command, obs_by_agent], dim=-1) # [N * M, A, command_dim + obs_per_agent]
        agent_out = self.agent_mlp(agent_input) # [N * M, A, action_logits_dim + agent_msg_dim]
        action_logits = agent_out[:, :, :self.action_logits_dim] # [N * M, A, action_logits_dim]
        agent_msg = agent_out[:, :, self.action_logits_dim:] # [N * M, A, agent_msg_dim]

        # from here, we're computing the command for the next step (at least up to and including lstm, where we record hidden state)
        
        # pool messages
        pooled_msg, _ = self.attn(self.query.expand(obs.shape[0], -1, -1), agent_msg, agent_msg) # [N * M, 1, agent_msg_dim]
        pooled_msg = pooled_msg.view(obs.shape[0], -1) # [N * M, agent_msg_dim]
        
        # mlp pooled messages
        pooled_msg = self.pooled_msg_mlp(pooled_msg) # [N * M, pooled_msg_dim]
        # assert(pooled_msg.shape[0] == obs.shape[0])
        # assert(pooled_msg.shape[1] == self.pooled_msg_dim)
        # assert(len(pooled_msg.shape) == 2)
        
        # lstm
        # unsqueeze gives sequence length of 1
        _, (h_new, c_new) = self.lstm(pooled_msg.unsqueeze(1), (h_curr, c_curr)) # _, 2 of [1, N * M, lstm_hidden_size]
        new_rnn_states = torch.stack([h_new, c_new], dim=0)

        # no need to compute the command here; we don't use it until next time step anyway.
        # we'll just recover it with the saved hidden state on the next forward pass.

        # assert(action_logits.shape[0] == obs.shape[0])
        # assert(action_logits.shape[1] == num_agents)
        # assert(action_logits.shape[2] == self.action_logits_dim)
        # assert(len(action_logits.shape) == 3)
        flattened_action_logits = action_logits.reshape(action_logits.shape[0], -1) # [N * M, A * action_logits_dim]
        return flattened_action_logits, new_rnn_states

    # call forward repeatedly
    def fwd_sequence(self, in_sequences, start_hidden, sequence_breaks):
        seq_len = in_sequences.shape[0]

        hidden_dim_per_layer = start_hidden.shape[-1]

        zero_hidden = torch.zeros((2, 1, 1,
                                   hidden_dim_per_layer),
                                  device=start_hidden.device,
                                  dtype=start_hidden.dtype)

        out_sequences = []

        cur_hidden = start_hidden
        for i in range(seq_len):
            cur_features = in_sequences[i]
            cur_breaks = sequence_breaks[i]

            out, new_hidden = self.forward(cur_features, cur_hidden)
            out_sequences.append(out)

            cur_hidden = torch.where(
                cur_breaks.bool(), zero_hidden, new_hidden)

        return torch.stack(out_sequences, dim=0)

class RecurrentAttentionCriticEncoder(nn.Module):
    def __init__(self,
                 obs_per_agent,
                 agent_msg_dim,
                 agent_msg_mlp_num_layers,
                 num_attn_heads,
                 pooled_msg_dim,
                 pooled_msg_mlp_num_layers,
                 lstm_hidden_size,
                 out_mlp_num_layers,
                 num_critic_channels,
                 ):
        super().__init__()
        self.obs_per_agent = obs_per_agent
        self.agent_msg_dim = agent_msg_dim
        self.agent_msg_mlp_num_layers = agent_msg_mlp_num_layers
        self.num_attn_heads = num_attn_heads
        self.pooled_msg_dim = pooled_msg_dim
        self.pooled_msg_mlp_num_layers = pooled_msg_mlp_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.num_critic_channels = num_critic_channels
        self.out_mlp_num_layers = out_mlp_num_layers

        
        self.hidden_shape = (2, 1, self.lstm_hidden_size)

        self.agent_msg_mlp = MLP(
            input_dim = obs_per_agent,
            num_channels = agent_msg_dim,
            num_layers = agent_msg_mlp_num_layers
        )

        self.query = nn.Parameter(torch.rand(1, 1, agent_msg_dim))
        self.attn = nn.MultiheadAttention(agent_msg_dim, num_heads=num_attn_heads, batch_first=True)

        self.pooled_msg_mlp = MLP(
            input_dim = agent_msg_dim,
            num_channels = pooled_msg_dim,
            num_layers = pooled_msg_mlp_num_layers
        )

        self.lstm = nn.LSTM(
            input_size = pooled_msg_dim,
            hidden_size = lstm_hidden_size,
            batch_first = True
        )

        self.out_mlp = MLP(
            input_dim = lstm_hidden_size,
            num_channels = num_critic_channels,
            num_layers = out_mlp_num_layers
        )

    def forward(self, obs, rnn_states):        
        # # obs: [N * M, A * -1]
        # assert(len(obs.shape) == 2)
        # assert(obs.shape[1] % self.obs_per_agent == 0)
        num_agents = obs.shape[1] // self.obs_per_agent

        h_curr = rnn_states[0, ...] # [num_layers, N * M, lstm_hidden_size]
        c_curr = rnn_states[1, ...] # [num_layers, N * M, lstm_hidden_size]

        # assert(h_curr.shape[0] == 1)
        # assert(c_curr.shape[0] == 1)

        # assert(h_curr.shape[1] == obs.shape[0])
        # assert(c_curr.shape[1] == obs.shape[0])

        # agents: obs -> msg
        obs_by_agent = obs.view(obs.shape[0], num_agents, self.obs_per_agent) # [N * M, A, obs_per_agent]
        agent_msg = self.agent_msg_mlp(obs_by_agent) # [N * M, A, agent_msg_dim]
        
        # pool messages
        pooled_msg, _ = self.attn(self.query.expand(obs.shape[0], -1, -1), agent_msg, agent_msg) # [N * M, 1, agent_msg_dim]
        pooled_msg = pooled_msg.view(obs.shape[0], -1) # [N * M, agent_msg_dim]
        
        # mlp pooled messages
        pooled_msg = self.pooled_msg_mlp(pooled_msg) # [N * M, pooled_msg_dim]
        # assert(pooled_msg.shape[0] == obs.shape[0])
        # assert(pooled_msg.shape[1] == self.pooled_msg_dim)
        # assert(len(pooled_msg.shape) == 2)
        
        # lstm
        # unsqueeze gives sequence length of 1
        _, (h_new, c_new) = self.lstm(pooled_msg.unsqueeze(1), (h_curr, c_curr)) # _, 2 of [1, N * M, lstm_hidden_size]
        new_rnn_states = torch.stack([h_new, c_new], dim=0)
        lstm_out = h_new[-1, ...] # hidden state of last layer

        out = self.out_mlp(lstm_out)
        return out, new_rnn_states

    # call forward repeatedly
    def fwd_sequence(self, in_sequences, start_hidden, sequence_breaks):
        seq_len = in_sequences.shape[0]

        hidden_dim_per_layer = start_hidden.shape[-1]

        zero_hidden = torch.zeros((2, 1, 1,
                                   hidden_dim_per_layer),
                                  device=start_hidden.device,
                                  dtype=start_hidden.dtype)

        out_sequences = []

        cur_hidden = start_hidden
        for i in range(seq_len):
            cur_features = in_sequences[i]
            cur_breaks = sequence_breaks[i]

            out, new_hidden = self.forward(cur_features, cur_hidden)
            out_sequences.append(out)

            cur_hidden = torch.where(
                cur_breaks.bool(), zero_hidden, new_hidden)

        return torch.stack(out_sequences, dim=0)