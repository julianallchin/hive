import torch
import torch.nn as nn
import torch.nn.functional as F

from .action import DiscreteActionDistributions
from .actor_critic import ActorCritic, MaskedDiscreteActor, Critic
from .cfg import ModelConfig, Consts

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

# class LinearLayerDiscreteActor(DiscreteActor):
#     def __init__(self, actions_num_buckets, in_channels):
#         total_action_dim = sum(actions_num_buckets)
#         impl = nn.Linear(in_channels, total_action_dim)

#         super().__init__(actions_num_buckets, impl)

#         nn.init.orthogonal_(self.impl.weight, gain=0.01)
#         nn.init.constant_(self.impl.bias, 0)

class MaskedLinearLayerDiscreteActor(MaskedDiscreteActor):
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

class HiveBlock(nn.Module):
    """
    Per-ant trunk  →  (logits | msg) split  →  attention pool
                   →  global LSTM → head MLP → new command
    * If act_dim == 0 the logits slice is empty (critic-only copy).
    * The block also owns the cmd linear that turns LSTM h_t into c_t.
    """
    def __init__(self, obs_dim, pre_act_dim):
        super().__init__()
        self.pre_act_dim  = pre_act_dim

        # ── per-ant trunk ───────────────────────────────────────────────
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + ModelConfig.cmd_dim, ModelConfig.ant_trunk_hid_dim), nn.ReLU(),
            nn.Linear(ModelConfig.ant_trunk_hid_dim, pre_act_dim + ModelConfig.msg_dim), nn.ReLU()
        )

        # ── comms + memory ─────────────────────────────────────────────
        self.attn_heads = nn.ModuleList([
            nn.MultiheadAttention(ModelConfig.msg_dim, 1, batch_first=True)
            for _ in range(ModelConfig.heads)
        ])

        # [1, 1, msg_dim]
        self.attn_query_param = nn.Parameter(torch.randn(1, 1, ModelConfig.msg_dim))

        self.attn_embed = nn.Linear(ModelConfig.msg_dim * ModelConfig.heads, ModelConfig.aggr_msg_dim)
        
        self.lstm = nn.LSTM(ModelConfig.aggr_msg_dim, ModelConfig.lstm_dim, 1, batch_first=True)


    def forward(self, hidden_state, flat_obs, flat_active):
        """
        hidden_state : (2,1,N,H)  from LSTM   (can be None at t=0)
        flat_obs  : [N, A, obs_dim + cmd_dim] where obs_dim is per ant
        flat_active: [N, A, 1]
        returns: logits, new_hidden: [N, H], [2, 1, N, H]
        """
        # Mask flat_obs
        flat_obs = flat_obs * flat_active

        # Run ants trunk
        y = self.trunk(flat_obs) # [N, A, pre_act_dim + msg_dim]

        assert y.shape[0] == hidden_state.shape[2]
        assert y.shape[1] == Consts.MAX_AGENTS
        assert y.shape[2] == self.pre_act_dim + ModelConfig.msg_dim

        logits = y[:, :, :self.pre_act_dim]
        msg    = y[:, :, self.pre_act_dim:]

        # helper = msg.reshape(y.shape[0], -1)
        # helper2= helper[:, :ModelConfig.lstm_dim]
        # return logits, helper2, torch.zeros(2, 1, y.shape[0], ModelConfig.lstm_dim).to(y.device)

        assert logits.shape[0] == hidden_state.shape[2]
        assert logits.shape[1] == Consts.MAX_AGENTS
        assert logits.shape[2] == self.pre_act_dim

        assert msg.shape[0] == hidden_state.shape[2]
        assert msg.shape[1] == Consts.MAX_AGENTS
        assert msg.shape[2] == ModelConfig.msg_dim

        # flat_active has shape [N, A, 1]. Squeeze to [N, A]:
        active_mask = flat_active.squeeze(-1).bool()

        # key_padding_mask wants True where tokens should be masked, i.e. where active_mask is False:
        key_padding_mask = ~active_mask

        # Run attention
        expanded_query = self.attn_query_param.expand(msg.shape[0], -1, -1)
        outs = []
        for attn_head in self.attn_heads:
            out, _ = attn_head(expanded_query, msg, msg, key_padding_mask=key_padding_mask)
            outs.append(out)
        
        assert outs[0].shape[0] == hidden_state.shape[2] # N
        assert outs[0].shape[1] == 1 # A
        assert outs[0].shape[2] == ModelConfig.msg_dim # msg_dim
        
        # concat the heads
        pooled = torch.cat(outs, dim=2)
        
        assert pooled.shape[0] == hidden_state.shape[2]
        assert pooled.shape[1] == 1
        assert pooled.shape[2] == ModelConfig.msg_dim * ModelConfig.heads

        # embed the pooled message
        pooled = self.attn_embed(pooled)
        
        assert pooled.shape[0] == hidden_state.shape[2]
        assert pooled.shape[1] == 1
        assert pooled.shape[2] == ModelConfig.aggr_msg_dim

        _, (h_t, c_t) = self.lstm(pooled, (hidden_state[0], hidden_state[1]))

        assert h_t.shape[0] == 1
        assert h_t.shape[1] == hidden_state.shape[2]
        assert h_t.shape[2] == ModelConfig.lstm_dim

        assert c_t.shape[0] == 1
        assert c_t.shape[1] == hidden_state.shape[2]
        assert c_t.shape[2] == ModelConfig.lstm_dim

        new_rnn_state = torch.stack([h_t, c_t], dim=0)

        out = h_t.squeeze(0)

        assert out.shape[0] == hidden_state.shape[2]
        assert out.shape[1] == ModelConfig.lstm_dim

        assert new_rnn_state.shape[0] == 2
        assert new_rnn_state.shape[1] == 1
        assert new_rnn_state.shape[2] == hidden_state.shape[2]
        assert new_rnn_state.shape[3] == ModelConfig.lstm_dim

        return logits, out, new_rnn_state

class HiveEncoderRNN(nn.Module):
    def __init__(self, obs_dim, pre_act_dim):
        super().__init__()

        self.num_layers = 1
        self.hidden_shape = (2, 1, ModelConfig.lstm_dim) # correct!
        self.obs_dim = obs_dim
        self.pre_act_dim = pre_act_dim
        

        # Ant MLP -> Attention -> LSTM
        # self.hive_block = HiveBlock(obs_dim, pre_act_dim)

        # # Command MLP
        # self.cmd_head = nn.Sequential(
        #     nn.Linear(ModelConfig.lstm_dim, ModelConfig.lstm_dim), nn.ReLU(),
        #     nn.Linear(ModelConfig.lstm_dim, ModelConfig.cmd_dim)
        # )

        
        if (self.pre_act_dim > 0): # actor
            self.testing_actor_mlp = nn.Sequential(
                nn.Linear(self.obs_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, self.pre_act_dim)
            )
        else:
            self.testing_critic_mlp = nn.Sequential(
                nn.Linear(self.obs_dim * Consts.MAX_AGENTS, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, ModelConfig.lstm_dim)
            )

    # def _compute_cmd_prev(self, lstm_state):
    #     """cmd_{t-1} = cmd_head(h_{t-1}); if state is None -> zeros"""
    #     if lstm_state is None:
    #         return torch.zeros(1, ModelConfig.cmd_dim, device=lstm_state.device)
    #     h_prev = lstm_state[0, 0]            # shape [B, H]
    #     return self.cmd_head(h_prev)  # (actor & critic share cmd)
        
    def forward(self, processed_obs, rnn_states_in):
        """
        processed_obs         : [N, A, obs_dim + 1]    (+1 is for active_agents mask)
        rnn_states_in         : (2,1,B,H)  from LSTM   (can be None at t=0)

        returns:
            primary_output: 
            - [N, A * (pre_act_dim + 1)] (if pre_act_dim > 0; ie actor. +1 is for active_agents mask)
            - [N, H] (if pre_act_dim == 0; ie critic)
            new_a_state: (2,1,B,H)
        """


        # obs [N, A * features per ant]; active agents is [N, A]
        assert processed_obs.shape[1] == Consts.MAX_AGENTS
        assert(processed_obs.shape[2] == self.obs_dim + 1)

        obs = processed_obs[:, :, :self.obs_dim] # [N, A, obs_dim]
        active_agents = processed_obs[:, :, self.obs_dim:] # [N, A, 1]

        assert obs.shape[0] == active_agents.shape[0]
        assert obs.shape[1] == Consts.MAX_AGENTS
        assert active_agents.shape[1] == Consts.MAX_AGENTS
        assert obs.shape[2] == self.obs_dim
        assert active_agents.shape[2] == 1
        assert (active_agents[:, :, 0] == 1).all() # at least one active agent in all envs
        
        
        # TODO: remove (and comment back in real code). temporarily testing simple model.
        assert(active_agents[:, :, :] == 1).all() # TODO: remove (after we add dynamic agent number. temporarily testing that all agents are on)
        if (self.pre_act_dim > 0): # actor
            action_logits = self.testing_actor_mlp(obs) # [N, A, pre_act_dim]
            assert action_logits.shape[0] == obs.shape[0]
            assert action_logits.shape[1] == Consts.MAX_AGENTS
            assert action_logits.shape[2] == self.pre_act_dim

            unflattened_out = torch.cat([action_logits, active_agents], dim=2)

            assert unflattened_out.shape[0] == obs.shape[0]
            assert unflattened_out.shape[1] == Consts.MAX_AGENTS
            assert unflattened_out.shape[2] == self.pre_act_dim + 1

            out = unflattened_out.view(unflattened_out.shape[0], -1)

            assert out.shape[0] == obs.shape[0]
            assert out.shape[1] == Consts.MAX_AGENTS * (self.pre_act_dim + 1)

        else: # critic
            flattened_obs = obs.reshape(obs.shape[0], -1)
            assert flattened_obs.shape[0] == obs.shape[0]
            assert flattened_obs.shape[1] == Consts.MAX_AGENTS * self.obs_dim
            assert(len(flattened_obs.shape) == 2)

            out = self.testing_critic_mlp(flattened_obs) # [N, A * obs_dim] -> [N, H]
            assert out.shape[0] == obs.shape[0]
            assert out.shape[1] == ModelConfig.lstm_dim
            
        new_a_state = torch.zeros(2, 1, obs.shape[0], ModelConfig.lstm_dim, device=obs.device)
        return out, new_a_state
        

        # N = obs.shape[0]

        # # cmd_prev should return [N, ModelConfig.cmd_dim]
        # cmd_prev = self._compute_cmd_prev(rnn_states_in)
        # assert cmd_prev.shape[0] == N
        # assert cmd_prev.shape[1] == ModelConfig.cmd_dim
        
        # # make a copy for each agent
        # expanded_cmd_prev = cmd_prev.view(N, 1, ModelConfig.cmd_dim).expand(N, Consts.MAX_AGENTS, -1)
        # # [N, A, cmd_dim]
        # assert expanded_cmd_prev.shape[0] == N
        # assert expanded_cmd_prev.shape[1] == Consts.MAX_AGENTS
        # assert expanded_cmd_prev.shape[2] == ModelConfig.cmd_dim
        # assert len(expanded_cmd_prev.shape) == 3

        # # now concat the obs and the command
        # flat_in = torch.cat([obs, expanded_cmd_prev], dim=2) # [N, A, obs_dim + cmd_dim]
        # assert flat_in.shape[0] == N
        # assert flat_in.shape[1] == Consts.MAX_AGENTS
        # assert flat_in.shape[2] == self.obs_dim + ModelConfig.cmd_dim

        # logits, lstm_hidden, new_a_state = self.hive_block(
        #     rnn_states_in,
        #     flat_in, active_agents)

        # assert logits.shape[0] == N
        # assert logits.shape[1] == Consts.MAX_AGENTS
        # assert logits.shape[2] == self.pre_act_dim
        # assert len(logits.shape) == 3

        # assert lstm_hidden.shape[0] == N
        # assert lstm_hidden.shape[1] == ModelConfig.lstm_dim
        # assert len(lstm_hidden.shape) == 2

        # assert new_a_state.shape[0] == 2
        # assert new_a_state.shape[1] == 1
        # assert new_a_state.shape[2] == N
        # assert new_a_state.shape[3] == ModelConfig.lstm_dim
        # assert len(new_a_state.shape) == 4

        # if self.pre_act_dim > 0:
        #     primary_output_by_agent = torch.cat([logits, active_agents], dim=2)
        #     primary_output = primary_output_by_agent.view(N, -1) # one (combined) action per world
        #     assert primary_output.shape[0] == N
        #     assert primary_output.shape[1] == Consts.MAX_AGENTS * (self.pre_act_dim + 1)
        #     # shape: [N, A * (pre_act_dim + 1)]. last 1 in last dim is active_agents mask
        # else:
        #     primary_output = lstm_hidden
        #     # shape: [N, H]
        
        # return primary_output, new_a_state

    def fwd_sequence(self, in_sequences, start_hidden, sequence_breaks):
        seq_len = in_sequences.shape[0]

        hidden_dim_per_layer = start_hidden.shape[-1]

        zero_hidden = torch.zeros((2, self.num_layers, 1,
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