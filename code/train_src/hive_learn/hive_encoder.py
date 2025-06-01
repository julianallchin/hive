
from .hive_block import HiveBlock
from torch import nn
import torch
from hive_learn.cfg import ModelConfig, Consts

class HiveEncoder(nn.Module):
    def __init__(self, obs_dim, pre_act_dim):
        super().__init__()

        self.rnn_state_shape = (2, 1, ModelConfig.lstm_dim) # correct!
        self.pre_act_dim = pre_act_dim

        # Ant MLP -> Attention -> LSTM
        self.hive_block = HiveBlock(obs_dim, pre_act_dim)

        # Command MLP
        self.cmd_head = nn.Sequential(
            nn.Linear(ModelConfig.lstm_dim, ModelConfig.lstm_dim), nn.ReLU(),
            nn.Linear(ModelConfig.lstm_dim, ModelConfig.cmd_dim)
        )

    def _compute_cmd_prev(self, lstm_state):
        """cmd_{t-1} = cmd_head(h_{t-1}); if state is None -> zeros"""
        if lstm_state is None:
            return torch.zeros(1, ModelConfig.cmd_dim, device=lstm_state.device)
        h_prev = lstm_state[0, 0]            # shape [B, H]
        return self.cmd_head(h_prev)  # (actor & critic share cmd)
        
    def forward(self, rnn_states_in, processed_obs):
        """
        rnn_states_in : (2,1,B,H)  from LSTM   (can be None at t=0)
        processed_obs : [B*N, obs_dim]         already contains cmd_{t-1}
        """

        raise NotImplementedError

        obs, active_agents = processed_obs

        # Compute previous command and concat with local obs
        cmd_prev = self._compute_cmd_prev(rnn_states_in)
        flat_in  = torch.cat([obs, cmd_prev], dim=1)

        print("encoder rnn state shape", rnn_states_in.shape)


        # Run ant comm block
        logits, _, new_a_state = self.hive_block(
            rnn_states_in,
            flat_in, active_agents)

        return logits, new_a_state
        
    def fwd_inplace(self, rnn_states_out, rnn_states_in, processed_obs):
        """
        fwd_inplace is forward but it manages the rnn state.
        rnn_states_out : (2,1,B,H)  from LSTM   (can be None at t=0)
        rnn_states_in  : (2,1,B,H)  from LSTM   (can be None at t=0)
        processed_obs         : [B*N, obs_dim]         already contains cmd_{t-1}
        """
        
        # obs [N, A * features per ant]; active agents is [N, A]
        obs, active_agents = processed_obs
        assert obs.shape[0] == active_agents.shape[0]
        assert active_agents.shape[1] == Consts.MAX_AGENTS
        N = obs.shape[0]

        # Reshape obs to [N, A, features per ant]
        obs = obs.view(N, Consts.MAX_AGENTS, -1)

        # cmd_prev should return [N, ModelConfig.cmd_dim]
        cmd_prev = self._compute_cmd_prev(rnn_states_in)
        assert cmd_prev.shape[0] == N

        # Reshape cmd_prev to [N, 1, ModelConfig.cmd_dim]
        reshaped_cmd_prev = cmd_prev.unsqueeze(1)
        
        # repeat along the agent dimension
        expanded_cmd_prev = reshaped_cmd_prev.expand(N, Consts.MAX_AGENTS, -1)

        # now concat the obs and the command
        flat_in = torch.cat([obs, expanded_cmd_prev], dim=2)
        assert flat_in.shape[0] == N
        assert flat_in.shape[1] == Consts.MAX_AGENTS
        assert flat_in.shape[2] == obs.shape[2] + ModelConfig.cmd_dim

        logits, lstm_hidden, new_a_state = self.hive_block(
            rnn_states_in,
            flat_in, active_agents.unsqueeze(-1))

        if rnn_states_out is not None:
            rnn_states_out[...] = new_a_state

        if self.pre_act_dim > 0:
            return logits
        else:
            return lstm_hidden

    def fwd_sequence(self, rnn_start_states, sequence_breaks,
                     flattened_processed_obs):

        raise NotImplementedError
        
        # cmds_prev = self._compute_cmd_prev(rnn_start_sta
        obs, active_agents = flattened_processed_obs

        # Reshape flattened_processed_obs to [seq len, num of seqs, obs dim]
        flattened_processed_obs_seq = obs.view(
            *sequence_breaks.shape[0:2], *obs.shape[1:])

        seq_len = flattened_processed_obs_seq.shape[0]

        hidden_dim_per_layer = rnn_start_states.shape[-1]

        zero_hidden = torch.zeros((2, self.num_layers, 1,
                                   hidden_dim_per_layer),
                                  device=rnn_start_states.device,
                                  dtype=rnn_start_states.dtype)

        out_sequences = []

        cur_hidden = rnn_start_states
        for i in range(seq_len):
            current_obs = flattened_processed_obs_seq[i]
            current_breaks = sequence_breaks[i]
            
            logits, new_hidden = self.forward(cur_hidden, current_obs)

            out_sequences.append(logits)

            cur_hidden = torch.where(
                current_breaks.bool(), zero_hidden, new_hidden)

        stacked_logits = torch.stack(out_sequences, dim=0)
        return stacked_logits.view(-1, *stacked_logits.shape[2:])
            


            
            
        
        

        