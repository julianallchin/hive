
from .ant_comm_block import AntCommBlock
from torch import nn
import torch

class HiveEncoder(nn.Module):
    def __init__(self, obs_dim, out_dim, msg_dim,
                 ant_trunk_hid_dim, heads, lstm_dim,
                 cmd_dim):
        super().__init__()

        self.rnn_state_shape = (2, 1, lstm_dim) # correct!

        # Ant MLP -> Attention -> LSTM
        self.ant_comm_block = AntCommBlock(obs_dim, out_dim, msg_dim,
                 ant_trunk_hid_dim, heads, lstm_dim,
                 cmd_dim)

        # Command MLP
        self.cmd_head = nn.Sequential(
            nn.Linear(lstm_dim, lstm_dim), nn.ReLU(),
            nn.Linear(lstm_dim, cmd_dim)
        )

    def _compute_cmd_prev(self, lstm_state):
        """cmd_{t-1} = cmd_head(h_{t-1}); if state is None -> zeros"""
        if lstm_state is None:
            return torch.zeros(1, self.cmd_dim, device=lstm_state.device)
        h_prev = lstm_state[0, 0]            # shape [B, H]
        return self.cmd_head(h_prev)  # (actor & critic share cmd)
        
    def forward(self, rnn_states_in, processed_obs):
        """
        rnn_states_in : (2,1,B,H)  from LSTM   (can be None at t=0)
        processed_obs : [B*N, obs_dim]         already contains cmd_{t-1}
        """

        obs, active_agents = processed_obs

        # Compute previous command and concat with local obs
        cmd_prev = self._compute_cmd_prev(rnn_states_in)
        flat_in  = torch.cat([obs, cmd_prev], dim=1)

        print("encoder rnn state shape", rnn_states_in.shape)


        # Run ant comm block
        logits, _, new_a_state = self.ant_comm_block(
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
        obs, active_agents = processed_obs
        
        cmd_prev = self._compute_cmd_prev(rnn_states_in)
        flat_in  = torch.cat([obs, cmd_prev], dim=1)

        print("encoder rnn state shape", rnn_states_in.shape)

        logits, _, new_a_state = self.ant_comm_block(
            rnn_states_in,
            flat_in, active_agents)

        print("new a state shape", new_a_state.shape)

        if rnn_states_out is not None:
            rnn_states_out[...] = new_a_state

        return logits

    def fwd_sequence(self, rnn_start_states, sequence_breaks,
                     flattened_processed_obs):
        
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
            


            
            
        
        

        