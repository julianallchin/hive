from .ant_comm_block import AntCommBlock
from .actor_critic      import Backbone, RecurrentStateConfig
import torch

class HiveBackbone(Backbone):
    def __init__(self, process_obs, base_obs_dim, cfg):
        """
        base_obs_dim : dimensionality from `process_obs` (without cmd)
        cfg          : ModelConfig
        """
        super().__init__()
        self.process_obs = process_obs
        self.cmd_dim     = cfg.cmd_dim

        obs_dim_with_cmd = base_obs_dim + cfg.cmd_dim

        self.actor_blk  = AntCommBlock(obs_dim_with_cmd, cfg.out_dim,
                                       cfg.msg_dim, cfg.ant_trunk_hid_dim,
                                       cfg.heads, cfg.lstm_dim,
                                       cfg.cmd_dim)

        self.critic_blk = AntCommBlock(obs_dim_with_cmd, 0,  # out_dim = 0 !
                                       cfg.msg_dim, cfg.ant_trunk_hid_dim,
                                       cfg.heads, cfg.lstm_dim,
                                       cfg.cmd_dim)

        # RNN state: (actor_LSTM, critic_LSTM)
        self.recurrent_cfg = RecurrentStateConfig([
            self.actor_blk.hidden_shape,
            self.critic_blk.hidden_shape,
        ])

    # ---------------------------------------------------------------
    def _split_state(self, states):
        return (states[0] if states else None,
                states[1] if states else None)

    def _compute_cmd_prev(self, lstm_state):
        """cmd_{t-1} = cmd_head(h_{t-1}); if state is None -> zeros"""
        if lstm_state is None:
            return torch.zeros(1, self.cmd_dim, device='cpu')
        h_prev = lstm_state[0, 0]            # shape [B, H]
        return self.actor_blk.cmd_head(h_prev)  # (actor & critic share cmd)

    # ---------------------------------------------------------------
    def forward(self, rnn_states, *obs):
        base, active  = self.process_obs(*obs)      # [B*N, base_obs_dim]
        B      = base.shape[0] // active.shape[0]  # actually 1 here, but ok
        N_ants = active.shape[0] // (base.shape[0] // B)

        cmd_prev = self._compute_cmd_prev(self._split_state(rnn_states)[0])
        cmd_rep  = cmd_prev.repeat_interleave(N_ants, dim=0)  # [B*N, cmd_dim]

        flat_in  = torch.cat([base, cmd_rep], dim=1)

        logits, _, new_a_state = self.actor_blk(
            self._split_state(rnn_states)[0],
            flat_in, active, N_ants)

        _, critic_value_feat, _, new_c_state = self.critic_blk(
            self._split_state(rnn_states)[1],
            flat_in, active, N_ants)

        return logits, critic_value_feat, (new_a_state, new_c_state)

    # fwd_actor_only / fwd_critic_only / fwd_rollout / fwd_sequence
    # replicate the same pattern: build cmd_prev, concat, call just
    # the relevant block, return the needed feature(s).

    def fwd_actor_only(self, rnn_states_out, rnn_states_in, *obs_in):
        base, active = self.process_obs(*obs_in)      # [B*N, base_obs_dim]
        B = base.shape[0] // active.shape[0]  # actually 1 here, but ok
        N_ants = active.shape[0] // (base.shape[0] // B)

        # Get previous command from LSTM state
        cmd_prev = self._compute_cmd_prev(self._split_state(rnn_states_in)[0])
        cmd_rep = cmd_prev.repeat_interleave(N_ants, dim=0)  # [B*N, cmd_dim]

        # Combine observation with command
        flat_in = torch.cat([base, cmd_rep], dim=1)

        # Only run actor block
        logits, _, new_a_state = self.actor_blk(
            self._split_state(rnn_states_in)[0],
            flat_in, active, N_ants)

        # Update RNN state if output is provided
        if rnn_states_out is not None:
            rnn_states_out[0][...] = new_a_state
            # Keep critic state unchanged
            if len(rnn_states_out) > 1 and rnn_states_in is not None and len(rnn_states_in) > 1:
                rnn_states_out[1][...] = rnn_states_in[1]

        return logits

    def fwd_critic_only(self, rnn_states_out, rnn_states_in, *obs_in):
        base, active = self.process_obs(*obs_in)      # [B*N, base_obs_dim]
        B = base.shape[0] // active.shape[0]  # actually 1 here, but ok
        N_ants = active.shape[0] // (base.shape[0] // B)

        # Get previous command from LSTM state
        cmd_prev = self._compute_cmd_prev(self._split_state(rnn_states_in)[0])
        cmd_rep = cmd_prev.repeat_interleave(N_ants, dim=0)  # [B*N, cmd_dim]

        # Combine observation with command
        flat_in = torch.cat([base, cmd_rep], dim=1)

        # Only run critic block
        _, critic_value_feat, _, new_c_state = self.critic_blk(
            self._split_state(rnn_states_in)[1],
            flat_in, active, N_ants)

        # Update RNN state if output is provided
        if rnn_states_out is not None:
            if len(rnn_states_out) > 1 and rnn_states_in is not None and len(rnn_states_in) > 0:
                rnn_states_out[0][...] = rnn_states_in[0]  # Keep actor state unchanged
            rnn_states_out[1][...] = new_c_state

        return critic_value_feat

    def fwd_rollout(self, rnn_states_out, rnn_states_in, *obs_in):
        base, active = self.process_obs(*obs_in)      # [B*N, base_obs_dim]
        B = base.shape[0] // active.shape[0]  # actually 1 here, but ok
        N_ants = active.shape[0] // (base.shape[0] // B)

        # Get previous command from LSTM state
        cmd_prev = self._compute_cmd_prev(self._split_state(rnn_states_in)[0])
        cmd_rep = cmd_prev.repeat_interleave(N_ants, dim=0)  # [B*N, cmd_dim]

        # Combine observation with command
        flat_in = torch.cat([base, cmd_rep], dim=1)

        # Run both blocks
        actor_state, critic_state = self._split_state(rnn_states_in)
        
        logits, _, new_a_state = self.actor_blk(
            actor_state, flat_in, active, N_ants)
            
        _, critic_value_feat, _, new_c_state = self.critic_blk(
            critic_state, flat_in, active, N_ants)

        # Update RNN state if output is provided
        if rnn_states_out is not None:
            rnn_states_out[0][...] = new_a_state
            rnn_states_out[1][...] = new_c_state

        return logits, critic_value_feat

    def fwd_sequence(self, rnn_start_states, dones, *obs_in):
        # Process observation sequences
        # obs_in has shape [T, B*N, ...]
        T = obs_in[0].shape[0]  # Sequence length
        
        # Initialize outputs
        actor_features_seq = []
        critic_features_seq = []
        
        # Split initial states
        actor_state, critic_state = self._split_state(rnn_start_states)
        
        # Process the sequence step by step
        for t in range(T):
            # Get observations at this timestep
            obs_t = [o[t] for o in obs_in]
            base, active = self.process_obs(*obs_t)  # [B*N, base_obs_dim]
            B = base.shape[0] // active.shape[0]
            N_ants = active.shape[0] // (base.shape[0] // B)
            
            # Get command from current LSTM state
            cmd_prev = self._compute_cmd_prev(actor_state)
            cmd_rep = cmd_prev.repeat_interleave(N_ants, dim=0)  # [B*N, cmd_dim]
            
            # Combine observation with command
            flat_in = torch.cat([base, cmd_rep], dim=1)
            
            # Run both blocks
            logits, _, new_a_state = self.actor_blk(
                actor_state, flat_in, active, N_ants)
                
            _, critic_value_feat, _, new_c_state = self.critic_blk(
                critic_state, flat_in, active, N_ants)
            
            # Apply done mask to reset states where episodes ended
            done_mask = dones[t].unsqueeze(0).unsqueeze(-1)
            # TODO: need to add active agent mask
            
            # Reset states where episodes ended
            if actor_state is not None:
                actor_state = torch.where(
                    done_mask, torch.zeros_like(new_a_state), new_a_state)
            else:
                actor_state = new_a_state
                
            if critic_state is not None:
                critic_state = torch.where(
                    done_mask, torch.zeros_like(new_c_state), new_c_state)
            else:
                critic_state = new_c_state
            
            # Store outputs for this timestep
            actor_features_seq.append(logits)
            critic_features_seq.append(critic_value_feat)
        
        # Stack outputs along time dimension
        actor_features = torch.stack(actor_features_seq, dim=0)
        critic_features = torch.stack(critic_features_seq, dim=0)
        
        return actor_features, critic_features, (actor_state, critic_state)
    