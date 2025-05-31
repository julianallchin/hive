from .ant_comm_block import AntCommBlock
from .actor_critic      import Backbone, RecurrentStateConfig
import torch

class HiveBackbone(Backbone):
    def __init__(self, process_obs, base_obs_dim, cfg):
        """
        base_obs_dim : dimensionality from `process_obs` (without cmd)
        cfg …         act_dim, msg_dim, cmd_dim, hid_dim, heads, lstm_dim, out_dim
        """
        super().__init__()
        self.process_obs = process_obs
        self.cmd_dim     = cfg.cmd_dim

        obs_dim_with_cmd = base_obs_dim + cfg.cmd_dim

        self.actor_blk  = AntCommBlock(obs_dim_with_cmd, cfg.act_dim,
                                       cfg.msg_dim, cfg.hid_dim, cfg.heads,
                                       cfg.lstm_dim, cfg.out_dim, cfg.cmd_dim)

        self.critic_blk = AntCommBlock(obs_dim_with_cmd, 0,  # act_dim = 0 !
                                       cfg.msg_dim, cfg.hid_dim, cfg.heads,
                                       cfg.lstm_dim, cfg.out_dim, cfg.cmd_dim)

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
