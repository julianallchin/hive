import torch, torch.nn as nn, torch.nn.functional as F
from .rnn import LSTM   # same lightweight wrapper you already have

class AntCommBlock(nn.Module):
    """
    Per-ant trunk  →  (logits | msg) split  →  attention pool
                   →  global LSTM → head MLP → new command
    * If act_dim == 0 the logits slice is empty (critic-only copy).
    * The block also owns the cmd linear that turns LSTM h_t into c_t.
    """
    def __init__(self, obs_dim, pre_action_dim, msg_dim,
                 ant_trunk_hid_dim, heads, lstm_dim,
                 cmd_dim):
        super().__init__()
        self.pre_action_dim  = pre_action_dim
        self.msg_dim  = msg_dim
        self.cmd_dim  = cmd_dim

        # ── per-ant trunk ───────────────────────────────────────────────
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, ant_trunk_hid_dim), nn.ReLU(),
            nn.Linear(ant_trunk_hid_dim, pre_action_dim + msg_dim), nn.ReLU()
        )

        # ── comms + memory ─────────────────────────────────────────────
        self.attn = nn.MultiheadAttention(msg_dim, heads, batch_first=True)
        self.lstm = LSTM(msg_dim, lstm_dim, 1)

        self.hidden_shape = (2, 1, lstm_dim)   # for RecurrentStateConfig

    def forward(self, rnn_state, flat_obs, flat_alive, N_ants):
        """
        rnn_state : (2,1,B,H)  from LSTM   (can be None at t=0)
        flat_obs  : [B*N, obs_dim]         already contains cmd_{t-1}
        flat_alive: [B*N, 1]
        """
        y = self.trunk(flat_obs)                       # [B*N, act+msg]

        if self.pre_action_dim:
            logits = y[:, :self.pre_action_dim]               # slice ①
            msg    = y[:,  self.pre_action_dim:]              # slice ②
        else:
            logits = y.new_zeros(y.shape[0], 0)
            msg    = y                                 # critic keeps all

        msg3d = msg.view(-1, N_ants, self.msg_dim)
        pooled, _ = self.attn(msg3d, msg3d, msg3d,
                              key_padding_mask=self._pad_mask(flat_alive, N_ants))
        pooled = pooled.mean(1)                        # [B, msg_dim]

        h_t, c_t = self.lstm(pooled, rnn_state)  # h_t used twice

        return logits, h_t, c_t
