import torch, torch.nn as nn, torch.nn.functional as F
from hive_learn.cfg import Consts, ModelConfig

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

        new_hidden = torch.stack([h_t, c_t], dim=0)

        out = h_t.squeeze(0)

        assert out.shape[0] == hidden_state.shape[2]
        assert out.shape[1] == ModelConfig.lstm_dim

        assert new_hidden.shape[0] == 2
        assert new_hidden.shape[1] == 1
        assert new_hidden.shape[2] == hidden_state.shape[2]
        assert new_hidden.shape[3] == ModelConfig.lstm_dim

        return logits, out, new_hidden
