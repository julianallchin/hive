import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .action import DiscreteActionDistributions
from .profile import profile
from .models import AntMLP, HivemindAttention
from .rnn import LSTM

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

    def _flatten_obs_sequence(self, obs):
        return [o.view(-1, *o.shape[2:]) for o in obs]

    def forward(self, rnn_states_in, *obs_in):
        raise NotImplementedError

    def fwd_actor_only(self, rnn_states_out, rnn_states_in, *obs_in):
        raise NotImplementedError

    def fwd_critic_only(self, rnn_states_out, rnn_states_in, *obs_in):
        raise NotImplementedError

    def fwd_rollout(self, rnn_states_out, rnn_states_in, *obs_in):
        raise NotImplementedError

    def fwd_sequence(self, rnn_start_states, dones, *obs_in):
        raise NotImplementedError


class DiscreteActor(nn.Module):
    def __init__(self, actions_num_buckets, impl):
        super().__init__()

        self.actions_num_buckets = actions_num_buckets
        self.impl = impl

    def forward(self, features_in):
        logits = self.impl(features_in)

        return DiscreteActionDistributions(
                self.actions_num_buckets, logits=logits)


class Critic(nn.Module):
    def __init__(self, impl):
        super().__init__()
        self.impl = impl 

    def forward(self, features_in):
        return self.impl(features_in)


class RecurrentStateConfig:
    def __init__(self, shapes):
        self.shapes = shapes
    

class ActorCritic(nn.Module):
    def __init__(self, backbone, actor, critic):
        super().__init__()

        self.backbone = backbone 
        self.recurrent_cfg = backbone.recurrent_cfg
        self.actor = actor
        self.critic = critic

    # Direct call intended for debugging only, should use below
    # specialized functions
    def forward(self, rnn_states, *obs):
        actor_features, critic_features, new_rnn_states = self.backbone(
            rnn_states, *obs)

        action_dists = self.actor(actor_features)
        values = self.critic(critic_features)

        return action_dists, values, new_rnn_states

    def fwd_actor(self, actions_out, rnn_states_out, rnn_states_in, *obs_in):
        actor_features = self.backbone.fwd_actor_only(
            rnn_states_out, rnn_states_in, *obs_in)

        action_dists = self.actor(actor_features)
        action_dists.best(out=actions_out)

    def fwd_critic(self, values_out, rnn_states_out, rnn_states_in, *obs_in):
        features = self.backbone.fwd_critic_only(
            rnn_states_out, rnn_states_in, *obs_in)
        values_out[...] = self.critic(features)

    def fwd_rollout(self, actions_out, log_probs_out, values_out,
                      rnn_states_out, rnn_states_in, *obs_in):
        actor_features, critic_features = self.backbone.fwd_rollout(
            rnn_states_out, rnn_states_in, *obs_in)

        action_dists = self.actor(actor_features)
        values = self.critic(critic_features)

        action_dists.sample(actions_out, log_probs_out)
        values_out[...] = values

    def fwd_update(self, rnn_states, sequence_breaks, rollout_actions, *obs):
        actor_features, critic_features = self.backbone.fwd_sequence(
            rnn_states, sequence_breaks, *obs)

        action_dists = self.actor(actor_features)
        values = self.critic(critic_features)

        T, N = rollout_actions.shape[0:2]
        flattened_actions = rollout_actions.view(
            T * N, *rollout_actions.shape[2:])

        log_probs, entropies = action_dists.action_stats(flattened_actions)

        log_probs = log_probs.view(T, N, *log_probs.shape[1:])
        entropies = entropies.view(T, N, *entropies.shape[1:])
        values = values.view(T, N, *values.shape[1:])

        return log_probs, entropies, values

class BackboneEncoder(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.rnn_state_shape = None

    def forward(self, rnn_states, *inputs):
        features = self.net(*inputs)
        return features, None

    def fwd_inplace(self, rnn_states_out, rnn_states_in, *inputs):
        return self.net(*inputs)

    # *inputs come in pre-flattened
    def fwd_sequence(self, rnn_start_states,
                     sequence_breaks, *flattened_inputs):
        return self.net(*flattened_inputs)

class RecurrentBackboneEncoder(nn.Module):
    def __init__(self, net, rnn):
        super().__init__()
        self.net = net
        self.rnn = rnn
        self.rnn_state_shape = rnn.hidden_shape

    def forward(self, rnn_states_in, *inputs):
        features = self.net(*inputs)

        rnn_out, new_rnn_states = self.rnn(features, rnn_states_in)

        return rnn_out, new_rnn_states

    def fwd_inplace(self, rnn_states_out, rnn_states_in, *inputs):
        features = self.net(*inputs)
        rnn_out, new_rnn_states = self.rnn(features, rnn_states_in)

        # FIXME: proper inplace
        if rnn_states_out != None:
            rnn_states_out[...] = rnn_states_in

        return rnn_out

    # *inputs come in pre-flattened
    def fwd_sequence(self, rnn_start_states, sequence_breaks,
                     *flattened_inputs):
        features = self.net(*flattened_inputs)
        features_seq = features.view(
            *sequence_breaks.shape[0:2], *features.shape[1:])

        with profile('rnn.fwd_sequence'):
            rnn_out_seq = self.rnn.fwd_sequence(
                features_seq, rnn_start_states, sequence_breaks)

        rnn_out_flattened = rnn_out_seq.view(-1, *rnn_out_seq.shape[2:])
        return rnn_out_flattened

class BackboneShared(Backbone):
    def __init__(self, process_obs, encoder):
        super().__init__()
        self.process_obs = process_obs
        self.encoder = encoder 

        if encoder.rnn_state_shape:
            self.recurrent_cfg = RecurrentStateConfig([encoder.rnn_state_shape])
            self.extract_rnn_state = lambda x: x[0] if x != None else None
            self.package_rnn_state = lambda x: (x,)
        else:
            self.recurrent_cfg = RecurrentStateConfig([])
            self.extract_rnn_state = lambda x: None
            self.package_rnn_state = lambda x: ()

    def forward(self, rnn_states_in, *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        features, new_rnn_states = self.encoder(
            self.extract_rnn_state(rnn_states_in), processed_obs)
        return features, features, self.package_rnn_state(new_rnn_states)

    def _rollout_common(self, rnn_states_out, rnn_states_in, *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        return self.encoder.fwd_inplace(
            self.extract_rnn_state(rnn_states_out),
            self.extract_rnn_state(rnn_states_in),
            processed_obs,
        )

    def fwd_actor_only(self, rnn_states_out, rnn_states_in, *obs_in):
        return self._rollout_common(
            rnn_states_out, rnn_states_in, *obs_in)

    def fwd_critic_only(self, rnn_states_out, rnn_states_in, *obs_in):
        return self._rollout_common(
            rnn_states_out, rnn_states_in, *obs_in)

    def fwd_rollout(self, rnn_states_out, rnn_states_in, *obs_in):
        features = self._rollout_common(
            rnn_states_out, rnn_states_in, *obs_in)

        return features, features

    def fwd_sequence(self, rnn_start_states, sequence_breaks, *obs_in):
        with torch.no_grad():
            flattened_obs = self._flatten_obs_sequence(obs_in)
            processed_obs = self.process_obs(*flattened_obs)
        
        features = self.encoder.fwd_sequence(
            self.extract_rnn_state(rnn_start_states),
            sequence_breaks, processed_obs)

        return features, features


class BackboneSeparate(Backbone):
    def __init__(self, process_obs, actor_encoder, critic_encoder):
        super().__init__()
        self.process_obs = process_obs
        self.actor_encoder = actor_encoder
        self.critic_encoder = critic_encoder

        rnn_state_shapes = []

        if actor_encoder.rnn_state_shape == None:
            self.extract_actor_rnn_state = lambda rnn_states: None
        else:
            actor_rnn_idx = len(rnn_state_shapes)
            rnn_state_shapes.append(actor_encoder.rnn_state_shape)
            self.extract_actor_rnn_state = \
                lambda rnn_states: rnn_states[actor_rnn_idx]

        if critic_encoder.rnn_state_shape == None:
            self.extract_critic_rnn_state = lambda rnn_states: None
        else:
            critic_rnn_idx = len(rnn_state_shapes)
            rnn_state_shapes.append(critic_encoder.rnn_state_shape)
            self.extract_critic_rnn_state = \
                lambda rnn_states: rnn_states[critic_rnn_idx]

        if (actor_encoder.rnn_state_shape and
                critic_encoder.rnn_state_shape):
            self.package_rnn_states = lambda a, c: (a, c)
        elif actor_encoder.rnn_state_shape:
            self.package_rnn_states = lambda a, c: (a,)
        elif critic_encoder.rnn_state_shape:
            self.package_rnn_states = lambda a, c: (c,)
        else:
            self.package_rnn_states = lambda a, c: ()

        self.recurrent_cfg = RecurrentStateConfig(rnn_state_shapes)

    def forward(self, rnn_states, *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        actor_rnn_state = self.extract_actor_rnn_state(rnn_states)
        critic_rnn_state = self.extract_critic_rnn_state(rnn_states)

        actor_features, new_actor_rnn_state = self.actor_encoder(
            actor_rnn_state, processed_obs)
        critic_features, new_critic_rnn_state = self.critic_encoder(
            critic_rnn_state, processed_obs)

        return (actor_features, critic_features,
                self.package_rnn_states(new_actor_rnn_state,
                                        new_critic_rnn_state))

    def _rollout_common(self, rnn_states_out, rnn_states_in,
                        *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        actor_rnn_state_in = self.extract_actor_rnn_state(rnn_states_in)
        critic_rnn_state_in = self.extract_critic_rnn_state(rnn_states_in)

        actor_features = self.actor_encoder.fwd_inplace(
            None, actor_rnn_state_in, processed_obs)
        critic_features = self.critic_encoder.fwd_inplace(
            None, critic_rnn_state_in, processed_obs)

        if rnn_states_out != None:
            rnn_states_out[...] = rnn_states_in

        return actor_features, critic_features

    def fwd_actor_only(self, rnn_states_out, rnn_states_in,
                       *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        actor_rnn_state = self.extract_actor_rnn_state(rnn_states_in)
        actor_features = self.actor_encoder.fwd_inplace(
            None, actor_rnn_state, processed_obs)

        if rnn_states_out != None:
            rnn_states_out[...] = rnn_states_in

        return actor_features

    def fwd_critic_only(self, rnn_states_out, rnn_states_in,
                        *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        critic_rnn_state = self.extract_critic_rnn_state(rnn_states_in)
        critic_features = self.critic_encoder.fwd_inplace(
            None, critic_rnn_state, processed_obs)

        if rnn_states_out != None:
            rnn_states_out[...] = rnn_states_in

        return critic_features

    def fwd_rollout(self, rnn_states_out, rnn_states_in, *obs_in):
        actor_features, critic_features = self._rollout_common(
            rnn_states_out, rnn_states_in, *obs_in)

        return actor_features, critic_features

    def fwd_sequence(self, rnn_start_states, sequence_breaks, *obs_in):
        with torch.no_grad():
            flattened_obs = self._flatten_obs_sequence(obs_in)
            processed_obs = self.process_obs(*flattened_obs)
        
        actor_features = self.actor_encoder.fwd_sequence(
            self.extract_actor_rnn_state(rnn_start_states),
            sequence_breaks, processed_obs)

        critic_features = self.critic_encoder.fwd_sequence(
            self.extract_critic_rnn_state(rnn_start_states),
            sequence_breaks, processed_obs)

        return actor_features, critic_features

class HiveBackbone(Backbone):
    """Backbone for the hivemind architecture that coordinates multiple ants
    
    This backbone implements the hierarchical RL architecture where:
    1. Each ant processes its local observations with command from hivemind
    2. Ants generate actions and messages to the hivemind
    3. Hivemind aggregates messages through attention
    4. Hivemind processes global information through LSTM
    5. Hivemind generates command for next timestep
    """
    def __init__(self, num_ants, ant_local_obs_dim, command_dim, ant_mlp_hidden, 
                 phys_action_total_logits, message_dim,
                 attn_heads, attn_output_dim,
                 lstm_hidden_dim, lstm_layers,
                 ant_mlp_module=None, hive_attention_module=None, 
                 hive_lstm_module=None, command_mlp_module=None):
        super().__init__()

        self.num_ants = num_ants  # Number of ants per simulation

        # Store dimensions for command generation
        self._command_dim = command_dim
        self._lstm_hidden_dim = lstm_hidden_dim

        # Initialize or use provided ant MLP module
        self.ant_mlp = ant_mlp_module if ant_mlp_module else \
            AntMLP(ant_local_obs_dim, command_dim, ant_mlp_hidden,
                   phys_action_total_logits, message_dim)

        # Initialize or use provided attention module for hivemind
        self.hive_attention = hive_attention_module if hive_attention_module else \
            HivemindAttention(message_dim, attn_heads, attn_output_dim)
        
        # Initialize or use provided LSTM for hivemind state tracking
        self.hive_lstm = hive_lstm_module if hive_lstm_module else \
            LSTM(in_channels=attn_output_dim, hidden_channels=lstm_hidden_dim, num_layers=lstm_layers)
        
        # MLP to generate command from LSTM hidden state
        self.command_mlp = command_mlp_module if command_mlp_module else \
            nn.Sequential(
                nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(lstm_hidden_dim // 2, command_dim)
            )

        # Initialize command MLP weights
        for idx, layer in enumerate(self.command_mlp):
            if isinstance(layer, nn.Linear):
                if idx == len(self.command_mlp) - 1:  # Last layer - command output
                    nn.init.orthogonal_(layer.weight, gain=0.1)
                else:  # Hidden layers
                    nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

        # Configure recurrence based on the Hivemind's LSTM
        self.recurrent_cfg = RecurrentStateConfig([self.hive_lstm.hidden_shape])
        
        # Helper methods to handle LSTM state extraction and packaging
        self.extract_lstm_hc = lambda rnn_states_tuple: rnn_states_tuple[0] if rnn_states_tuple else None
        self.package_lstm_hc = lambda hc_tuple: (hc_tuple,)
        
        # Process observation - identity by default, override in subclass if needed
        self.process_obs = lambda *obs_list: obs_list[0]

    def _get_command_from_state(self, lstm_hc_state, batch_size, device):
        """Generate hivemind command from LSTM state"""
        if lstm_hc_state is not None:
            # Extract hidden state from last LSTM layer
            # lstm_hc_state shape: [2, num_layers, batch_size, hidden_dim]
            # h has shape [num_layers, batch_size, hidden_dim]
            h_last_layer = lstm_hc_state[0][-1]  # [batch_size, hidden_dim]
            command = self.command_mlp(h_last_layer)   # [batch_size, command_dim]
        else:
            # Initial step or no recurrent state, use a zero command
            command = torch.zeros(batch_size, self._command_dim, device=device)
        return command

    def _forward_steps(self, current_hive_lstm_hc, ant_obs_input, 
                       rnn_states_out_tuple_for_update=None):
        """Core forward pass for the hivemind architecture"""
        batch_sims = ant_obs_input.shape[0]  # Number of parallel simulations
        device = ant_obs_input.device

        # 1. Generate Command from previous LSTM state
        command_for_ants = self._get_command_from_state(current_hive_lstm_hc, batch_sims, device)
        
        # Expand command for each ant: [BatchSims, CommandDim] -> [BatchSims, NumAnts, CommandDim]
        command_expanded = command_for_ants.unsqueeze(1).expand(-1, self.num_ants, -1)

        # Flatten batch dimensions for AntMLP:
        # ant_obs_input: [BatchSims, NumAnts, ObsDim] -> [BatchSims * NumAnts, ObsDim]
        # command_expanded: [BatchSims, NumAnts, CommandDim] -> [BatchSims * NumAnts, CommandDim]
        flat_ant_obs = ant_obs_input.reshape(batch_sims * self.num_ants, -1)
        flat_command_expanded = command_expanded.reshape(batch_sims * self.num_ants, -1)

        # 2. Ant MLPs: Process observations and generate actions and messages
        flat_phys_action_logits, flat_messages = self.ant_mlp(flat_ant_obs, flat_command_expanded)
        
        # Reshape messages for attention: [BatchSims, NumAnts, MessageDim]
        ant_messages_for_attn = flat_messages.view(batch_sims, self.num_ants, -1)

        # 3. Hivemind Attention: Aggregate ant messages to global message
        global_message = self.hive_attention(ant_messages_for_attn)  # [BatchSims, AttnOutputDim]

        # 4. Hivemind LSTM: Update LSTM state using global message
        # Input to LSTM needs to be [seq_len=1, batch, input_size]
        lstm_input = global_message.unsqueeze(0)  # [1, BatchSims, AttnOutputDim]
        
        lstm_output_feat, next_hive_lstm_hc = self.hive_lstm(lstm_input, current_hive_lstm_hc)
        
        # Actor features are the flattened physical action logits for all ants
        actor_features = flat_phys_action_logits  # [BatchSims * NumAnts, PhysActionLogitsDim]
        
        # Critic features are derived from the hivemind's state - use LSTM output
        # Squeeze seq_len dim: [1, BatchSims, LSTMHiddenDim] -> [BatchSims, LSTMHiddenDim]
        critic_features = lstm_output_feat.squeeze(0)

        # Update rnn_states_out if provided (for fwd_inplace pattern)
        if rnn_states_out_tuple_for_update is not None:
            # Copy the new LSTM state to the output tensor
            rnn_states_out_tuple_for_update[0].copy_(next_hive_lstm_hc)

        return actor_features, critic_features, self.package_lstm_hc(next_hive_lstm_hc)

    def forward(self, rnn_states_in, *obs_in):
        # Extract observations
        ant_obs = self.process_obs(*obs_in)
        
        # Extract current LSTM state from rnn_states_in
        current_lstm_hc = self.extract_lstm_hc(rnn_states_in)
        
        # Forward pass through the hivemind architecture
        actor_features, critic_features, new_rnn_states = self._forward_steps(
            current_lstm_hc, ant_obs)
            
        return actor_features, critic_features, new_rnn_states

    def fwd_rollout(self, rnn_states_out, rnn_states_in, *obs_in):
        # Process observations
        ant_obs = self.process_obs(*obs_in)
        
        # Extract current LSTM state
        current_lstm_hc = self.extract_lstm_hc(rnn_states_in)

        # Forward pass with state update
        actor_features, critic_features, _ = self._forward_steps(
            current_lstm_hc, ant_obs, rnn_states_out)
            
        return actor_features, critic_features

    def fwd_actor_only(self, rnn_states_out, rnn_states_in, *obs_in):
        # Similar to fwd_rollout, but only return actor features
        ant_obs = self.process_obs(*obs_in)
        current_lstm_hc = self.extract_lstm_hc(rnn_states_in)
        
        actor_features, _, _ = self._forward_steps(current_lstm_hc, ant_obs, rnn_states_out)
        return actor_features
        
    def fwd_critic_only(self, rnn_states_out, rnn_states_in, *obs_in):
        # Similar to fwd_rollout, but only return critic features
        ant_obs = self.process_obs(*obs_in)
        current_lstm_hc = self.extract_lstm_hc(rnn_states_in)
        
        _, critic_features, _ = self._forward_steps(current_lstm_hc, ant_obs, rnn_states_out)
        return critic_features

    def fwd_sequence(self, rnn_start_states, sequence_breaks, *obs_in_seq):
        # obs_in_seq[0] should be [T, N_batch, NumAnts, AntObsDim]
        # Directly process without flattening
        ant_obs_seq = self.process_obs(*obs_in_seq)
        
        # Get dimensions
        T, N_batch = ant_obs_seq.shape[0], ant_obs_seq.shape[1]
        device = ant_obs_seq.device

        # Initialize current LSTM state for the batch
        current_lstm_hc_batch = self.extract_lstm_hc(rnn_start_states)

        # Lists to collect features across timesteps
        all_actor_features_flat_steps = []
        all_critic_features_steps = []

        # Process each timestep
        for t in range(T):
            # Get observations for this timestep: [N_batch, NumAnts, AntObsDim]
            ant_obs_t = ant_obs_seq[t]
            
            # Forward step without state update (we'll manually update)
            actor_features_t_flat, critic_features_t, next_lstm_hc_packaged = \
                self._forward_steps(current_lstm_hc_batch, ant_obs_t)
            
            # Collect features for this timestep
            all_actor_features_flat_steps.append(actor_features_t_flat)
            all_critic_features_steps.append(critic_features_t)

            # Extract new LSTM state
            next_lstm_hc = self.extract_lstm_hc(next_lstm_hc_packaged)
            
            # Apply sequence breaks (episode dones) to reset LSTM state
            # Get done mask: [N_batch, 1] -> expanded to LSTM state shape
            done_mask_t = sequence_breaks[t].view(-1, 1, 1, 1)
            done_mask_t = done_mask_t.expand_as(next_lstm_hc)
            
            # Reset LSTM state when done is True
            zeros_like_state = torch.zeros_like(next_lstm_hc)
            current_lstm_hc_batch = torch.where(done_mask_t, zeros_like_state, next_lstm_hc)

        # Concatenate results over time
        # Actor features: List of [N_batch * NumAnts, LogitsDim] -> [T * N_batch * NumAnts, LogitsDim]
        final_actor_features = torch.cat(all_actor_features_flat_steps, dim=0)
        
        # Critic features: List of [N_batch, LSTMHiddenDim] -> [T * N_batch, LSTMHiddenDim]
        final_critic_features = torch.cat(all_critic_features_steps, dim=0)
        
        return final_actor_features, final_critic_features
