import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from .action_utils import (
    get_action_dims,
    sample_actions,
    log_prob_actions,
    flat_to_multi_discrete,
    multi_discrete_to_flat,
    TOTAL_ACTIONS
)

class MLP(nn.Module):
    """Flexible MLP with LayerNorm and ReLU activations"""
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                if layer == self.net[-1]:  # Output layer
                    nn.init.orthogonal_(layer.weight, gain=0.01)
                else:  # Hidden layers
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.net(x)

class RecurrentPPOAgent(nn.Module):
    """
    Recurrent PPO Agent for Madrona environment with multi-discrete action space.
    Uses an MLP for feature extraction and an LSTM for temporal processing.
    """
    def __init__(self, obs_dim, hidden_size=128, lstm_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        
        # Get action space dimensions
        action_dims = get_action_dims()
        self.action_dims = action_dims
        
        # Feature extractor (MLP)
        self.feature_extractor = MLP([obs_dim, 256, hidden_size])
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=False
        )
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            else:
                nn.init.constant_(param, 0)
        
        # Policy head outputs logits for each action dimension
        self.actor = nn.Linear(hidden_size, action_dims['total'])
        
        # Value head
        self.critic = nn.Linear(hidden_size, 1)
        
        # Initialize output layers
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0)
        nn.init.orthogonal_(self.critic.weight)
        nn.init.constant_(self.critic.bias, 0)
    
    def forward(self, x, lstm_state, done=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, obs_dim) or (batch_size, obs_dim)
            lstm_state: Tuple of (hidden_state, cell_state) each of shape (num_layers, batch_size, hidden_size)
            done: Optional tensor of shape (batch_size,) indicating terminal states
        
        Returns:
            action_logits: Logits for the action distribution [batch_size, total_actions]
            value: State value estimate [batch_size]
            new_lstm_state: Updated LSTM state
        """
        # Handle single timestep input
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, batch_size, obs_dim)
        
        # Extract features
        features = self.feature_extractor(x)  # (seq_len, batch_size, hidden_size)
        
        # Process through LSTM
        seq_len, batch_size, _ = features.shape
        
        # Initialize LSTM state if not provided
        if lstm_state is None:
            device = x.device
            h = torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=device)
            c = torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=device)
            lstm_state = (h, c)
        
        # Reset LSTM states for done episodes
        if done is not None:
            h, c = lstm_state
            h = h * (1 - done).view(1, -1, 1)
            c = c * (1 - done).view(1, -1, 1)
            lstm_state = (h, c)
        
        # Forward through LSTM
        lstm_out, new_lstm_state = self.lstm(features, lstm_state)
        
        # Get action logits and value
        action_logits = self.actor(lstm_out)  # (seq_len, batch_size, total_actions)
        value = self.critic(lstm_out).squeeze(-1)  # (seq_len, batch_size)
        
        # Return logits and value for the last timestep
        return action_logits[-1], value[-1], new_lstm_state
    
    def get_action(self, x, lstm_state, done=None):
        """
        Sample an action from the policy.
        
        Args:
            x: Current observation (batch_size, obs_dim)
            lstm_state: Current LSTM state
            done: Optional terminal state indicator
            
        Returns:
            action: Sampled action (batch_size, 4) [move_amount, move_angle, rotate, grab]
            log_prob: Log probability of the action (batch_size,)
            entropy: Entropy of the action distribution (batch_size,)
            value: State value estimate (batch_size,)
            new_lstm_state: Updated LSTM state
        """
        with torch.no_grad():
            # Get action logits and value
            action_logits, value, new_lstm_state = self(x.unsqueeze(0), lstm_state, done)
            
            # Sample actions and compute log probabilities
            actions, log_probs = sample_actions(action_logits)
            
            # Compute entropy of the action distribution
            action_probs = torch.softmax(action_logits, dim=-1)
            action_dists = torch.distributions.Categorical(probs=action_probs)
            entropy = action_dists.entropy()
            
            return (
                actions,  # [batch_size, 4]
                log_probs,  # [batch_size]
                entropy,    # [batch_size]
                value,      # [batch_size]
                new_lstm_state
            )
    
    def get_log_prob_entropy(self, x, action, lstm_state=None, done=None):
        """
        Compute log probability and entropy for given actions.
        
        Args:
            x: Observations (batch_size, obs_dim)
            action: Actions to evaluate (batch_size, 4)
            lstm_state: LSTM state
            done: Optional terminal state indicator
            
        Returns:
            log_prob: Log probability of the actions (batch_size,)
            entropy: Entropy of the action distribution (batch_size,)
        """
        # Get action logits
        action_logits, _, _ = self(x.unsqueeze(0), lstm_state, done)
        
        # Compute log probability of the actions
        log_prob = log_prob_actions(action_logits, action)
        
        # Compute entropy
        action_probs = torch.softmax(action_logits, dim=-1)
        action_dists = torch.distributions.Categorical(probs=action_probs)
        entropy = action_dists.entropy()
        
        return log_prob, entropy
    
    def get_value(self, x, lstm_state, done=None):
        """
        Get the value estimate for a state.
        
        Args:
            x: Observation (batch_size, obs_dim)
            lstm_state: LSTM state
            done: Optional terminal state indicator
            
        Returns:
            value: State value estimate (batch_size,)
        """
        with torch.no_grad():
            _, value, _ = self(x.unsqueeze(0), lstm_state, done)
            return value