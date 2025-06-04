import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.tensordict import TensorDict
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.modules.distributions import NormalParamExtractor

# Assume these are defined from your environment setup
# from your original script:
# device = torch.device("cpu")
# env = VmasEnv(...) # n_agents, observation_spec, action_spec
# n_agents = env.n_agents
# obs_spec_agent = env.observation_spec["agents", "observation"] # Shape [..., n_agents, obs_dim]
# action_spec_agent = env.action_spec[("agents", "action")]     # Shape [..., n_agents, action_dim]

# For demonstration, let's mock these if not running the full script
class MockSpace:
    def __init__(self, shape, low=None, high=None):
        self.shape = torch.Size(shape)
        self.low = low if low is not None else -torch.ones(shape)
        self.high = high if high is not None else torch.ones(shape)

class MockEnv:
    def __init__(self, n_agents, obs_dim, action_dim, device):
        self.n_agents = n_agents
        self.device = device
        self.observation_spec = TensorDict({
            ("agents", "observation"): MockSpace([n_agents, obs_dim])
        }, batch_size=[])
        self.action_spec = TensorDict({
            ("agents", "action"): MockSpace([n_agents, action_dim], low=-1.0, high=1.0)
        }, batch_size=[])
        # For ProbabilisticActor, it needs unbatched_action_spec
        self.unbatched_action_spec = self.action_spec.clone()

# Example Hyperparameters for the new architecture
n_agents_example = 3
obs_dim_example = 18
action_dim_example = 2
device_example = torch.device("cpu")

env = MockEnv(n_agents_example, obs_dim_example, action_dim_example, device_example)
n_agents = env.n_agents
obs_per_agent_dim = env.observation_spec["agents", "observation"].shape[-1]
action_per_agent_dim = env.action_spec[("agents", "action")].shape[-1]


# New hyperparameters for your architecture
message_dim = 64  # Dimension of message from each ant
command_dim = 32  # Dimension of command from central net
attention_heads = 4
central_mlp_hidden_dim = 128
ant_action_mlp_hidden_dim = 128 # For the final ant MLP

# --- 1. Ant Message Generation Network ---
# Each ant processes its observation to create a message
# Input: ("agents", "observation") of shape [batch_size, n_agents, obs_per_agent_dim]
# Output: ("agents", "message") of shape [batch_size, n_agents, message_dim]
ant_message_net = MultiAgentMLP(
    n_agent_inputs=obs_per_agent_dim,
    n_agent_outputs=message_dim,
    n_agents=n_agents,
    share_params=True,
    centralised=False, # Each ant processes its own observation
    depth=2,
    num_cells=64, # Hidden units in each ant's message MLP
    activation_class=nn.Tanh,
    device=device_example,
)
ant_message_module = TensorDictModule(
    module=ant_message_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "message")],
)

# --- 2. Central Command Generation Network ---
# Takes all messages, uses attention to aggregate, then MLP for command
# Input: ("agents", "message") of shape [batch_size, n_agents, message_dim]
# Output: ("central_command") of shape [batch_size, command_dim]
class CentralNet(nn.Module):
    def __init__(self, n_agents, message_dim, command_dim, attention_heads, mlp_hidden_dim, device):
        super().__init__()
        self.n_agents = n_agents
        self.message_dim = message_dim
        self.device = device

        # Attention to aggregate messages
        # Input to attention: (batch_size, n_agents, message_dim)
        # We'll use messages as query, key, and value
        self.attention = nn.MultiheadAttention(
            embed_dim=message_dim, num_heads=attention_heads, batch_first=True, device=device
        )
        # MLP to process aggregated info
        # Option 1: Average attended outputs
        # Option 2: Use a learnable query vector for attention, and its output is the input to MLP
        # Let's do Option 1 for simplicity: average over agents after attention
        self.mlp = nn.Sequential(
            nn.Linear(message_dim, mlp_hidden_dim), # Input is dim of one attended message
            nn.Tanh(),
            nn.Linear(mlp_hidden_dim, command_dim),
        ).to(device)

    def forward(self, messages):
        # messages shape: [batch_size, n_agents, message_dim]
        # MHA expects (batch, seq_len, embed_dim)
        # Here, seq_len is n_agents
        attn_output, _ = self.attention(messages, messages, messages)
        # attn_output shape: [batch_size, n_agents, message_dim]

        # Aggregate: e.g., mean pooling over agents
        aggregated_message = attn_output.mean(dim=1) # Shape: [batch_size, message_dim]

        command = self.mlp(aggregated_message) # Shape: [batch_size, command_dim]
        return command

central_command_net = CentralNet(
    n_agents=n_agents,
    message_dim=message_dim,
    command_dim=command_dim,
    attention_heads=attention_heads,
    mlp_hidden_dim=central_mlp_hidden_dim,
    device=device_example
)
central_command_module = TensorDictModule(
    module=central_command_net,
    in_keys=[("agents", "message")], # from ant_message_module
    out_keys=[("central_command")],  # Global command
)


# --- 3. Ant Action Generation Network ---
# Each ant takes its observation AND the central command to produce action parameters
# Input: ("agents", "observation"), ("central_command")
# Output: ("agents", "loc"), ("agents", "scale") for the distribution
# The MultiAgentMLP will concatenate these inputs for each agent.
# Input per agent: obs_per_agent_dim + command_dim
ant_action_net_input_dim = obs_per_agent_dim + command_dim
ant_action_net = MultiAgentMLP(
    n_agent_inputs=ant_action_net_input_dim,
    n_agent_outputs=action_per_agent_dim * 2, # For loc and scale
    n_agents=n_agents,
    share_params=True,
    centralised=False, # Each ant processes its own combined input
    depth=2,
    num_cells=ant_action_mlp_hidden_dim,
    activation_class=nn.Tanh,
    device=device_example,
)
# We need to ensure the ("central_command") is correctly broadcast and concatenated.
# TensorDictModule can handle multiple in_keys. MultiAgentMLP expects a single
# flat input per agent. We might need a small wrapper or rely on MultiAgentMLP's
# internal processing if it can take a list of specs for concatenation.
# For now, let's assume we'll manually create the concatenated input for it.
# A cleaner way: build a small nn.Module that does the concatenation.

class AntActionInputCombiner(nn.Module):
    def __init__(self, ant_action_network_backend):
        super().__init__()
        self.ant_action_network_backend = ant_action_network_backend

    def forward(self, observation_agents, central_command):
        # observation_agents: [batch, n_agents, obs_dim]
        # central_command: [batch, command_dim]

        # Expand central_command to match n_agents dimension
        # [batch, command_dim] -> [batch, 1, command_dim] -> [batch, n_agents, command_dim]
        expanded_command = central_command.unsqueeze(1).expand(-1, observation_agents.size(1), -1)

        # Concatenate along the last dimension
        # combined_input: [batch, n_agents, obs_dim + command_dim]
        combined_input = torch.cat([observation_agents, expanded_command], dim=-1)
        return self.ant_action_network_backend(combined_input) # Pass to the MultiAgentMLP

# The MultiAgentMLP now receives the combined input directly
ant_action_combiner_wrapper = AntActionInputCombiner(ant_action_net)

ant_action_module = TensorDictModule(
    module=ant_action_combiner_wrapper,
    in_keys=[("agents", "observation"), ("central_command")],
    out_keys=[("agents", "loc_scale_cat")], # temp key
)

# Add NormalParamExtractor to split loc and scale
param_extractor = TensorDictModule(
    NormalParamExtractor(), # splits the last dim of input into loc and scale
    in_keys=[("agents", "loc_scale_cat")],
    out_keys=[("agents", "loc"), ("agents", "scale")]
)

# --- Assemble the Actor Policy ---
# This sequential module defines the entire actor forward pass
actor_network_sequence = TensorDictSequential(
    ant_message_module,
    central_command_module,
    ant_action_module,
    param_extractor
)

# Finally, wrap with ProbabilisticActor
policy_actor = ProbabilisticActor(
    module=actor_network_sequence, # This is our complex sequence
    spec=env.unbatched_action_spec, # Use unbatched_spec for single instance processing guide
    in_keys=[("agents", "loc"), ("agents", "scale")], # Keys that NormalParamExtractor outputs
    out_keys=[("agents", "action")], # Corresponds to env.action_key
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.unbatched_action_spec[("agents", "action")].space.low,
        "high": env.unbatched_action_spec[("agents", "action")].space.high,
        "tanh_loc": False # Important for TanhNormal if loc isn't pre-tanh'd
    },
    return_log_prob=True,
    log_prob_key=("agents", "sample_log_prob"),
)

# --- Test the policy ---
# Create a dummy observation tensordict
batch_size = 4
dummy_obs_data = torch.randn(batch_size, n_agents, obs_per_agent_dim, device=device_example)
# The input to the policy is typically what env.reset() or env.step() would return
# which usually has ("agents", "observation") and other keys.
# For a standalone test, ensure the initial in_keys of the first module are present.
input_td = TensorDict(
    {("agents", "observation"): dummy_obs_data},
    batch_size=[batch_size],
    device=device_example
)

print("Input TensorDict device:", input_td.device)
print("Policy actor device:", policy_actor.device)
# It's good practice to ensure all submodules are on the correct device.
# TensorDictModule and MultiAgentMLP handle device placement if 'device' arg is passed.

# Perform a forward pass
output_td = policy_actor(input_td.clone()) # clone because modules can modify in-place

print("\nOutput TensorDict after policy pass:")
print(output_td)
print("\nKeys in output TensorDict:", output_td.keys(True, True)) # include_nested=True, leaves_only=True
assert ("agents", "action") in output_td.keys(True)
assert ("agents", "sample_log_prob") in output_td.keys(True)
assert ("agents", "message") in output_td.keys(True)
assert ("central_command") in output_td.keys() # Not nested under "agents"
assert ("agents", "loc") in output_td.keys(True)
assert ("agents", "scale") in output_td.keys(True)

print(f"\nAction shape: {output_td[('agents', 'action')].shape}") # Expected: [batch, n_agents, action_dim]
print(f"Message shape: {output_td[('agents', 'message')].shape}") # Expected: [batch, n_agents, message_dim]
print(f"Central command shape: {output_td['central_command'].shape}") # Expected: [batch, command_dim]
print(f"Log prob shape: {output_td[('agents', 'sample_log_prob')].shape}") # Expected: [batch, n_agents]