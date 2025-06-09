from madrona_escape_room_learn import (
    ActorCritic, DiscreteActor, Critic, 
    BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder,
)

from madrona_escape_room_learn.models import *

from madrona_escape_room_learn.rnn import LSTM

from madrona_escape_room_learn.cfg import NonRecurrentModelConfig, RecurrentModelConfig, MLPModelConfig, EnvParams

import math
import torch

def setup_obs(sim):
    # Each should be transformed to num_worlds, num models per world, {1 or A}, (>=1 feature dim)...
    self_obs_tensor = sim.self_observation_tensor().to_torch()
    lidar_tensor = sim.lidar_tensor().to_torch()
    steps_remaining_tensor = sim.steps_remaining_tensor().to_torch()

    N, M, A = self_obs_tensor.shape[0:3] # num worlds, num models per world, num agents per model
    batch_size = N * M

    id_tensor = torch.arange(A).float()
    if A > 1:
        id_tensor = id_tensor / (A - 1)
    id_tensor = id_tensor.to(device=self_obs_tensor.device)
    id_tensor = id_tensor.view(1, 1, A, 1).expand(batch_size, -1, -1, -1)

    obs_tensors = [
        self_obs_tensor.view(batch_size, *self_obs_tensor.shape[2:]),
        lidar_tensor.view(batch_size, *lidar_tensor.shape[2:]),
        steps_remaining_tensor.view(batch_size, *steps_remaining_tensor.shape[2:]),
        id_tensor.view(batch_size, *id_tensor.shape[2:])
    ]

    num_obs_features_per_agent = 0
    for tensor in obs_tensors:
        num_obs_features_per_agent += math.prod(tensor.shape[2:]) # starts counting after agent dim

    return obs_tensors, num_obs_features_per_agent, A

def process_obs(self_obs, lidar, steps_remaining, id_tensor):
    assert(not torch.isnan(self_obs).any())
    assert(not torch.isinf(self_obs).any())

    assert(not torch.isnan(lidar).any())
    assert(not torch.isinf(lidar).any())

    assert(not torch.isnan(steps_remaining).any())
    assert(not torch.isinf(steps_remaining).any())

    # custom processing for individual inputs
    num_agents = self_obs.shape[1]
    
    steps_remaining = steps_remaining.expand(-1, num_agents, -1).float() / EnvParams.episode_len

    obs_by_agent = torch.cat([
        self_obs.view(*self_obs.shape[0:2], -1),
        lidar.view(*lidar.shape[0:2], -1),
        steps_remaining.view(*steps_remaining.shape[0:2], -1),
        id_tensor.view(*id_tensor.shape[0:2], -1),
        ], dim = -1
    ) # [N * models, agents, feature_dims]

    return obs_by_agent.view(obs_by_agent.shape[0], -1) # [N * models, agents * features]

def make_mlp_policy(num_obs_features_per_agent, num_agents_per_model):
    # ---------- MLP ----------
    actor_encoder = BackboneEncoder(
        net = MLP(
            input_dim = num_agents_per_model * num_obs_features_per_agent,
            num_channels = num_agents_per_model * MLPModelConfig.num_actor_channels,
            num_layers = MLPModelConfig.num_actor_layers
        )
    )

    critic_encoder = BackboneEncoder(
        net = MLP(
            input_dim = num_agents_per_model * num_obs_features_per_agent,
            num_channels = num_agents_per_model * MLPModelConfig.num_critic_channels,
            num_layers = MLPModelConfig.num_critic_layers
        )
    )

    backbone = BackboneSeparate(
        process_obs = process_obs,
        actor_encoder = actor_encoder,
        critic_encoder = critic_encoder
    )

    return ActorCritic(
        backbone = backbone,
        actor = LinearLayerDiscreteActor(
            num_agents_per_model * [4, 8, 5, 2],
            MLPModelConfig.num_actor_channels,
        ),
        critic = LinearLayerCritic(MLPModelConfig.num_critic_channels)
    )

def make_non_recurrent_policy(num_obs_features_per_agent):
    # [N * M, agents * observations] -> [N * models, agents * channels] (independent of num agents)
    actor_encoder = BackboneEncoder(
        net = DictatorAttentionActorEncoder(
            obs_per_agent = num_obs_features_per_agent,
            agent_msg_dim = NonRecurrentModelConfig.agent_msg_dim,
            agent_msg_mlp_num_layers = NonRecurrentModelConfig.agent_msg_mlp_num_layers,
            num_attn_heads = NonRecurrentModelConfig.num_attn_heads,
            command_mlp_num_layers = NonRecurrentModelConfig.command_mlp_num_layers,
            agent_action_mlp_num_layers = NonRecurrentModelConfig.agent_action_mlp_num_layers,
            command_dim = NonRecurrentModelConfig.command_dim,
            action_logits_dim = ModelConfig.action_logits_dim,
        )
    )

    # [N * M, agents * observations] -> attention -> [N * models, channels] (independent of num_agents)
    critic_encoder = BackboneEncoder(
        net = AttentionEncoder(
            input_dim_per_agent = num_obs_features_per_agent,
            msg_dim = ModelConfig.agent_msg_dim,
            mlp1_num_layers = ModelConfig.agent_msg_mlp_num_layers,
            num_attn_heads = ModelConfig.num_attn_heads,
            mlp2_num_layers = ModelConfig.command_mlp_num_layers,
            output_dim = ModelConfig.num_critic_channels
        )
    )

    backbone = BackboneSeparate(
        process_obs = process_obs,
        actor_encoder = actor_encoder,
        critic_encoder = critic_encoder
    )

    # actor: [N * models, agents * channels] -> [N * models, agents * actions]; each agent processed independently
    # critic: [N * models, channels] -> [N * models, 1]
    return ActorCritic(
        backbone = backbone,
        actor = MultiAgentLinearDiscreteActor(
            [4, 8, 5, 2],
            ModelConfig.action_logits_dim,
        ),
        critic = LinearLayerCritic(ModelConfig.num_critic_channels)
    )

def make_recurrent_policy(num_obs_features_per_agent):
    actor_encoder = RecurrentBackboneEncoder(
        net = nn.Identity(),
        rnn = RecurrentAttentionActorEncoder(
            obs_per_agent = num_obs_features_per_agent,
            agent_msg_dim = RecurrentModelConfig.agent_msg_dim,
            agent_mlp_num_layers = RecurrentModelConfig.agent_msg_mlp_num_layers,
            lstm_hidden_size = RecurrentModelConfig.lstm_hidden_size,
            num_attn_heads = RecurrentModelConfig.num_attn_heads,
            pooled_msg_dim = RecurrentModelConfig.pooled_msg_dim,
            pooled_msg_mlp_num_layers = RecurrentModelConfig.pooled_msg_mlp_num_layers,
            agent_action_mlp_num_layers = RecurrentModelConfig.agent_action_mlp_num_layers,
            command_dim = RecurrentModelConfig.command_dim,
            action_logits_dim = RecurrentModelConfig.action_logits_dim,
            command_mlp_num_layers = RecurrentModelConfig.command_mlp_num_layers,
        )
    )

    critic_encoder = RecurrentBackboneEncoder(
        net = nn.Identity(),
        rnn = RecurrentAttentionCriticEncoder(
            obs_per_agent = num_obs_features_per_agent,
            agent_msg_dim = RecurrentModelConfig.agent_msg_dim,
            agent_msg_mlp_num_layers = RecurrentModelConfig.agent_msg_mlp_num_layers,
            num_attn_heads = RecurrentModelConfig.num_attn_heads,
            pooled_msg_dim = RecurrentModelConfig.pooled_msg_dim,
            pooled_msg_mlp_num_layers = RecurrentModelConfig.pooled_msg_mlp_num_layers,
            lstm_hidden_size = RecurrentModelConfig.lstm_hidden_size,
            out_mlp_num_layers = RecurrentModelConfig.out_mlp_num_layers,
            num_critic_channels = RecurrentModelConfig.num_critic_channels,
        )
    )

    backbone = BackboneSeparate(
        process_obs = process_obs,
        actor_encoder = actor_encoder,
        critic_encoder = critic_encoder
    )


    # actor: [N * models, agents * channels] -> [N * models, agents * actions]; each agent processed independently
    # critic: [N * models, channels] -> [N * models, 1]
    return ActorCritic(
        backbone = backbone,
        actor = MultiAgentLinearDiscreteActor(
            [4, 8, 5, 2],
            RecurrentModelConfig.action_logits_dim,
        ),
        critic = LinearLayerCritic(NonRecurrentModelConfig.num_critic_channels)
    )

def make_policy(num_obs_features_per_agent, num_agents_per_model, num_channels, separate_value):
    return make_mlp_policy(num_obs_features_per_agent, num_agents_per_model)
    # return make_non_recurrent_policy(num_obs_features_per_agent)
    # return make_recurrent_policy(num_obs_features_per_agent)