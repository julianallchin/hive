from madrona_escape_room_learn import (
    ActorCritic, DiscreteActor, Critic, 
    BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder,
)

from madrona_escape_room_learn.models import *

from madrona_escape_room_learn.rnn import LSTM

import math
import torch

def setup_obs(sim):
    # Each should be transformed to num_worlds, num models per world, num agents per model, ...(>=1 feature dim)
    self_obs_tensor = sim.self_observation_tensor().to_torch()
    lidar_tensor = sim.lidar_tensor().to_torch()
    steps_remaining_tensor = sim.steps_remaining_tensor().to_torch()

    N, M, agents_per_model = self_obs_tensor.shape[0:3]
    batch_size = N * M

    # TODO: verify the increments are passed to all agents on later steps
    # steps_remaining_tensor = steps_remaining_tensor.expand(-1, -1, agents_per_model, -1)

    # Add in an agent ID tensor
    # id_tensor = torch.arange(A).float()
    # if A > 1:
        # id_tensor = id_tensor / (A - 1)
    # id_tensor = id_tensor.to(device=self_obs_tensor.device)
    # INCORRECT LOGIC HERE RN id_tensor = id_tensor.view(1, 2).expand(N, 2).reshape(batch_size, 1)

    obs_tensors = [
        self_obs_tensor.view(batch_size, *self_obs_tensor.shape[2:]),
        lidar_tensor.view(batch_size, *lidar_tensor.shape[2:]),
        steps_remaining_tensor.view(batch_size, *steps_remaining_tensor.shape[2:]),
        # id_tensor,
    ]

    num_obs_features_per_agent = 0
    for tensor in obs_tensors:
        num_obs_features_per_agent += math.prod(tensor.shape[2:])

    return obs_tensors, num_obs_features_per_agent, agents_per_model

def process_obs(self_obs, lidar, steps_remaining):
    assert(not torch.isnan(self_obs).any())
    assert(not torch.isinf(self_obs).any())

    assert(not torch.isnan(lidar).any())
    assert(not torch.isinf(lidar).any())

    assert(not torch.isnan(steps_remaining).any())
    assert(not torch.isinf(steps_remaining).any())

    # custom processing for individual inputs
    num_agents = self_obs.shape[1]
    steps_remaining = steps_remaining.expand(-1, num_agents, -1).float() / 500

    obs_by_agent = torch.cat([
        self_obs.view(*self_obs.shape[0:2], -1),
        lidar.view(*lidar.shape[0:2], -1),
        steps_remaining.view(*steps_remaining.shape[0:2], -1)
        ], dim = -1
    ) # [N * models, agents, -1]

    return obs_by_agent.view(obs_by_agent.shape[0], -1) # [N * models, agents * features]

def make_policy(num_obs_features_per_agent, num_agents_per_model, num_channels, separate_value):
    # #encoder = RecurrentBackboneEncoder(
    # #    net = MLP(
    # #        input_dim = num_obs_features,
    # #        num_channels = num_channels,
    # #        num_layers = 2,
    # #    ),
    # #    rnn = LSTM(
    # #        in_channels = num_channels,
    # #        hidden_channels = num_channels,
    # #        num_layers = 1,
    # #    ),
    # #)

    # encoder = BackboneEncoder(
    #     net = MLP(
    #         input_dim = num_obs_features_per_agent * num_agents_per_model,
    #         num_channels = num_channels * num_agents_per_model,
    #         num_layers = 3
    #     )
    # )

    # # if separate_value:
    # #     backbone = BackboneSeparate(
    # #         process_obs = process_obs,
    # #         actor_encoder = encoder,
    # #         critic_encoder = RecurrentBackboneEncoder(
    # #             net = MLP(
    # #                 input_dim = num_obs_features_per_agent * num_agents_per_model,
    # #                 num_channels = num_channels,
    # #                 num_layers = 2,
    # #             ),
    # #             rnn = LSTM(
    # #                 in_channels = num_channels,
    # #                 hidden_channels = num_channels,
    # #                 num_layers = 1,
    # #             ),
    # #         )
    # #     )
    # # else:
    # backbone = BackboneShared(
    #     process_obs = process_obs,
    #     encoder = encoder,
    # )

    # return ActorCritic(
    #     backbone = backbone,
    #     actor = LinearLayerDiscreteActor(
    #         num_agents_per_model * [4, 8, 5, 2],
    #         num_channels * num_agents_per_model,
    #     ),
    #     critic = LinearLayerCritic(num_channels * num_agents_per_model)
    # )

    # actor_encoder = BackboneEncoder(
    #     net = MultiAgentSharedMLP(
    #         input_dim_per_agent = num_obs_features_per_agent,
    #         num_channels_per_agent = num_channels,
    #         num_layers = 3
    #     )
    # )


    # these can be different; should make them configurable
    msg_dim = num_channels
    command_dim = num_channels

    actor_encoder = DictatorAttentionActorEncoder(
        obs_per_agent = num_obs_features_per_agent,
        agent_msg_dim = msg_dim,
        num_layers = 2,
        command_dim = command_dim,
        action_logits_dim = num_channels
    )

    # critic_encoder = BackboneEncoder(
    #     net = MLP(
    #         input_dim = num_agents_per_model * num_obs_features_per_agent,
    #         num_channels = num_channels,
    #         num_layers = 3
    #     )
    # )

    critic_encoder = BackboneEncoder(
        net = AttentionEncoder(
            input_dim_per_agent = num_obs_features_per_agent,
            num_channels_per_agent = num_channels,
            num_layers = 2,
            output_dim = num_channels
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
            num_channels * num_agents_per_model,
        ),
        critic = LinearLayerCritic(num_channels)
    )