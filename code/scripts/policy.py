from madrona_escape_room_learn import (
    ActorCritic, MaskedDiscreteActor, Critic, BackboneSeparate, RecurrentBackboneEncoder,
)

from madrona_escape_room_learn.models import (
    HiveEncoderRNN, MaskedLinearLayerDiscreteActor, LinearLayerCritic,
)
from madrona_escape_room_learn.cfg import ModelConfig, Consts

import math
import torch

def setup_obs(sim):
    self_obs_tensor = sim.self_observation_tensor().to_torch() # [N, A, ?]
    lidar_tensor = sim.lidar_tensor().to_torch() # [N, A, ?, ?]
    steps_remaining_tensor = sim.steps_remaining_tensor().to_torch() # [N, 1]
    active_agents_tensor = sim.active_agents_tensor().to_torch() # [N, A]

    assert self_obs_tensor.shape[1] == Consts.MAX_AGENTS
    assert lidar_tensor.shape[1] == Consts.MAX_AGENTS
    assert steps_remaining_tensor.shape[1] == 1
    assert active_agents_tensor.shape[1] == Consts.MAX_AGENTS

    N, A = self_obs_tensor.shape[0:2]
    assert A == Consts.MAX_AGENTS

    # give each agent a steps_remaining observations
    steps_remaining_tensor = steps_remaining_tensor.view(N, 1, 1).expand(N, A, 1)

    # Add in an agent ID tensor
    id_tensor = torch.arange(A).float()
    if A > 1:
        id_tensor = id_tensor / (A - 1)

    id_tensor = id_tensor.to(device=self_obs_tensor.device)
    id_tensor = id_tensor.view(1, A, 1).expand(N, A, 1)

    active_agents_tensor = active_agents_tensor.view(N, A, 1)

    obs_tensors = [
        self_obs_tensor,
        lidar_tensor,
        steps_remaining_tensor,
        id_tensor,
    ]

    num_obs_features = 0
    for tensor in obs_tensors:
        num_obs_features += math.prod(tensor.shape[2:])


    obs_tensors.append(active_agents_tensor)

    # collapse the agent dimension into the feature dimension; there's one policy.
    collapsed_obs_tensors = [tensor.view(N, tensor.shape[1] * tensor.shape[2], *tensor.shape[3:]) for tensor in obs_tensors]

    return collapsed_obs_tensors, num_obs_features # num_obs_features should be for one agent, without active_agent flag; only used by make_policy

def process_obs(self_obs, lidar, steps_remaining, ids, active_agents):
    assert(not torch.isnan(self_obs).any())
    assert(not torch.isinf(self_obs).any())

    assert(not torch.isnan(lidar).any())
    assert(not torch.isinf(lidar).any())

    assert(not torch.isnan(steps_remaining).any())
    assert(not torch.isinf(steps_remaining).any())

    assert(not torch.isnan(ids).any())
    assert(not torch.isinf(ids).any())

    assert(not torch.isnan(active_agents).any())
    assert(not torch.isinf(active_agents).any())

    # should all have n_worlds in first dimension
    assert(self_obs.shape[0] == lidar.shape[0] == steps_remaining.shape[0] == ids.shape[0] == active_agents.shape[0])
    # instead, assert each dim 1 is divisible by MAX_AGENTS
    assert(self_obs.shape[1] % Consts.MAX_AGENTS == 0)
    assert(lidar.shape[1] % Consts.MAX_AGENTS == 0)
    assert(steps_remaining.shape[1] % Consts.MAX_AGENTS == 0)
    assert(ids.shape[1] % Consts.MAX_AGENTS == 0)
    assert(active_agents.shape[1] % Consts.MAX_AGENTS == 0)
    
    N = self_obs.shape[0]
    A = Consts.MAX_AGENTS


    # Note that active_agents is the last tensor. It is not an observation, but a mask.
    # It's just easiest to fit into the interface this way.
    return torch.cat([
        self_obs.view(N, A, -1),
        lidar.view(N, A, -1),
        steps_remaining.view(N, A, -1).float() / Consts.MAX_STEPS,
        ids.view(N, A, -1),
        active_agents.view(N, A, -1),
    ], dim=2)

def make_policy(num_obs_features):

    actor_encoder = RecurrentBackboneEncoder(
        net = torch.nn.Identity(),
        rnn = HiveEncoderRNN(num_obs_features, ModelConfig.pre_act_dim),
    )

    critic_encoder = RecurrentBackboneEncoder(
        net = torch.nn.Identity(),
        rnn = HiveEncoderRNN(num_obs_features, 0),
    )

    backbone = BackboneSeparate(
        process_obs = process_obs,
        actor_encoder = actor_encoder,
        critic_encoder = critic_encoder,
    )

    return ActorCritic(
        backbone = backbone,
        actor = MaskedLinearLayerDiscreteActor(
            [4, 8, 5, 2],
            ModelConfig.pre_act_dim,
        ),
        critic = LinearLayerCritic(ModelConfig.lstm_dim),
    )
