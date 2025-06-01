from hive_learn import (
    ActorCritic,  BackboneSeparate,
)   
from hive_learn.cfg import ModelConfig, Consts

from hive_learn.models import (
    LinearLayerDiscreteActor, LinearLayerCritic,
)

from hive_learn.rnn import LSTM

import math
import torch
from hive_learn.cfg import Consts
from hive_learn.hive_encoder import HiveEncoder

# Modified to ignore partner, door, and room entity obs
def setup_obs(sim):
    self_obs_tensor = sim.self_observation_tensor().to_torch()
    # partner_obs_tensor = sim.partner_observations_tensor().to_torch()
    # room_ent_obs_tensor = sim.room_entity_observations_tensor().to_torch()
    # door_obs_tensor = sim.door_observation_tensor().to_torch()
    lidar_tensor = sim.lidar_tensor().to_torch()
    steps_remaining_tensor = sim.steps_remaining_tensor().to_torch()
    active_agents_tensor = sim.active_agents_tensor().to_torch()

    # Print shapes for debugging
    print(f"self_obs_tensor shape: {self_obs_tensor.shape}")
    print(f"lidar_tensor shape: {lidar_tensor.shape}")
    print(f"steps_remaining_tensor shape: {steps_remaining_tensor.shape}")
    print(f"active_agents_tensor shape: {active_agents_tensor.shape}")
    
    N, A = self_obs_tensor.shape[0:2]  # N = num_worlds, A = num_agents
    batch_size = N # batch size is num_worlds because we aren't processing obs per agent
    
    # Add in an agent ID tensor
    id_tensor = torch.arange(A).float()
    if A > 1:
        id_tensor = id_tensor / (A - 1)

    # We want id tensor to be [N, A]
    id_tensor = id_tensor.to(device=self_obs_tensor.device)
    id_tensor = id_tensor.view(1, A).expand(N, A)
    
    # Handle steps_remaining_tensor which is [num_worlds, 1]
    # Need to expand it to match the batch size by repeating for each agent
    steps_remaining_expanded = steps_remaining_tensor.view(N, 1).expand(N, A)
    
    # Handle active_agents_tensor which is [num_worlds, num_agents]
    active_agents_expanded = active_agents_tensor.view(N, A)
    
    obs_tensors = [
        self_obs_tensor.view(batch_size, *self_obs_tensor.shape[1:]),
        lidar_tensor.view(batch_size, *lidar_tensor.shape[1:]),
        steps_remaining_expanded,
        id_tensor,
    ]

    # compute the number of features
    total_feature_dim = 0
    for tensor in obs_tensors:
        total_feature_dim += math.prod(tensor.shape[1:])

    # stick on the active agents to pass it thru everywhere
    obs_tensors.append(active_agents_expanded)

    # return the obs tensors and the number of features
    return obs_tensors, total_feature_dim
    
# Modified to ignore partner, door, and room entity obs
def process_obs(self_obs, lidar, steps_remaining, ids, active_agents):
    assert(not torch.isnan(self_obs).any())
    assert(not torch.isinf(self_obs).any())

    assert(not torch.isnan(lidar).any())
    assert(not torch.isinf(lidar).any())

    assert(not torch.isnan(steps_remaining).any())
    assert(not torch.isinf(steps_remaining).any())

    assert(not torch.isnan(active_agents).any())
    assert(not torch.isinf(active_agents).any())
    
    obs_tensor = torch.cat([
        self_obs.view(self_obs.shape[0], -1),
        lidar.view(lidar.shape[0], -1),
        steps_remaining.float() / Consts.MAX_STEPS,
        ids,
    ], dim=1)

    return (obs_tensor, active_agents)

def make_policy(num_obs_features):
    actor_encoder = HiveEncoder(
        num_obs_features // Consts.MAX_AGENTS,
        ModelConfig.pre_act_dim,
    )

    critic_encoder = HiveEncoder(
        num_obs_features // Consts.MAX_AGENTS,
        0,
    )

    backbone = BackboneSeparate(
        process_obs,
        actor_encoder=actor_encoder,
        critic_encoder=critic_encoder,
    )

    return ActorCritic(
        backbone = backbone,
        actor = LinearLayerDiscreteActor(
            [4, 8, 5, 2],
            ModelConfig.pre_act_dim,
        ),
        critic = LinearLayerCritic(ModelConfig.lstm_dim),
    )