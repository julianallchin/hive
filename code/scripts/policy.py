from hive_learn import (
    ActorCritic, DiscreteActor, Critic, 
    BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder, AntCommBlock, HiveBackbone
)
from hive_learn.cfg import ModelConfig, Consts

from hive_learn.models import (
    MLP, LinearLayerDiscreteActor, LinearLayerCritic,
)

from hive_learn.rnn import LSTM

import math
import torch

from hive_learn.actor_encoder import HiveEncoder

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
    batch_size = N * A
    
    # Add in an agent ID tensor
    id_tensor = torch.arange(A).float()
    if A > 1:
        id_tensor = id_tensor / (A - 1)

    id_tensor = id_tensor.to(device=self_obs_tensor.device)
    id_tensor = id_tensor.view(1, A).expand(N, A).reshape(batch_size, 1)
    
    # Handle steps_remaining_tensor which is [num_worlds, 1]
    # Need to expand it to match the batch size by repeating for each agent
    steps_remaining_expanded = steps_remaining_tensor.repeat_interleave(A, dim=0)
    # So the final shape of steps_remaining_expanded is [batch_size, 1]
    
    # Handle active_agents_tensor which is [num_worlds, num_agents, 1]
    active_agents_expanded = active_agents_tensor.reshape(batch_size, 1)
    
    obs_tensors = [
        self_obs_tensor.view(batch_size, *self_obs_tensor.shape[2:]),
        lidar_tensor.view(batch_size, *lidar_tensor.shape[2:]),
        steps_remaining_expanded,
        id_tensor,
        active_agents_expanded,
    ]

    num_obs_features = 0
    for tensor in obs_tensors:
        num_obs_features += math.prod(tensor.shape[1:])

    return obs_tensors, num_obs_features - 1

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
    
    return torch.cat([
        self_obs.view(self_obs.shape[0], -1),
        lidar.view(lidar.shape[0], -1),
        steps_remaining.float() / Consts.MAX_STEPS, # TODO: should scale by sim length
        ids,
    ], dim=1), active_agents

def make_policy(num_obs_features):
    cfg = ModelConfig(
        pre_act_dim = 8,
        msg_dim = 8,
        ant_trunk_hid_dim = 256,
        heads = 4,
        lstm_dim = 256,
        cmd_dim = 16
    )

    # backbone = HiveBackbone(
    #     process_obs = process_obs,
    #     base_obs_dim = num_obs_features,
    #     cfg = cfg,
    # )

    actor_encoder = HiveEncoder(
        num_obs_features,
        cfg.pre_act_dim,
        cfg.msg_dim,
        cfg.ant_trunk_hid_dim,
        cfg.heads,
        cfg.lstm_dim,
        cfg.cmd_dim,
    )

    critic_encoder = HiveEncoder(
        num_obs_features,
        0,
        cfg.msg_dim,
        cfg.ant_trunk_hid_dim,
        cfg.heads,
        cfg.lstm_dim,
        cfg.cmd_dim,
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
            cfg.pre_act_dim,
        ),
        critic = LinearLayerCritic(cfg.lstm_dim),
    )