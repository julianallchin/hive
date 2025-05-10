# Interfacing Madrona Hive Simulation with Hivemind Training

This guide explains how to properly interface your Madrona hive simulation with the hierarchical reinforcement learning training architecture.

---

## 1. Observation and Action Structure

### Observation Format

The simulation must provide observations in the following format:

```python
sim.obs = [ant_observations, ...other_optional_observations]
```

Where:

* `ant_observations`: Tensor of shape `[batch_size, num_ants, ant_obs_dim]`

  * `batch_size`: Number of parallel simulations
  * `num_ants`: Number of ants per simulation (e.g., 100)
  * `ant_obs_dim`: Dimension of each ant's observation vector

Each ant’s observation should include:

* Self state (position, orientation)
* Task observations (relative polar coordinates to macguffin and goal)
* Raycast/LiDAR observations
* Any other relevant information

### Action Format

Training code outputs actions as:

```python
sim.actions: Tensor of shape [batch_size, num_ants, action_components]
```

* `action_components`: Total number of discrete action components (e.g., movement, grab)

### Other Required Tensors

* `sim.dones`: `[batch_size, 1]` - indicates episode completion
* `sim.rewards`: `[batch_size, 1]` - global hivemind reward

---

## 2. Configuring Madrona Components

### Simulation Setup

Ensure these singletons are registered:

```cpp
world.registerSingleton<WorldReset>();
world.registerSingleton<LevelState>();
world.registerSingleton<HiveReward>();
world.registerSingleton<HiveDone>();
world.registerSingleton<StepsRemaining>();
```

### Ant Archetype

```cpp
world.registerArchetype<Ant>()
    .withComponent<Position>()
    .withComponent<Rotation>()
    .withComponent<Lidar>()
    .withComponent<TaskObs>()
    .withComponent<AntAction>();
```

### Other Required Archetypes

```cpp
world.registerArchetype<Macguffin>().withComponent<Position>().withComponent<Physics>();
world.registerArchetype<Goal>().withComponent<Position>();
world.registerArchetype<Wall>().withComponent<Position>().withComponent<Rotation>().withComponent<Scale>();
world.registerArchetype<MovableObject>().withComponent<Position>().withComponent<Physics>();
```

---

## 3. Observations Export

Implement an `ExportObservations` system:

```cpp
class ExportObservations : public System {
public:
    void update(World &world) {
        auto &obs_tensor = world.getExported<ObservationTensor>();
        int32_t num_simulations = world.numWorldsInUse();
        int32_t ants_per_sim = world.getCurrentConfig().num_ants;
        int32_t obs_dim = /* total observation dimension */;

        obs_tensor.resize({num_simulations, ants_per_sim, obs_dim});

        world.parallelForEntities<Ant, Position, Rotation, Lidar, TaskObs>(
            [&](EntityRef ant_entity, Position &pos, Rotation &rot, Lidar &lidar, TaskObs &task_obs) {
                int32_t world_idx = ant_entity.worldIdx();
                int32_t ant_idx = /* index within simulation */;
                float *ant_obs_ptr = &obs_tensor[{world_idx, ant_idx, 0}];

                ant_obs_ptr[0] = pos.x;
                ant_obs_ptr[1] = pos.y;
                // ...
                for (int i = 0; i < lidar.num_rays; i++) {
                    ant_obs_ptr[lidar_offset + i] = lidar.distances[i];
                }

                ant_obs_ptr[task_offset + 0] = task_obs.macguffin_distance;
                ant_obs_ptr[task_offset + 1] = task_obs.macguffin_angle;
                ant_obs_ptr[task_offset + 2] = task_obs.goal_distance;
                ant_obs_ptr[task_offset + 3] = task_obs.goal_angle;
            }
        );
    }
};
```

---

## 4. Actions Import

```cpp
class ImportActions : public System {
public:
    void update(World &world) {
        auto &action_tensor = world.getImported<ActionTensor>();

        world.parallelForEntities<Ant, AntAction>(
            [&](EntityRef ant_entity, AntAction &action) {
                int32_t world_idx = ant_entity.worldIdx();
                int32_t ant_idx = /* index within simulation */;

                int32_t move_x_idx = action_tensor[{world_idx, ant_idx, 0}];
                int32_t move_y_idx = action_tensor[{world_idx, ant_idx, 1}];
                int32_t grab_idx = action_tensor[{world_idx, ant_idx, 2}];

                action.move_x = convertActionBucket(move_x_idx);
                action.move_y = convertActionBucket(move_y_idx);
                action.grab = grab_idx > 0;
            }
        );
    }

private:
    float convertActionBucket(int32_t idx) {
        return (idx / 2.0f) - 1.0f;
    }
};
```

---

## 5. Reward Calculation

```cpp
class CalculateReward : public System {
public:
    void update(World &world) {
        for (int32_t w_idx = 0; w_idx < world.numWorldsInUse(); w_idx++) {
            auto &reward = world.getSingleton<HiveReward>(w_idx);
            auto &steps = world.getSingleton<StepsRemaining>(w_idx);

            auto macguffin_pos = /* get macguffin position */;
            auto goal_pos = /* get goal position */;
            float current_dist = distance(macguffin_pos, goal_pos);

            float step_reward = (reward.prev_distance > 0)
                ? 0.1f * (reward.prev_distance - current_dist)
                : 0;

            float goal_reward = 0;
            if (current_dist <= 0.5f) {
                goal_reward = 1.0f;
                world.getSingleton<HiveDone>(w_idx).done = true;
            }

            float exist_penalty = -1.0f / world.getCurrentConfig().max_steps;

            reward.value = step_reward + goal_reward + exist_penalty;
            reward.prev_distance = current_dist;

            steps.remaining--;
            if (steps.remaining <= 0) {
                world.getSingleton<HiveDone>(w_idx).done = true;
            }
        }
    }
};
```

---

## 6. Python Training Configuration

```python
from madrona_escape_room_learn.cfg import TrainConfig, PPOConfig, HivemindConfig

action_buckets = [5, 5, 2]

hivemind_cfg = HivemindConfig(
    num_ants=100,
    ant_obs_dim=64,
    ant_mlp_hidden=128,
    command_dim=32,
    message_dim=32,
    attn_heads=4,
    attn_output_dim=64,
    lstm_hidden_dim=128,
    lstm_layers=1,
)

ppo_cfg = PPOConfig(
    num_mini_batches=4,
    clip_coef=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    num_epochs=4,
)

train_cfg = TrainConfig(
    num_updates=5000,
    steps_per_update=32,
    num_bptt_chunks=2,
    lr=3e-4,
    gamma=0.99,
    ppo=ppo_cfg,
    hivemind=hivemind_cfg,
    gae_lambda=0.95,
)
```

---

## 7. Model Initialization

```python
import torch
from madrona_escape_room_learn.actor_critic import ActorCritic, HiveBackbone, DiscreteActor
from madrona_escape_room_learn.models import IdentityNet, LinearLayerCritic

total_phys_action_logits = sum(action_buckets)

hive_backbone = HiveBackbone(
    num_ants=hivemind_cfg.num_ants,
    ant_local_obs_dim=hivemind_cfg.ant_obs_dim,
    command_dim=hivemind_cfg.command_dim,
    ant_mlp_hidden=hivemind_cfg.ant_mlp_hidden,
    phys_action_total_logits=total_phys_action_logits,
    message_dim=hivemind_cfg.message_dim,
    attn_heads=hivemind_cfg.attn_heads,
    attn_output_dim=hivemind_cfg.attn_output_dim,
    lstm_hidden_dim=hivemind_cfg.lstm_hidden_dim,
    lstm_layers=hivemind_cfg.lstm_layers
)

actor_module = DiscreteActor(
    actions_num_buckets=action_buckets,
    impl=IdentityNet()
)

critic_module = LinearLayerCritic(hivemind_cfg.lstm_hidden_dim)

policy = ActorCritic(
    backbone=hive_backbone,
    actor=actor_module,
    critic=critic_module
)
```

---

## 8. Simulation Interface Setup

```python
from madrona_escape_room_learn.cfg import SimInterface
from madrona_escape_room_learn.train import Trainer

sim_interface = SimInterface(
    step=sim.step,
    obs=[sim.ant_obs],
    actions=sim.actions,
    dones=sim.dones,
    rewards=sim.rewards
)

trainer = Trainer(
    policy=policy,
    optim=torch.optim.Adam(policy.parameters(), lr=train_cfg.lr),
    sim=sim_interface,
    cfg=train_cfg
)

trainer.train()
```

---

## 9. Key Implementation Notes

* **Ant Observations**: Ensure completeness and proper normalization.
* **LSTM State**: Managed automatically by the training code.
* **Communication**: Handled within the model, not the simulation.
* **Reward Function**:

  * Step reward: `r_t^step = 0.1 * (d_{t-1} - d_t)`
  * Goal reward: `r_t^goal = +1 if d_t <= ε`
  * Existential penalty: `r_t^exist = -1 / T_max` per timestep
* **Environment Randomization**: Vary ant count, wall positions, and objects on reset.
