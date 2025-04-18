# THRONG

## A Note on Shouting Ants

During a late-night brainstorming session with Ty and Diego, Ty remarked:

> “Ants will have to learn to be louder.”

It got me thinking — in systems where many agents share space and limited bandwidth, communication isn't just about speaking. It's about _being heard_, _when_, _by whom_, and _how much_.

This naturally led us toward attention — a way for agents to learn not just to broadcast messages, but to determine whose voice to listen to and when. It’s a framework not just for noise, but for meaningful signal in a dynamic, decentralized team.

---

## Project Purpose & Overview

This project is part of a Stanford reinforcement learning course. We are developing a multi-agent simulation environment where ants work together to move an object to a goal position in a 2D top-down world. The project explores:

- **Collaboration** between identical agents through shared models and memory.
- **Communication** via learned message vectors, broadcast and consumed at each step.
- **Adaptability** through variable environment configurations and agent counts.
- **Attention and memory**, both as emergent properties and architectural features.

We aim to understand and prototype shared intelligence, especially under the constraints of decentralized observation and real-time decision-making.

---

## Environment

### Basic Setup

- A variable number of ants are placed in a bounded 2D top-down space.
- A movable object is placed somewhere in the environment.
- A goal location is defined.
- Ants must coordinate to push or move the object to the goal.

### Characteristics

- **Agents (Ants)**: Homogeneous, fully shared policy and memory.
- **Object**: Varying properties (mass, shape, friction) across episodes.
- **Goal**: Fixed or moving; location is provided to agents as input.
- **Physics**: Object motion obeys simplified continuous dynamics.
- **Simulation Backend**: Madrona is used for simulation and visualization.

---

## Ant Observation Space

Each ant receives input at each timestep composed of the following:

- **Local State (4 dims)**

  - `x`, `y`: absolute position
  - `vel_x`, `vel_y`: local velocity

- **Task-Relevant Vectors (4 dims)**

  - Relative position vector from the ant to the object center
  - Relative position vector from the ant to the goal position

- **Raycast Perception (configurable, e.g. 6 rays × 3 features = 18 dims)**

  - Rays cast radially outward
  - Each ray returns:
    - Normalized distance [0–1]
    - One-hot entity class encoding: Wall, Object, Ant, or None

- **Cloud Communication Input (16 dims)**
  - Learned, aggregated communication vector from all other ants at previous step

### Total Input Dimension

For 6 raycasts:  
`2 (pos) + 2 (vel) + 2 (rel to object) + 2 (rel to goal) + 18 (raycasts) + 16 (cloud) = 42 dims`

---

## Ant Output Space

Each ant outputs two things at each timestep:

- **Action (2 dims)**

  - Target velocity in `x` and `y` directions

- **Message (16 dims)**
  - Communication vector, which becomes part of the cloud input for the next timestep

---

## Model Architecture

### Overview

All ants share a **single model** with a **shared LSTM memory**. This model:

- Processes input at each timestep (including cloud input)
- Updates global memory
- Produces per-ant action and communication output

### Architecture Flow (per timestep):

1. **Observation Gathering**

   - Each ant collects its local input

2. **Cloud Attention**

   - All ants' messages from the previous timestep are collected
   - Each ant computes attention over these messages:
     - Its message becomes a query
     - All messages are projected into keys and values
     - The weighted sum of values (via softmaxed dot-product attention) becomes `cloud_input_i`

3. **Input Construction**

   - Each ant’s raw observation is concatenated with `cloud_input_i`

4. **Shared LSTM**

   - All per-ant inputs are stacked and passed through a single shared LSTM
   - A single `(h_t, c_t)` pair is used across all ants
   - LSTM outputs a per-ant feature vector `o_i`

5. **Output Heads**

   - `o_i` is passed through two heads:
     - **Action Head** → 2D velocity vector
     - **Message Head** → 16D communication vector

6. **Message Broadcast**
   - Messages are stored and used as input for cloud attention at the next timestep

---

## Attention Mechanism Details

### Why Attention?

We use attention to allow ants to selectively focus on relevant messages from their peers. This replaces naive mean-pooling and supports dynamic agent counts.

### How It Works (per-ant):

```python
query_i = W_q(m_i)          # from ant i’s own message
keys    = W_k(messages)     # from all ants
values  = W_v(messages)

scores  = query_i @ keys.T / sqrt(d_k)
weights = softmax(scores)
cloud_input_i = weights @ values
```

- `messages`: the 16D communication vectors from each ant
- `W_q`, `W_k`, `W_v`: learned projections
- Output: `cloud_input_i`, a personalized read from the collective

This gives ants the ability to prioritize voices. Some ants will learn to "shout" when needed (output high-signal messages). Others will learn to listen selectively.

---

## Memory System

We use a **shared LSTM across all ants**, meaning:

- One `(h_t, c_t)` pair for the entire swarm
- All ants update this shared memory at each step
- The LSTM sees all per-ant inputs and cloud context
- Memory carries forward notions of intent, plans, or strategies across time

This setup supports a **true hivemind**: a collective brain with a unified memory across distributed bodies.

---

## Reward Function

Primary reward is based on **moving the object toward the goal**:

- `reward = -||object_position - goal_position||`
- Optionally use reward shaping:
  - Positive reward for velocity toward goal
  - Penalty for collisions
  - Bonus for aligned pushing

Reward is shared across all ants — no credit assignment mechanism (like COMA or VDN) is used initially.

---

## Training Strategy

- **Framework**: PyTorch (with potential for integration with RLlib later)
- **Algorithm**: PPO with parameter sharing and centralized rollout collection
- **Simulation**: Madrona used to parallelize thousands of environments

---

## Future Directions (Placeholder)

- Curriculum learning for object shape, mass, and goal location
- Dynamic spawning/despawning of ants (with attention-based robustness)
- Message compression and sparsity penalties
- Emergent role differentiation (e.g., scouts vs pushers)
- Visualizations of attention weights and memory activations

---
