# THRONG

## Project Purpose & Overview

This project is designed for a Stanford reinforcement learning course. It explores multi-agent collaboration by training multiple ants to move an object to a goal position in a top-down 2D environment. The environment and object characteristics are intentionally variable and part of the learning challenge.

- **Multiple Ants:** The number of ants is variable and can be configured per experiment. Ants must cooperate to move an object to the goal.
- **Object:** The object can vary in shape and properties; learning to manipulate different objects is a key challenge for the agents.
- **Goal:** The primary task is to move the object as close as possible to a designated goal position.
- **Challenges:** The environment supports different challenge configurations to test generalization and robustness.
- **Environment Size:** The size is variable and not fixed at this stage.
- **Visualization:** Madrona will be used for simulation and visualization.
- **Status:** The project is currently in the theoretical/design phase; no setup or installation steps are provided yet.

## Ant Environment Overview

Simple first. Top-down 2D. Goal-oriented.

### Ant Agent Definition:

#### Inputs (Per Ant at time `t`):

Total dimension: `2 (local pos) + 2 (local vel) + 2 (rel object pos) + 2 (rel goal pos) + (6 * 3) (raycasts) + 16 (received cloud message) = 42 dimensions` (Assuming 6 rays).

- **Local State (4 dims):**
  - X, Y (continuous position)
  - Vel X, Vel Y (velocity)
- **Task Information (4 dims):**
  - Object X, Y (relative vector from ant to object center)
  - Goal X, Y (relative vector from ant to goal position)
- **Perception - Raycasts (Variable dims, e.g., 6 rays \* 3 features = 18 dims):**
  - Arranged in a circle (number of rays is configurable, e.g., 6).
  - Each ray returns:
    - Normalized Distance (1.0 = touching, 0.0 = max distance or no hit).
    - Object Class Encoding (e.g., one-hot: Wall, Ant, Object, Null). _Initial spec mentioned single 'Class', assuming one-hot or similar encoding._
- **Communication Input (16 dims):**
  - Received Cloud Message (`cloud_input` from step `t-1`'s aggregation). See "Cloud Communication Mechanism".

#### Outputs (Per Ant at time `t`):

Total dimension: `2 (action) + 16 (message) = 18 dimensions`.

- **Action (2 dims):**
  - Target Vel X
  - Target Vel Y (These likely translate to forces/accelerations applied by the ant)
- **Communication Output (16 dims):**
  - Cloud Message (`message_i` for ant `i` at step `t`)

## Agent Model & Communication

### Model Architecture (Per Ant)

All ants share the exact same model architecture and weights (parameter sharing). The model processes the inputs at each time step to produce actions and an outgoing message. An internal recurrent state (LSTM) allows the agent to integrate information over time.

1.  **Input Concatenation:** The various input components (Local State, Task Info, Raycasts, Received Cloud Message) are concatenated into a single input vector.
    - Size: `42` (for 6 rays) + `LSTM hidden state size` (from previous step `t-1`).
2.  **Recurrent Core (LSTM):**
    - The concatenated input vector is fed into an LSTM layer along with the hidden state (`h_{t-1}`, `c_{t-1}`) from the previous timestep.
    - The LSTM processes the current observation in the context of its memory.
    - It outputs a new feature vector (e.g., the LSTM's output `o_t`) and updates its hidden state (`h_t`, `c_t`) for the next timestep (`t+1`).
    - _Note:_ The size of the LSTM hidden state is a hyperparameter (e.g., 128).
3.  **Output Heads (MLPs):** The output feature vector from the LSTM (or potentially further processed by MLP layers) is fed into two separate linear layers (or small MLPs) to produce the final outputs:
    - **Action Head:** Produces the 2D target velocity vector.
    - **Message Head:** Produces the 16D cloud message vector.

**Conceptual Flow (Single Ant, Time Step `t`):**

```
Inputs(t):
  - Local State
  - Task Info
  - Raycasts
  - Received Cloud Message (from t-1 aggregation)
  - LSTM Hidden State (h_{t-1}, c_{t-1})
     |
     V
[ Input Concatenation ]
     |
     V
[ LSTM Layer ] -> New LSTM Hidden State (h_t, c_t) -> (Used in step t+1)
     |
     V
[ LSTM Output Features (o_t) ] -> Optional [ MLP Layers ]
     |
     +--- [ Action Head (Linear/MLP) ] ---> Action Output (Vel X, Vel Y) at t
     |
     +--- [ Message Head (Linear/MLP) ] --> Cloud Message Output (16-dim vector) at t
```

_Note:_ The exact sizes of intermediate MLP layers are hyperparameters to be tuned during experimentation (e.g., the `64 x 128 x 128` mentioned earlier might refer to potential MLP layer sizes after the LSTM, leading to the final 18 outputs).

### Cloud Communication Mechanism ("Cloud Brain")

This mechanism enables information sharing and coordination across all ants.

1.  **Message Generation:** At each simulation step `t`, every ant `i` uses its model to generate an outgoing 16-dimensional `message_i` based on its current observations and internal state.
2.  **Aggregation:** A central mechanism (conceptually the "cloud") collects all `message_i` from all `N` ants currently active in the environment. It calculates the mean of these vectors:
    `cloud_input = (message_1 + message_2 + ... + message_N) / N`
3.  **Broadcast:** This aggregated `cloud_input` vector is then broadcast back to _all_ ants.
4.  **Input for Next Step:** Each ant receives this `cloud_input` as part of its input observations for the _next_ simulation step (`t+1`).

This creates a shared contextual signal derived from the collective state/intentions (as encoded in the messages) of the entire group in the previous step.

### Weight Sharing

- **Homogeneous Agents:** All ants are considered identical in their capabilities and learning process.
- **Parameter Sharing:** A single neural network model (with a single set of weights) is trained. Every ant in the environment uses an identical copy of this model.
- **Benefits:**
  - Significantly reduces the number of parameters to learn.
  - Allows agents to learn general cooperative behaviors from the collective experience of all ants.
  - Promotes scalability as the number of ants changes.

## Reward Structure

```mermaid
flowchart TD
    A[Object Position] --> B[Calculate Distance to Goal]
    C[Goal Position] --> B
    B --> D[Reward = -||ObjectPosition - GoalPosition||]
    D --> E[Shared Reward to All Ants]
    E --> F[Encourages Cooperative Behavior]
```

- The primary reward signal is based on improving the object's proximity to the goal position.
- No advanced learning systems (e.g., curriculum learning, specific multi-agent credit assignment techniques beyond parameter sharing and shared reward) are planned for the initial phase. The shared reward implicitly encourages cooperation.

## Perception & Raycasting

- Perception relies on a configurable number of rays cast outwards from the ant's center in a circular pattern.
- Each ray travels up to a maximum distance.
- If a ray hits an entity (Wall, another Ant, the Object), it returns:
  - **Normalized Distance:** A value between [0, 1], where 1 means touching/very close, and 0 means hit at max distance or no hit.
  - **Class Information:** An encoding representing the type of entity hit (e.g., Wall, Ant, Object). If nothing is hit, a 'Null' class is returned.
- This provides the ant with local awareness of its immediate surroundings.

## Simulation & Training

- **Simulation:** Madrona will be used for its high-performance parallel simulation capabilities, suitable for MARL and efficient visualization.
- **Training:** PyTorch will be the framework for defining the neural network models and implementing the reinforcement learning algorithm (e.g., PPO, SAC adapted for multi-agent parameter sharing).

---

### LATER

(Add further details and extensions here)

- Curriculum Learning (e.g., varying object shapes/weights, number of ants).
- More sophisticated MARL algorithms (e.g., MAPPO, VDN, QMIX if applicable).
- Refined reward shaping (e.g., penalties for collision, bonus for pushing object).
- Analysis of emergent communication protocols in the cloud messages.
- Different object types and interaction physics.
