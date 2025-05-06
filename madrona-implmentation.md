The core idea is to define your simulation's state (data) via ECS components and archetypes, and its logic (behavior) via ECS systems orchestrated by a task graph.

## 1. Defining Simulation State: Components and Archetypes (`src/types.hpp`)

You'll define C++ structs for your components and then combine them into archetypes.

**Key Entities & Their Potential Components:**

- **Ant (`Agent` Archetype):**

  - `madrona::phys::RigidBody`: Includes `Position`, `Rotation` (might be 2D specific, so Z-axis rotation), `Velocity`, `Scale`, `ObjectID`, `ExternalForce`, `ExternalTorque`, etc. Ants need to physically interact.
  - `ActionIn`: (Input from Policy)
    - `math::Vector2 targetVelocity`: The desired velocity vector for the ant.
    - `math::Vector<float, MSG_DIM> messageToBroadcast`: The 16-dim message vector.
  - `ObservationOut`: (Output to Policy)
    - `math::Vector2 localPosition`: Current ant's (x,y).
    - `math::Vector2 localVelocity`: Current ant's (vx, vy).
    - `math::Vector2 relativeToObject`: Vector from ant to MacGuffin.
    - `math::Vector2 relativeToGoal`: Vector from ant to Goal.
    - `RaycastObservation raycasts[NUM_RAYS]`: Each `RaycastObservation` could be `{ float distance; EntityType hitType; }`.
    - `math::Vector<float, MSG_DIM> receivedAggregatedMessage`: The cloud communication input $h^t$.
  - `EntityType`: An enum value like `EntityType::Ant`.
  - `AgentID`: Potentially an integer ID for this agent (0 to N-1 within its world).
  - `madrona::render::Renderable`: If you want to visualize them.
  - `madrona::render::RenderCamera`: If you want first-person ant views (less likely for top-down).

- **MacGuffin (`MacGuffinEntity` Archetype):**

  - `madrona::phys::RigidBody`: For its position, velocity, mass, friction, etc. It's dynamic.
  - `EntityType`: `EntityType::MacGuffin`.
  - `madrona::render::Renderable`.

- **Walls (`StaticObstacleEntity` Archetype):**

  - `madrona::phys::RigidBody`: Position, rotation, scale. Crucially, `ResponseType` will be `Static`.
  - `EntityType`: `EntityType::Wall`.
  - `madrona::render::Renderable`.

- **Movable Obstacles (e.g., Blocks) (`DynamicObstacleEntity` Archetype):**

  - `madrona::phys::RigidBody`: Dynamic, pushable.
  - `EntityType`: `EntityType::MovableBlock`.
  - `madrona::render::Renderable`.

- **Doors (`DoorEntity` Archetype - more complex):**
  - `madrona::phys::RigidBody`: Could be static or kinematic.
  - `OpenState`: (Like in escape room) `bool isOpen`.
  - `DoorProperties`: How it opens (e.g., triggered by MacGuffin proximity, ant action).
  - `EntityType`: `EntityType::Door`.
  - `madrona::render::Renderable`.

**Singleton Components (Per-World Global State):**

- `WorldReset`: Standard Madrona component to signal a reset.
- `GoalConfigSingleton`:
  - `math::Vector2 goalPosition`.
  - (Potentially) `math::Vector2 goalVelocity` if it's a moving goal.
- `EpisodeInfoSingleton`:
  - `int32_t stepsRemaining`.
  - `float currentReward`.
  - `bool episodeDone`.
- `SimConfigSingleton`: (If parameters like object mass, friction vary per episode but are fixed within an episode)
  - `float macGuffinMass`.
  - `float macGuffinFriction`.
  - `int32_t numAntsInWorld`.

**`EntityType` Enum:**

```cpp
// In types.hpp or consts.hpp
enum class EntityType : uint32_t {
    None,
    Ant,
    MacGuffin,
    Wall,
    MovableBlock,
    Door,
    // ... other types
    NumTypes,
};
```

**Communication Handling with Python/Torch:**
Your "Global LSTM" and the attention mechanism over all messages $m^{t-1}$ to produce $g^{t-1}$ and then $h^t$ will likely live in your Python training code, not directly as a Madrona system operating across all batched worlds simultaneously.
So, Madrona will:

1.  Export all `messageToBroadcast` from ants at step `t`.
2.  Python receives these, runs attention, runs the global LSTM, gets $h^{t+1}$.
3.  Python sends $h^{t+1}$ back to Madrona.
4.  Madrona makes $h^{t+1}$ available to ants as `receivedAggregatedMessage` for step `t+1`.

This means you'll need tensors for:

- `AllAntMessagesOut` (Tensor of all `messageToBroadcast` from all ants in a world, or across all worlds).
- `BroadcastMemoryIn` (Tensor of $h^t$ to be made available to ants).

You could have a singleton `BroadcastMemorySingleton` in Madrona that Python writes to, and ants read from.

## 2. Defining Simulation Logic: Systems and Task Graph (`src/sim.cpp`)

In `Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &cfg)`, you'll define the systems and their dependencies.

**Key Systems:**

1.  **`resetSystem` / `levelGenSystem`:**

    - Input: `WorldReset` singleton.
    - Output: Populates the world with ants, MacGuffin, goal, walls, obstacles. Sets initial positions, velocities. Resets `EpisodeInfoSingleton`.
    - Logic: Called when `WorldReset::reset` is true or `EpisodeInfoSingleton::episodeDone` is true (if auto-reset). Uses `generateWorld()` from `level_gen.cpp`.

2.  **`agentActionSystem`:**

    - Input: `ActionIn` (from policy), `RigidBody` (for ant's current state).
    - Output: Modifies `ExternalForce` / `ExternalTorque` on the ant's `RigidBody` (or directly sets `Velocity` if your physics model allows).
    - Logic: Translates `targetVelocity` from `ActionIn` into physical forces/impulses. The `messageToBroadcast` part of `ActionIn` is effectively just written and will be exported.

3.  **`physicsStepSystemGroup`:** (Using `PhysicsSystem::setupPhysicsStepTasks`)

    - Input: `RigidBody` components of all physical entities.
    - Output: Updated `Position`, `Rotation`, `Velocity` for all dynamic `RigidBody` entities. Handles collisions.
    - Logic: Runs the Madrona physics engine.

4.  **`raycastSystem`:** (If using raycasts for ant perception)

    - Input: `Position`, `Rotation` (of ant), `broadphase::BVH` (built by physics).
    - Output: Populates `RaycastObservation` part of `ObservationOut`.
    - Logic: For each ant, casts N rays, finds closest hits, records distance and `EntityType`. Madrona's BVH can be used for efficient raycasting.

5.  **`doorLogicSystem`:** (If you have doors)

    - Input: `DoorProperties`, `OpenState`, `Position` (of MacGuffin, ants).
    - Output: Updates `OpenState::isOpen`.
    - Logic: Checks conditions (e.g., MacGuffin near a pressure plate, ant "pushes" a button conceptually).

6.  **`doorVisualUpdateSystem`:** (If doors animate)

    - Input: `OpenState`, `Position` (of door).
    - Output: Modifies `Position` (of door, e.g., moves it up/down).
    - Logic: Animates door based on `OpenState`.

7.  **`observationSystem`:**

    - Input: `RigidBody` (of ant, MacGuffin), `GoalConfigSingleton`, `BroadcastMemorySingleton` (this is $h^t$ from Python), `RaycastObservation` (if separate system).
    - Output: Populates `ObservationOut` for each ant.
    - Logic:
      - Fills `localPosition`, `localVelocity`.
      - Calculates `relativeToObject`, `relativeToGoal`.
      - Copies `raycasts` (if done by `raycastSystem`) or performs raycasts here.
      - Copies `receivedAggregatedMessage` from `BroadcastMemorySingleton` (which was updated by Python).

8.  **`rewardAndDoneSystem`:**
    - Input: `Position` (of MacGuffin), `GoalConfigSingleton`, `EpisodeInfoSingleton`.
    - Output: Updates `EpisodeInfoSingleton::currentReward`, `EpisodeInfoSingleton::episodeDone`.
    - Logic:
      - Calculates $d_t = \| \text{object}_t - \text{goal} \|_2$.
      - Calculates step reward: $0.1 (d_{t-1} - d_t)$. $d_{t-1}$ would need to be stored from previous step, e.g., in `EpisodeInfoSingleton`.
      - Adds goal reward if $d_t \le \epsilon$.
      - Applies existential penalty.
      - Sets `episodeDone` if goal reached or `stepsRemaining` is 0.
      - Decrements `stepsRemaining`.

**Simplified Task Graph Order (Conceptual):**

```mermaid
graph TD
    A[Reset System (Optional)] --> B(Level Gen System)
    B --> C{Policy Input Ready?}
    C -- Yes --> D[Agent Action System]
    D --> E[Physics Broadphase Setup]
    E --> F[Door Logic System (Optional)]
    F --> G[Physics Substeps & Solve]
    G --> H[Physics Cleanup]
    H --> I[Raycast System (Optional)]
    I --> J[Door Visual Update System (Optional)]
    J --> K[Observation System]
    K --> L[Reward and Done System]
    L --> M{Episode End?}
    M -- No --> C
    M -- Yes --> A
```

**Madrona Task Graph Nodes (`setupTasks`):**

1.  **`resetSystemNode`**: Handles `WorldReset` and calls `generateWorld`.
    - Dependencies: None (or previous step's `doneNode`).
2.  **`agentActionNode`**: `ParallelForNode` for `agentActionSystem`.
    - Dependencies: `resetSystemNode`.
3.  **`doorLogicNode`**: (If applicable) `ParallelForNode` for `doorLogicSystem`.
    - Dependencies: `agentActionNode` (or physics if doors react to physical state).
4.  **`physicsNodes`**: Group of nodes from `PhysicsSystem::setupBroadphaseTasks`, `PhysicsSystem::setupPhysicsStepTasks`, `PhysicsSystem::setupCleanupTasks`.
    - Dependencies: `agentActionNode` (forces applied), `doorLogicNode` (door states might affect physics).
5.  **`raycastNode`**: `ParallelForNode` for `raycastSystem`.
    - Dependencies: `physicsNodes` (needs updated BVH and positions).
6.  **`doorVisualUpdateNode`**: (If applicable) `ParallelForNode` for `doorVisualUpdateSystem`.
    - Dependencies: `doorLogicNode` (needs `OpenState`), `physicsNodes` (for door's own position if it's physically simulated).
7.  **`observationNode`**: `ParallelForNode` for `observationSystem`.
    - Dependencies: `physicsNodes`, `raycastNode`, `doorVisualUpdateNode`. This system needs all latest world state.
8.  **`rewardAndDoneNode`**: `ParallelForNode` for `rewardAndDoneSystem`.
    - Dependencies: `observationNode` (or directly `physicsNodes` if it only needs MacGuffin position).

GPU sorting nodes (`SortArchetypeNode`) would be added as in the escape room example if running on GPU.

## 3. Level Generation (`src/level_gen.cpp`)

The `generateWorld(Engine &ctx)` function will be responsible for:

- Reading `SimConfigSingleton` if parameters vary.
- Creating ant entities, setting their initial positions (e.g., random, around object).
- Creating the MacGuffin entity, setting its mass, friction, initial position.
- Setting the `GoalConfigSingleton::goalPosition`.
- Creating wall entities based on some level layout (procedural or fixed).
- Creating movable obstacle/door entities.
- Initializing `EpisodeInfoSingleton`.

## 4. Manager and Python Bindings (`src/mgr.hpp`, `src/mgr.cpp`, `src/bindings.cpp`)

This part is similar to `madrona_escape_room`.

- `Manager` class initializes Madrona, loads assets (if any visual/physics assets beyond primitives).
- Exports tensors for:
  - `ObservationOut` (for all ants).
  - `EpisodeInfoSingleton::currentReward` (or per-ant rewards).
  - `EpisodeInfoSingleton::episodeDone` (or per-ant dones).
  - `AllAntMessagesOut` (concatenated `messageToBroadcast` from all ants).
- Imports tensors for:
  - `ActionIn` (for all ants).
  - `BroadcastMemoryIn` (the $h^t$ vector from Python, to be written to `BroadcastMemorySingleton`).
  - `WorldReset`.

**The Training Loop Step in Python:**

1.  Get `ObservationOut_t`, `AllAntMessagesOut_t`, `Reward_{t-1}`, `Done_{t-1}` from Madrona.
2.  Use `AllAntMessagesOut_t` to compute $g^t$ (attention) and then $h^{t+1}$ (Global LSTM).
3.  Feed `ObservationOut_t` (which includes $h^t$ that was sent in the _previous_ Python step) to the policy network.
4.  Policy outputs `ActionIn_{t+1}` (target velocity & message to broadcast for _next_ Madrona step).
5.  Send `ActionIn_{t+1}` and `BroadcastMemoryIn` (which contains $h^{t+1}$) to Madrona.
6.  Madrona `mgr.step()`.

There's a one-step delay in the message aggregation and LSTM processing relative to the ant's action decision, which is typical.

## Tips for SWARM:

- **Start Simple:** Get a single ant pushing a block to a goal. Then add more ants. Then add walls. Then communication.
- **2D Physics:** Madrona physics is 3D. You can constrain movement to the XY plane by:
  - Setting Z components of forces/velocities to 0.
  - Setting inverse mass / inertia tensor components for Z-axis movement/rotation to 0 for entities you want to keep planar.
  - Using a large ground plane.
- **Variable Agent Count:** Your Python policy (especially the global LSTM and attention) needs to handle this. Madrona can spawn a variable number of agents per world if `level_gen.cpp` is set up for it, but the tensors exported (e.g., `ObservationOut`) will usually have a fixed maximum agent dimension for batching. You'd use padding or masking. The design of $h^t$ being broadcast implies all ants get the _same_ memory vector, which simplifies things.
- **Debugging:** The `viewer` is invaluable. Print tensor values. Madrona's error messages are generally helpful.

This detailed breakdown should give you a solid foundation for building SWARM with Madrona. Remember to iterate and test frequently!
