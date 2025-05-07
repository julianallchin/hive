# High-Level Strategy

1.  **Simplify Core Logic:** Remove the room-based progression, buttons, and doors from Escape Room (unless you specifically want doors/buttons as obstacles later).
2.  **Introduce New Entities:** Ants, MacGuffin.
3.  **Implement SWARM Mechanics:** Ant movement (target velocity), MacGuffin physics, communication pipeline.
4.  **Modify Observation/Action Spaces:** Adapt to SWARM's needs.
5.  **Update Reward System:** Focus on MacGuffin-to-goal distance.

---

**I. ECS: Entities, Components, and Archetypes for SWARM**

Let's define the core "things" and their data.

**1. Components:**

- **Madrona Standard Components (from `madrona/components.hpp`, `madrona/physics.hpp`, etc.):**

  - `Position`, `Rotation`, `Scale`, `ObjectID`
  - `Velocity`, `ResponseType`, `ExternalForce`, `ExternalTorque`
  - `RigidBody` (bundle component including many of the above)
  - `madrona::render::Renderable`
  - `madrona::render::RenderCamera` (if ants have individual camera views, though your description sounds like top-down global view for rendering, and agent-local raycasts for observation)

- **SWARM-Specific Components (you'll define these in `src/types.hpp`):**
  - `EntityType`: (Enum: `Ant`, `MacGuffin`, `Wall`, `MovableObstacle`, `None`) - To identify entity types, especially for raycasts.
  - `Action`:
    - `targetVelocity`: `math::Vector2` (for ants)
    - `sentMessage`: `math::Vector<16>` (for ants, the message they broadcast)
  - `Observation`: (These will be bundled per-ant)
    - `SelfStateObs`: `math::Vector4` (pos x,y, vel x,y)
    - `TaskVectorObs`: `math::Vector4` (relative pos to MacGuffin, relative pos to Goal)
    - `RaycastObs`: `math::Vector<NUM_RAYS * RAY_FEATURES>` (e.g., 6 rays \* (dist, type_encoded) = 12 floats)
    - `ReceivedMessageObs`: `math::Vector<16>` (the $h^t$ global memory broadcast to this ant)
  - `Reward`: `float` (per-ant)
  - `Done`: `int32_t` (per-ant)
  - `WorldReset`: `int32_t` (singleton, same as Escape Room)
  - `EpisodeInfo`: (Singleton)
    - `targetGoalPosition`: `math::Vector3`
    - `macGuffinStartPos`: `math::Vector3`
    - `stepsRemaining`: `int32_t`
  - `MacGuffinState`: (Component for the MacGuffin entity)
    - Could be empty if all its state is covered by `RigidBody`.
  - `CommunicationBuffer`: (Singleton)
    - `allAntMessages`: `HeapArray<math::Vector<16>>` (stores $m^{t-1}_i$ for all ants $i$)
    - `aggregatedGlobalVector`: `math::Vector<16>` (stores $g^{t-1}$)
    - `lstmHiddenState`: `math::Vector<LSTM_HIDDEN_DIM>` (stores $(h^t, c^t)$ - you'll need to decide how to represent this tuple as a flat vector or two vectors)

**2. Archetypes:**

- `Ant`:
  - `RigidBody`
  - `EntityType`
  - `Action`
  - `SelfStateObs`, `TaskVectorObs`, `RaycastObs`, `ReceivedMessageObs` (or a combined `AgentObservation` struct)
  - `Reward`, `Done`
  - `madrona::render::Renderable`
  - (Optional) `madrona::render::RenderCamera`
- `MacGuffin`:
  - `RigidBody`
  - `EntityType`
  - `MacGuffinState` (if needed)
  - `madrona::render::Renderable`
- `Wall`: (Static Obstacle)
  - `RigidBody` (with `ResponseType::Static`)
  - `EntityType`
  - `madrona::render::Renderable`
- `MovableObstacle`: (e.g., a pushable block, distinct from MacGuffin)
  - `RigidBody` (with `ResponseType::Dynamic`)
  - `EntityType`
  - `madrona::render::Renderable`
- **(Optional) `DoorEntity`, `ButtonEntity`**: If you re-introduce these from Escape Room.

---

**II. Task Graph Setup for SWARM**

The order of systems is crucial, especially for the communication loop.

1.  **Reset & Setup (Start of Step):**

    - `ResetSystem(WorldReset&, EpisodeInfo&, CommunicationBuffer&)`:
      - Checks `Done` flags or `WorldReset` signal.
      - If reset: Cleans up old dynamic entities (MacGuffin, MovableObstacles if any from previous episode).
      - Calls `generateWorld` (places Ants, MacGuffin, sets `EpisodeInfo.targetGoalPosition`).
      - Resets `CommunicationBuffer.allAntMessages`, `.aggregatedGlobalVector`, `.lstmHiddenState` to zeros.
      - Resets `EpisodeInfo.stepsRemaining`.

2.  **Communication Pipeline (Processing previous step's messages):**

    - *Assumption: `Action.sentMessage` was populated by the policy at the end of the *previous* simulation step and is available now.*
    - `GatherMessagesSystem(forEach Ant: Action.sentMessage, &CommunicationBuffer)`:
      - Iterates over all ants.
      - Copies each ant's `Action.sentMessage` (which is $m_i^{t-1}$) into `CommunicationBuffer.allAntMessages`.
      - _Note:_ This system reads the `Action` component that Python wrote to at the end of the last step. The `Action` component will be overwritten later in _this_ step by the new policy output.
    - `AggregateMessagesSystem(CommunicationBuffer&)`: (This might be tricky to do fully in Madrona ECS if complex attention is needed. A simpler aggregation like averaging is easier. For full self-attention as in your diagram, this might partially happen in Python or be a very custom GPU kernel if you're ambitious).
      - Input: `CommunicationBuffer.allAntMessages`.
      - Process: Performs attention/aggregation to compute $g^{t-1}$.
      - Output: `CommunicationBuffer.aggregatedGlobalVector`.
    - `GlobalLSTMSystem(CommunicationBuffer&)`:
      - Input: `CommunicationBuffer.aggregatedGlobalVector` ($g^{t-1}$), `CommunicationBuffer.lstmHiddenState` (which holds $h^{t-1}, c^{t-1}$).
      - Process: LSTM forward pass.
      - Output: Updates `CommunicationBuffer.lstmHiddenState` to $h^t, c^t$.

3.  **Observation Collection (For current step's decision):**

    - `BroadcastMemorySystem(forEach Ant: &ReceivedMessageObs, CommunicationBuffer)`:
      - Copies $h^t$ from `CommunicationBuffer.lstmHiddenState` to each ant's `ReceivedMessageObs` component.
    - `CollectAgentObservationsSystem(forEach Ant: Position, Velocity, &SelfStateObs, &TaskVectorObs, &RaycastObs, EpisodeInfo, Position [MacGuffin])`:
      - Calculates local state for `SelfStateObs`.
      - Calculates relative vectors to MacGuffin (queried by its entity or a known singleton) and `EpisodeInfo.targetGoalPosition` for `TaskVectorObs`.
      - Performs raycasts (like `lidarSystem` in Escape Room) and populates `RaycastObs`. Needs `broadphase::BVH` to be built (see Physics section).

4.  **Action Application & Physics:**

    - _At this point, Python/training code receives all observation components (SelfStateObs, TaskVectorObs, RaycastObs, ReceivedMessageObs) and provides new `Action` (targetVelocity, sentMessage) values._ Madrona's job is to write these into the `Action` component for each ant.
    - `AntMovementSystem(forEach Ant: Action.targetVelocity, Rotation, &ExternalForce, &ExternalTorque)`:
      - Reads `Action.targetVelocity`.
      - Converts it into forces/torques to apply to the ant. (Similar to `movementSystem` in Escape Room but for velocity control).
    - **(Optional) `MovableObstacleInteractionSystem`**: If you have doors/buttons, this is where their logic (`buttonSystem`, `doorOpenSystem`, `setDoorPositionSystem` from Escape Room) would go.
    - **Physics Sub-pipeline (Standard Madrona):**
      - `phys::PhysicsSystem::setupBroadphaseTasks()`: Builds BVH. _Crucial: this must run AFTER entities are placed/moved by reset or game logic but BEFORE raycasts or collision queries._ You might need two broadphase builds: one early for raycasts if positions are stable before agent actions, and one after agent actions for physics. Or, ensure raycasts use the BVH from the _previous_ step's physics update if agent actions don't drastically change the queryable environment before observations are made. For SWARM, one BVH build after `ResetSystem` and before `CollectAgentObservationsSystem` might be sufficient for raycasts, and then the main physics step will use that and update it.
      - `phys::PhysicsSystem::setupPhysicsStepTasks()`: Collision detection, solver.
      - `phys::PhysicsSystem::setupCleanupTasks()`.

5.  **Reward and Episode Termination:**
    - `RewardSystem(forEach Ant: &Reward, Position [MacGuffin], EpisodeInfo)`:
      - Calculates distance $d_t = \| \text{MacGuffin}_t - \text{goal} \|_2$.
      - Calculates reward based on $d_{t-1} - d_t$, goal achievement, existential penalty.
      - Updates `Reward` component.
      - Updates $d_{t-1}$ for the next step (could store in `EpisodeInfo` or a component on MacGuffin).
    - `StepTrackerSystem(forEach Ant: &Done, EpisodeInfo, Position [MacGuffin])`:
      - Decrements `EpisodeInfo.stepsRemaining`.
      - Checks if MacGuffin reached `EpisodeInfo.targetGoalPosition` or `stepsRemaining == 0`.
      - Sets `Done` flags.

**Task Graph Node Order Sketch:**

```
(External: Python provides Action_prev.sentMessage for each Ant)
1. ResetSystem (if needed, calls generateWorld)
   -> IF RESET: BroadphaseUpdate (after new entity placement)
2. GatherMessagesSystem (Ant.Action_prev.sentMessage -> CommBuffer.allAntMessages)
3. AggregateMessagesSystem (CommBuffer.allAntMessages -> CommBuffer.aggregatedGlobalVector)
4. GlobalLSTMSystem (CommBuffer.aggregatedGlobalVector, CommBuffer.lstmHiddenState_prev -> CommBuffer.lstmHiddenState_curr)
5. BroadcastMemorySystem (CommBuffer.lstmHiddenState_curr -> Ant.ReceivedMessageObs)
6. CollectAgentObservationsSystem (Ant physics state, relative states, raycasts -> Ant observation components)
   -> Depends on Broadphase being up-to-date if raycasting against dynamic world.

(External: Python receives all Ant Observation components, computes new Action_curr {targetVel, sentMessage_curr}, writes them to Ant.Action)

7. AntMovementSystem (Ant.Action_curr.targetVelocity -> Ant.ExternalForce/Torque)
8. (Optional Movable Obstacle Logic Systems)
9. Physics Pipeline (Broadphase, Step, Cleanup) -> Updates all entity Positions, Velocities
10. RewardSystem (MacGuffin pos, Goal pos -> Ant.Reward)
11. StepTrackerSystem (Episode conditions -> Ant.Done, EpisodeInfo.stepsRemaining--)
```

The tricky part is the loop: messages from step $t-1$ (produced by policy) influence observations for step $t$, which then produces actions and messages for step $t$. The Python side will manage getting `sentMessage` from the policy output and feeding it back into the simulation for the _next_ step (likely by writing it to the `Action` component that `GatherMessagesSystem` reads next iteration).

---

**III. Changes to Escape Room Files:**

- **`src/types.hpp`:**

  - **Remove:** `Room`, `LevelState`, `DoorProperties`, `OpenState`, `ButtonState`, `GrabState`, `Progress`, `OtherAgents`, `RoomEntityObservations`, `DoorObservation`, `Lidar` (replace with `RaycastObs`), specific observation structs like `SelfObservation` (replace with your new ones).
  - **Add:** Your new components listed above (`EntityType`, SWARM `Action`, `SelfStateObs`, `TaskVectorObs`, `RaycastObs`, `ReceivedMessageObs`, `EpisodeInfo`, `MacGuffinState`, `CommunicationBuffer`).
  - **Modify Archetypes:** Define `Ant`, `MacGuffin`, `Wall`, `MovableObstacle` using the new components. Remove Escape Room archetypes.

- **`src/sim.hpp` & `src/sim.cpp`:**

  - **`Sim::Config`:** Remove `renderBridge` if not using agent-specific cameras. (Though batch rendering might still use a bridge).
  - **`Sim` struct (WorldBase data):**
    - Remove `floorPlane`, `borders`, `agents` arrays (entities will be created dynamically or queried).
    - Remove `curWorldEpisode`, `rng` if not using complex procedural generation per world beyond initial setup.
    - Add any global world data not fitting in singletons, e.g., reference to the MacGuffin entity if there's only one and it's easier to track directly.
  - **`Sim::registerTypes`:** Register all your new components and archetypes. Unregister old ones. Update `ExportID` enum and export calls for new observations/actions.
  - **`Sim::setupTasks`:** This is a major rewrite. Implement the new task graph logic described above.
    - Remove systems like `movementSystem` (replace with `AntMovementSystem`), `grabSystem`, `setDoorPositionSystem`, `buttonSystem`, `doorOpenSystem`, `agentZeroVelSystem`, `collectObservationsSystem` (replace with SWARM version), `lidarSystem` (replace with raycast part of your obs collection), `rewardSystem` (replace), `bonusRewardSystem`.
    - Add your new systems: `GatherMessagesSystem`, `AggregateMessagesSystem`, `GlobalLSTMSystem`, `BroadcastMemorySystem`, `CollectAgentObservationsSystem`, `AntMovementSystem`, SWARM `RewardSystem`.
  - **Constructor `Sim::Sim(...)`:**
    - Simplify. Call `createPersistentEntities` (for walls) and `initWorld` (or `generateWorld`).
    - Physics init will be similar.
  - **`initWorld` / `resetSystem`:** Implement logic to place Ants, MacGuffin, and set `EpisodeInfo`.
  - **`cleanupWorld`:** Destroy MacGuffin, dynamic MovableObstacles. Ants might persist and be reset.

- **`src/level_gen.hpp` & `src/level_gen.cpp`:**

  - `createPersistentEntities`: Can be used to create the outer bounding Walls. Ants might also be persistent and just reset.
  - `generateWorld`:
    - Remove all room generation logic.
    - Implement logic to:
      - Place the MacGuffin at `EpisodeInfo.macGuffinStartPos`.
      - Place/reset Ants around their starting area.
      - Set `EpisodeInfo.targetGoalPosition`.
      - (Optional) Randomly place a few `MovableObstacle` entities.
    - No need for `Room` struct or iterating through rooms.

- **`src/consts.hpp`:**

  - Update `numAgents` (make it configurable if possible, though fixed at compile time is easier initially).
  - Remove room-specific constants.
  - Add constants for SWARM: `NUM_RAYS`, `RAY_FEATURES`, `MESSAGE_DIM`, `LSTM_HIDDEN_DIM`, MacGuffin properties (mass, friction), world boundaries.

- **`src/mgr.hpp` & `src/mgr.cpp`:**

  - **`ExportID` enum:** Update to match the new tensors you're exporting (e.g., `SelfStateObsTensor`, `TaskVectorObsTensor`, `RaycastObsTensor`, `ReceivedMessageObsTensor`, `AntActionTensor` which includes `targetVelocity` and `sentMessage`).
  - **Tensor export functions:** Add/modify functions like `selfStateObsTensor()`, `raycastObsTensor()`, etc.
  - **Asset Loading (`loadRenderObjects`, `loadPhysicsObjects`):**
    - Change paths to load assets for Ant, MacGuffin, Walls.
    - Update `SimObject` enum.

- **`src/bindings.cpp`:**

  - Expose the new tensor-exporting functions from the `Manager` to Python.

- **Python side (`scripts/` and `train_src/`):**
  - **`scripts/policy.py`:**
    - `setup_obs`: Adapt to the new set of observation tensors from `SimManager`.
    - `process_obs`: Concatenate/process the new observation tensors.
    - `make_policy`: The policy network (ActorCritic) will now need to:
      - Take the `ReceivedMessageObs` as an input.
      - Output both `targetVelocity` and `sentMessage`. This might mean your `LinearLayerDiscreteActor` needs to become/be accompanied by a `LinearLayerContinuousActorAndMessageEmitter`.
  - **`scripts/train.py`:**
    - The main training loop will now orchestrate the higher-level communication:
      1.  `sim.step()` (Madrona runs its task graph, including `GatherMessagesSystem`, `AggregateMessagesSystem`, `GlobalLSTMSystem`, `BroadcastMemorySystem`, `CollectAgentObservationsSystem`).
      2.  Retrieve all observation tensors from `sim`, including the `ReceivedMessageObs` (which is $h^t$).
      3.  Pass these observations to the PyTorch `ActorCritic` policy.
      4.  The policy outputs `actions` (target velocities) and `new_messages` ($m_i^t$).
      5.  Send these `actions` and `new_messages` back to the simulation via the `sim.action_tensor()` (you'll need to decide how `action_tensor` is structured to accept both velocity and message, or have a separate `message_tensor()`).
    - The global LSTM and aggregation (if complex attention) would be PyTorch modules called within this Python training loop, operating on the `allAntMessages` (if you export that tensor) or the `aggregatedGlobalVector` (if Madrona does a simpler aggregation).
  - **`train_src/madrona_escape_room_learn/action.py`:**
    - `DiscreteActionDistributions` will likely need to be replaced or augmented if your `targetVelocity` is continuous and `sentMessage` is also continuous. You might need separate distributions or a single multivariate Gaussian.
  - **`train_src/madrona_escape_room_learn/actor_critic.py`:**
    - `Actor` part of `ActorCritic` needs to output both target velocity and the message.
    - The `Backbone` will need to incorporate the `ReceivedMessageObs` into its feature extraction.

---

**Key Challenges & Considerations:**

1.  **Communication Loop Implementation:** The most significant change is the explicit communication loop ($m \rightarrow g \rightarrow h \rightarrow \text{obs} \rightarrow \text{policy} \rightarrow m', a'$). You need to decide how much of the $m \rightarrow g \rightarrow h$ pipeline happens in Madrona's C++ task graph versus PyTorch modules in `scripts/train.py`.
    - **Madrona-centric:** Simpler aggregations (sum/average) and LSTM can be ECS systems. `CommunicationBuffer` singleton is key.
    - **Python-centric:** Export `allAntMessages` tensor. Python script runs PyTorch attention & LSTM modules. Then passes $h^t$ back to sim (e.g. via a dedicated tensor or part of the action tensor). This offers more flexibility for complex PyTorch models but adds data transfer.
      Your diagram suggests the LSTM and attention are global PyTorch modules, so Python-centric seems likely.
2.  **Continuous Actions:** Escape Room uses discrete actions. Your ants have continuous target velocity and messages. You'll need to adapt the policy output layers and action distribution handling in `train_src`.
3.  **Dynamic Agent Count:** If you want truly dynamic agent counts _without retraining_, the policy and communication aggregation must be designed to handle variable numbers of inputs/outputs (e.g., attention mechanisms are good for this). Madrona itself can handle dynamic entity counts.
4.  **Asset Creation:** You'll need 3D models and collision meshes for Ants and the MacGuffin.
5.  **Debugging:** Start simple. Get one ant moving. Then add the MacGuffin. Then multiple ants. Then communication.

This is a substantial refactor, but by breaking it down and leveraging the Escape Room's structure, it's definitely achievable! Good luck!
