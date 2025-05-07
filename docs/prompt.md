# Summary

We are modifying this codebase to create our own environment and model. Instead of escape room, we are creating a hive simulation where ants try to move a macguffin to a goal. We want to keep the core physics and raycasting the same as the origional. We want to keep the types mostly the same, but we are tracking slightly different state and RL model. A single hivemind controls all the ants, and the hivemind is trying to move the "macguffin" to the goal.

# World:

During each world episode, ants (100 or so) are spawned in a room (top down, no z axis movement, altho rendered in 3d) and the macguffin and goal are in the same room. there can be movable objects and walls in the room as well. The reward function rewards the ants for moving the macguffin to the goal. The ants can communicate with each other via a global message space.

We are trying to modify the existing codebase to match our project. The project is a hive simulation where ants try to move a macguffin to a goal. We want to keep the core physics and raycasting the same as the origional. We want to keep the types mostly the same, but we are tracking slightly different state and RL model.

We want a range to randomize the number of ants (100 or so) and movable objects (10 or so) and walls (2 or so). But we want curriculum learning so it starts with no movable objects and walls and increases the number of movable objects and walls over time.

Remove excape room stuff, get rid of z cords. this is 2d only.

Per world, there is a single rectangle room, with four walls. There are no sub rooms, doors, buttons, partner obs, etc. There is one macguffin and one goal per world. There is one hive mind (not in madrona) per world that controls the many ants. Keep all reward, world reset, etc. The reward is for all the agents as one hivemind. Not per agent. No partner observations.

Ants should be able to grab the same way as the current implmentation.

Done is when it gets close enought to the goal or after T_max.

## Model:

**ants:**

- in:
  - local obs
  - task obs
  - shared hidden state
- out:
  - actions
  - send message
- model:
  - mlp

**hivemind**

- in: all ant messages
- out: ant hidden state
- model:
  - self attention over ant messages -> global message
  - LSTM over global message
  - MLP on LSTM hidden state

## Rewards:

Let $d_t = \| \text{object}_t - \text{goal} \|_2$ be object-goal distance.

- **Step reward**: $r_t^{\mathrm{step}} = 0.1\,(d_{t-1} - d_t)$
- **Goal reward**: $r_t^{\mathrm{goal}} = +1$ if $d_t \le \epsilon$
- **Existential penalty**: $r_t^{\mathrm{exist}} = -\frac{1}{T_{max}}$ per timestep

## Objects:

- ant
  - same action space as existing
  - aditional state for ant:
    - task state (relative polar to macguffin, relative polar to goal)
    - send message state (16D message, global const)
  - input:
    - self state obs
      - includes lidar.
    - task obs (relative polar to macguffin, relative polar to goal)
    - raycast obs
    - hidden state obs (32D, global const)
  - output:
    - same output as existing (movemtn and grab)
    - send message (16D)
- macguffin
  - position state
  - whatever needed for physics (basically a movable obstacle)
  - (in the future: randomized movement actions)
- wall (static obstacle)
  - position state
  - whatever needed for physics (basically a movable obstacle)
- target/goal (position, not physical)
  - position state
  - (in the future: randomized movement actions)
  - no physical body
- movable object (like cube)
  - position state
  - whatever needed for physics (basically a movable obstacle)
- world state
  - whatever's needed to link these and track overall state and model

Any additional state should be tracked for the world, communiation, lidar, etc.

There are no buttons or doors.
