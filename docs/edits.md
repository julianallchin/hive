## Understanding Codebase

Singletons (one per world):

- WorldReset
- LevelState
- HiveReward
- HiveDone
- StepsRemaining

Archetypes (collections of entities):

- Ant
- Macguffin
- Goal
- Wall
- MovableObject

## types.hpp

- removed hivemind messages (will be handled python side)

## sim.hpp

- used windsurf to get an implementation (did not do before) so probs the reason we had so many compiliing errors

## sim.cpp

## consts.hpp

- also didn't seem to have been pulled in correctly. Windsurf reimplemented while implementing sim.hpp

## level_gen.hpp

## level_gen.cpp

level_gen.cpp
