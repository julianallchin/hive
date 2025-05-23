#pragma once

#include <madrona/types.hpp>

namespace madEscape {

namespace consts {
inline constexpr bool overrideActionsWithRandom = true;
// Ant population constraints
inline constexpr madrona::CountT minAnts = 0;     // Minimum number of ants
inline constexpr madrona::CountT maxAnts = 100;    // Maximum number of ants

// Movable objects constraints
inline constexpr madrona::CountT minMovableObjects = 0;  // Minimum number of movable objects
inline constexpr madrona::CountT maxMovableObjects = 5; // Maximum number of movable objects

// Interior walls constraints
inline constexpr madrona::CountT minWalls = 0;   // Minimum number of interior walls
inline constexpr madrona::CountT maxWalls = 5;   // Maximum number of interior walls

// world/borders
inline constexpr float worldLength = 40.0f;
inline constexpr float worldWidth = 40.0f;
inline constexpr float borderWidth = 0.5f;
inline constexpr float borderHeight = 5.0f;
inline constexpr float minBorderSpawnBuffer = 3.0f;
inline constexpr float maxBorderSpawnBuffer = 6.0f;
inline constexpr int maxWallPlacementAttempts = 30;
// ants
inline constexpr float antSize = 1.0f;
inline constexpr float grabRange = 0.1f;
inline constexpr float antMacguffinBuffer = 0.0f;
inline constexpr float antMovableObjectBuffer = 0.0f;
inline constexpr float antWallBuffer = 0.0f;
inline constexpr float antAntBuffer = 0.0f;
inline constexpr int maxAntPlacementAttemptsPerAnt = 30;
inline constexpr float antInverseMass = 1.0f;
// macguffin
inline constexpr float macguffinSize = 4.0f;
inline constexpr float macguffinInverseMass = 0.025f;
// goal
inline constexpr float goalSize = 4.0f;
// internal walls
inline constexpr float minWallLength = 5.0f;
inline constexpr float maxWallLength = 30.0f;
inline constexpr float wallHeight = 2.0f;
inline constexpr float wallMacguffinBuffer = 1.0f;
inline constexpr float wallGoalBuffer = 5.0f;
inline constexpr float wallWallBuffer = 3.0f;
inline constexpr int maxMovableObjectPlacementAttempts = 30;
// internal movable objects (eg cube obstructions)
inline constexpr float movableObjectMacguffinBuffer = 1.0f;
inline constexpr float movableObjectWallBuffer = 1.0f;
inline constexpr float movableObjectObjectBuffer = 1.0f;
inline constexpr float movableObjectSize = 2.0f; // default 2x2x2
inline constexpr float movableObjectMinScale = 0.5f; // random scaling factor on top of default
inline constexpr float movableObjectMaxScale = 2.0f; // random scaling factor on top of default
inline constexpr float movableObjectInverseMass = 0.05f;

// mesh default values
// (when an object is created, its true size is the scale param * default mesh size)
// don't touch these unless you use different meshes (ie .obj files)
inline constexpr float macguffinMeshSize = 2.0f;
inline constexpr float goalMeshSize = 2.0f;
inline constexpr float movableObjectMeshSize = 2.0f;
inline constexpr float antMeshSize = 2.5f;
inline constexpr float wallMeshHeight = 2.5f; // length and width are 1
inline constexpr float borderMeshHeight = 2.5f; // length and width are 1
inline constexpr float borderMeshX = 1.0f;
inline constexpr float borderMeshY = 1.0f;
inline constexpr float wallMeshY = 1.0f;
inline constexpr float wallMeshX = 1.0f;


// Reward for decreasing distance between macguffin and goal
inline constexpr float distanceRewardScale = 0.1f;
// Reward for successfully moving macguffin to goal
inline constexpr float goalReward = 1.0f;
// Small existential penalty per timestep
inline constexpr float existentialPenalty = -0.001f;

// Steps per episode
inline constexpr int32_t episodeLen = 1000;  // Longer episodes for the hive task

// Default values for curriculum learning
inline constexpr uint32_t defaultAnts = 20;             // Start with fewer ants
inline constexpr uint32_t defaultMovableObjects = 0;    // Start with no movable objects
inline constexpr uint32_t defaultWalls = 0;            // Start with no interior walls

// Distance threshold for considering macguffin at goal
inline constexpr float goalDistanceThreshold = 2.0f;

// How many discrete options for actions
inline constexpr madrona::CountT numMoveAmountBuckets = 4;
inline constexpr madrona::CountT numMoveAngleBuckets = 8;
inline constexpr madrona::CountT numTurnBuckets = 5;

// Number of lidar samples, arranged in circle around agent
inline constexpr madrona::CountT numLidarSamples = 30;

// Time (seconds) per step
inline constexpr float deltaT = 0.04f;

// Number of physics substeps
inline constexpr madrona::CountT numPhysicsSubsteps = 4.f;

inline constexpr float gravity = 9.8f; // positive values give downward gravity

}

}
