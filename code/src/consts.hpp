#pragma once

#include <madrona/types.hpp>

namespace madEscape {

namespace consts {
// Ant population constraints
inline constexpr madrona::CountT minAnts = 1;     // Minimum number of ants
inline constexpr madrona::CountT maxAnts = 100;    // Maximum number of ants

// Movable objects constraints
inline constexpr madrona::CountT minMovableObjects = 0;  // Minimum number of movable objects
inline constexpr madrona::CountT maxMovableObjects = 10; // Maximum number of movable objects

// Interior walls constraints
inline constexpr madrona::CountT minInteriorWalls = 0;   // Minimum number of interior walls
inline constexpr madrona::CountT maxInteriorWalls = 5;   // Maximum number of interior walls

// Various world / entity size parameters
inline constexpr float worldLength = 40.f;
inline constexpr float worldWidth = 40.f;  // Making it square since we don't have separate rooms
inline constexpr float wallWidth = 1.f;
inline constexpr float antRadius = 0.5f;    // Smaller than original agents
inline constexpr float macguffinRadius = 1.f;
inline constexpr float movableObjectRadius = 1.f;
inline constexpr float goalRadius = 1.5f;   // Visual indicator of goal position

// Reward for decreasing distance between macguffin and goal
inline constexpr float distanceRewardScale = 0.1f;
// Reward for successfully moving macguffin to goal
inline constexpr float goalReward = 1.0f;
// Small existential penalty per timestep
inline constexpr float existentialPenalty = -1.0f;

// Steps per episode
inline constexpr int32_t episodeLen = 300;  // Longer episodes for the hive task

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

}

}
