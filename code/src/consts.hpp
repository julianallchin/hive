#pragma once

#include <madrona/types.hpp>

namespace madEscape {

namespace consts {

// Generated levels assume 2 agents
<<<<<<< HEAD
inline constexpr int32_t minAgents = 50;
inline constexpr int32_t maxAgents = 100;
=======
inline constexpr int32_t minAgents = 2;
inline constexpr int32_t maxAgents = 2;
>>>>>>> f80edf18427f56774be0e694d0e8769e056a4b19

// Maximum number of interactive objects per challenge room. This is needed
// in order to setup the fixed-size learning tensors appropriately.
inline constexpr int32_t minCubes = 0;
inline constexpr int32_t maxCubes = 1;
inline constexpr int32_t minBarriers = 0;
inline constexpr int32_t maxBarriers = 1;


inline constexpr madrona::CountT maxTotalEntities = 
        maxAgents + maxCubes + maxBarriers + 6; // 6 for 4 side walls + floor + episodeTracker

// Various world / entity size parameters
inline constexpr float worldLength = 40.f;
inline constexpr float worldWidth = 40.f;

// inline constexpr float wallWidth = 1.f;
inline constexpr float borderWidth = 0.5f;
inline constexpr float borderHeight = 5.0f;
inline constexpr float minBorderSpawnBuffer = 3.0f;
inline constexpr float maxBorderSpawnBuffer = 6.0f;
inline constexpr int maxWallPlacementAttempts = 30;

inline constexpr float barrierWidth = 1.f;
inline constexpr float barrierHeight = 3.f;
inline constexpr float minBarrierLength = 5.0f;
inline constexpr float maxBarrierLength = 30.0f;
inline constexpr float barrierMacguffinBuffer = 1.0f;
inline constexpr float barrierGoalBuffer = 5.0f;
inline constexpr float barrierBarrierBuffer = 3.0f;
inline constexpr int maxBarrierPlacementAttempts = 30;

inline constexpr float cubeSize = 3.f;
inline constexpr float cubeInverseMass = 0.05f;
inline constexpr float cubeMacguffinBuffer = 1.0f;
inline constexpr float cubeBarrierBuffer = 1.0f;
inline constexpr float cubeCubeBuffer = 1.0f;
inline constexpr float movableObjectSize = 2.0f;
inline constexpr float cubeMinScaleFactor = 0.5f;
inline constexpr float cubeMaxScaleFactor = 2.0f;
inline constexpr int maxCubePlacementAttempts = 30;

inline constexpr float agentRadius = 0.5f;
inline constexpr float agentSize = (2 * agentRadius);
inline constexpr float agentMoveSpeed = 100.0f;
inline constexpr float agentTurnSpeed = 32.0f;
inline constexpr float agentInverseMass = 5.0f;
inline constexpr float agentMacguffinBuffer = 0.0f;
inline constexpr float agentCubeBuffer = 0.0f;
inline constexpr float agentBarrierBuffer = 0.0f;
inline constexpr float agentAgentBuffer = 0.0f;
inline constexpr int maxAgentPlacementAttemptsPerAgent = 30;

inline constexpr float macguffinSize = 5.0f;
inline constexpr float macguffinInverseMass = 0.05f;

inline constexpr float goalSize = 3.0f;

// Each unit of distance forward (+ y axis) rewards the agents by this amount
inline constexpr float distanceRewardScale = 100.0f;
// Each step that the agents don't make additional progress they get a small
// penalty reward
inline constexpr float existentialPenalty = -0.05f;
// reward for completing task successfully
inline constexpr float goalReward = 100.0f;

// Steps per episode
inline constexpr int32_t episodeLen = 500;

// How many discrete options for actions
inline constexpr madrona::CountT numMoveAmountBuckets = 4;
inline constexpr madrona::CountT numMoveAngleBuckets = 8;
inline constexpr madrona::CountT numTurnBuckets = 5;

// Number of lidar samples, arranged in circle around agent
inline constexpr madrona::CountT numLidarSamples = 30;

// Time (seconds) per step
inline constexpr float deltaT = 0.04f;

// Speed at which doors raise and lower
// inline constexpr float doorSpeed = 30.f;

// Number of physics substeps
inline constexpr madrona::CountT numPhysicsSubsteps = 4.f;

// mesh default values
// (when an object is created, its true size is the scale param * default mesh size)
// don't touch these unless you use different meshes (ie .obj files)
inline constexpr float cubeMeshSize = 2.0f;
inline constexpr float agentMeshSize = 2.5f;
inline constexpr float wallMeshHeight = 2.5f;   // length and width are 1       
inline constexpr float wallMeshX = 1.0f;
inline constexpr float wallMeshY = 1.0f;
}

}
