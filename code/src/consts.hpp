#pragma once

#include <madrona/types.hpp>

namespace madEscape
{

  // final
  namespace consts
  {
    inline constexpr int32_t minAgents = 5;
    inline constexpr int32_t maxAgents = 5;

    inline constexpr int32_t minCubes = 5;
    inline constexpr int32_t maxCubes = 5;
    inline constexpr int32_t minBarriers = 0;
    inline constexpr int32_t maxBarriers = 3;

    inline constexpr madrona::CountT maxTotalEntities =
        maxAgents + maxCubes + maxBarriers + 8; // 8 for 4 side walls + floor + episodeTracker + goal + macguffin

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
    inline constexpr float barrierBarrierBuffer = 8.0f;
    inline constexpr int maxBarrierPlacementAttempts = 30;

    inline constexpr float cubeSize = 1.5f;
    inline constexpr float cubeInverseMass = 0.2f;
    inline constexpr float cubeMacguffinBuffer = 1.0f;
    inline constexpr float cubeBarrierBuffer = 1.0f;
    inline constexpr float cubeCubeBuffer = 1.0f;
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

    inline constexpr float macguffinSize = 3.0f;
    inline constexpr float macguffinInverseMass = 0.05f;

    inline constexpr float goalSize = 3.0f;

    // Steps per episode
    inline constexpr int32_t episodeLen = 1000;

    // Each unit of distance forward (+ y axis) rewards the agents by this amount
    inline constexpr float distanceRewardScale = 1.0f;
    // Each step that the agents don't make additional progress they get a small
    // penalty reward
    inline constexpr float existentialPenalty = (0.0f * (-1.0f / episodeLen));
    // reward for completing task successfully
    inline constexpr float goalReward = 1.0f;

    // Reward for macguffin having velocity
    inline constexpr float macguffinStationaryPenalty = (1.0f * (-1.0f / episodeLen));
    inline constexpr float macguffinVelocityThreshold = .25f; // agent move speed is 100

    // How many discrete options for actions
    inline constexpr madrona::CountT numMoveAmountBuckets = 4;
    inline constexpr madrona::CountT numMoveAngleBuckets = 8;
    inline constexpr madrona::CountT numTurnBuckets = 5;

    // Number of lidar samples, arranged in circle around agent
    inline constexpr madrona::CountT numLidarSamples = 30;
    inline constexpr float lidarRange = 10.0f;

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
    inline constexpr float wallMeshHeight = 2.5f; // length and width are 1
    inline constexpr float wallMeshX = 1.0f;
    inline constexpr float wallMeshY = 1.0f;
  }
}
