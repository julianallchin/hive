#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/rand.hpp>
#include <madrona/physics.hpp>
#include <madrona/render/ecs.hpp>

#include "consts.hpp" // Assuming this will be updated with new constants

namespace madEscape
{

    // Forward declaration
    class Engine;

    // Madrona types used
    using madrona::CountT;
    using madrona::Entity;
    using madrona::RandKey;
    using madrona::base::ObjectID;
    using madrona::base::Position;
    using madrona::base::Rotation;
    using madrona::base::Scale;
    using madrona::phys::ExternalForce;
    using madrona::phys::ExternalTorque;
    using madrona::phys::ResponseType;
    using madrona::phys::RigidBody; // Bundle component
    using madrona::phys::Velocity;

    // WorldReset is a per-world singleton component that causes the current
    // episode to be terminated and the world regenerated.
    struct WorldReset
    {
        int32_t reset; // 1 to reset, 0 otherwise
    };

    // Polar coordinates for observations (distance and angle)
    struct PolarObservation
    {
        float r;     // distance
        float theta; // angle
    };

    // Defines the physical actions an ant can take.
    // This is an input to the simulation from the policy.
    struct AntAction
    {
        int32_t move_amount_idx; // Discrete bucket index for movement force
        int32_t move_angle_idx;  // Discrete bucket index for movement direction (egocentric)
        int32_t rotate_idx;      // Discrete bucket index for rotation torque
        int32_t grab_action;     // 0 = no-op, 1 = attempt to grab / release
    };

    // Observation data for a single ant, excluding Lidar.
    // This is an output from the simulation to the policy.
    struct AntObservationComponent
    {
        // Self state
        float global_x;          // Ant's global X position (normalized)
        float global_y;          // Ant's global Y position (normalized)
        float orientation_theta; // Ant's orientation angle in 2D plane (normalized -pi to pi)
        float is_grabbing;       // 1.0 if grabbing, 0.0 otherwise

        // Task state
        float polar_to_macguffin_r;     // Distance to macguffin
        float polar_to_macguffin_theta; // Angle to macguffin (egocentric)
        float polar_to_goal_r;          // Distance to goal
        float polar_to_goal_theta;      // Angle to goal (egocentric)
    };

    // Lidar sample structure (depth and type of hit entity)
    struct LidarSample
    {
        float depth;        // Distance to hit, normalized
        float encoded_type; // Encoded type of the entity hit
    };

    // Lidar sensor data for an ant.
    // This is an output from the simulation to the policy.
    struct Lidar
    {
        LidarSample samples[consts::numLidarSamples];
    };

    // Tracks if an ant is currently grabbing another entity
    struct GrabState
    {
        Entity constraint_entity; // Entity::none() if not grabbing
    };

    // Enum for various entity types in the simulation
    enum class EntityType : uint32_t
    {
        None,
        Ant,
        Macguffin,
        Goal,          // Non-physical target
        Wall,          // Static obstacle
        MovableObject, // Dynamic obstacle
        NumTypes,
    };

    // Singleton component for the hive's collective reward.
    struct HiveReward
    {
        float v;
    };

    // Singleton component indicating if the hive's episode is done.
    struct HiveDone
    {
        int32_t v; // 1 if done, 0 otherwise
    };

    // Singleton component tracking steps remaining in the global episode.
    struct StepsRemaining
    {
        uint32_t t;
    };
    
    // Singleton component for tracking the number of active ants per world
    struct AntCount
    {
        uint32_t count;
    };

    // Singleton component storing the state of the current level.
    struct LevelState
    {
        Entity macguffin;
        Entity goal;

        CountT num_current_movable_objects;
        Entity movable_objects[consts::maxMovableObjects];

        CountT num_current_walls;
        Entity walls[consts::maxWalls];
        // In future, might include curriculum learning parameters here
    };

    /* ECS Archetypes */

    // Archetype for Ants
    struct Ant : public madrona::Archetype<
                     // Physics components
                     RigidBody, // Includes Position, Rotation, Scale, ObjectID, Velocity, etc.
                     EntityType,

                     // Ant-specific state
                     GrabState,
                     Lidar,                   // Output to policy
                     AntObservationComponent, // Output to policy
                     
                     AntAction,               // Input from policy

                     // Rendering
                     madrona::render::Renderable
                     // madrona::render::RenderCamera, // Optional: if wanting per-ant camera views
                     >
    {
    };

    // Archetype for the Macguffin (the object to be moved)
    struct Macguffin : public madrona::Archetype<
                           RigidBody,
                           EntityType,
                           madrona::render::Renderable>
    {
    };

    // Archetype for the Goal (target location, non-physical)
    struct Goal : public madrona::Archetype<
                      Position, // Essential for location
                      ObjectID, // For rendering if visualized
                      Scale,    // For rendering if visualized
                      Rotation, // For rendering if visualized (though likely fixed for a 2D target marker)
                      EntityType,
                      madrona::render::Renderable>
    {
    };

    // Archetype for Walls (static obstacles)
    struct Wall : public madrona::Archetype<
                      RigidBody, // Will be set to ResponseType::Static
                      EntityType,
                      madrona::render::Renderable>
    {
    };

    // Archetype for Movable Objects (dynamic obstacles)
    struct MovableObject : public madrona::Archetype<
                               RigidBody,
                               EntityType,
                               madrona::render::Renderable>
    {
    };

    // Static asserts for component sizes if they are to be exported directly as fixed-size tensors
    // Note: Madrona handles padding, but good for sanity checking total data expected by Python.

    // Size of AntObservationComponent (8 floats)
    static_assert(sizeof(AntObservationComponent) == sizeof(float) * 8);

    // Size of Lidar data (numLidarSamples * 2 floats per sample)
    static_assert(sizeof(Lidar) == sizeof(LidarSample) * consts::numLidarSamples);
    static_assert(sizeof(LidarSample) == sizeof(float) * 2);

    // Size of AntAction (4 int32_t values)
    static_assert(sizeof(AntAction) == sizeof(int32_t) * 4);

} // namespace madEscape