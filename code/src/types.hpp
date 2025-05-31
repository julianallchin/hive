#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/rand.hpp>
#include <madrona/physics.hpp>
#include <madrona/render/ecs.hpp>

#include "consts.hpp"

namespace madEscape {

// Include several madrona types into the simulator namespace for convenience
using madrona::Entity;
using madrona::RandKey;
using madrona::CountT;
using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::base::ObjectID;
using madrona::phys::Velocity;
using madrona::phys::ResponseType;
using madrona::phys::ExternalForce;
using madrona::phys::ExternalTorque;
using madrona::phys::RigidBody;

// WorldReset is a per-world singleton component that causes the current
// episode to be terminated and the world regenerated
// (Singleton components like WorldReset can be accessed via Context::singleton
// (eg ctx.singleton<WorldReset>().reset = 1)
struct WorldReset {
    int32_t reset;
};

// Discrete action component. Ranges are defined by consts::numMoveBuckets (5),
// repeated here for clarity
struct Action {
    int32_t moveAmount; // [0, 3]
    int32_t moveAngle; // [0, 7]
    int32_t rotate; // [-2, 2]
    int32_t grab; // 0 = do nothing, 1 = grab / release
};

// Per-agent reward
// Exported as an [N * A, 1] float tensor to training code
struct Reward {
    float v;
};

// additional information needed to calculate reward
struct RewardHelper {
    float starting_dist;
    float prev_dist;
};

// Per-agent component that indicates that the agent's episode is finished
// This is exported per-agent for simplicity in the training code
struct Done {
    // Currently bool components are not supported due to
    // padding issues, so Done is an int32_t
    int32_t v;
};

// Observation state for the current agent.
// Positions are rescaled to the bounds of the play area to assist training.
struct SelfObservation {
    float globalX;
    float globalY;
    float globalZ;
    float theta;
    float isGrabbing;
    float polarToMacguffinR;     // Distance to macguffin
    float polarToMacguffinTheta; // Angle to macguffin (egocentric)
    float polarToGoalR;          // Distance to goal
    float polarToGoalTheta;      // Angle to goal (egocentric)
};

// The state of the world is passed to each agent in terms of egocentric
// polar coordinates. theta is degrees off agent forward.
struct PolarObservation {
    float r;
    float theta;
};

struct LidarSample {
    float depth;
    float encodedType;
};

// Linear depth values and entity type in a circle around the agent
struct Lidar {
    LidarSample samples[consts::numLidarSamples];
};

// Number of steps remaining in the episode. Allows non-recurrent policies
// to track the progression of time.
struct StepsRemaining {
    uint32_t t;
};

// Tracks if an agent is currently grabbing another entity
struct GrabState {
    Entity constraintEntity;
};

// Added state for the MacGuffin. Intentionally empty at the moment, as no additional state is needed.
// In the future, this could for example be replaced with actions, to allow the MacGuffin to become alive.
struct MacGuffinState{};

// whether a given Agent is alive
struct Active{
    int32_t v;
};

// This enum is used to track the type of each entity for the purposes of
// classifying the objects hit by each lidar sample.
enum class EntityType : uint32_t {
    None,
    Cube,
    Wall,
    Agent,
    MacGuffin,
    Goal,
    NumTypes,
};

struct EpisodeTracker : public madrona::Archetype <
    Reward,
    RewardHelper,
    Done,
    StepsRemaining
> {};

/* ECS Archetypes for the game */

// There are 2 Agents in the environment trying to get to the destination
struct Agent : public madrona::Archetype<
    // RigidBody is a "bundle" component defined in physics.hpp in Madrona.
    // This includes a number of components into the archetype, including
    // Position, Rotation, Scale, Velocity, and a number of other components
    // used internally by the physics.
    RigidBody,

    // Internal logic state.
    GrabState,
    EntityType,

    // Input
    Action,

    // Observations
    SelfObservation,
    Lidar,

    Active,

    // Visualization: In addition to the fly camera, src/viewer.cpp can
    // view the scene from the perspective of entities with this component
    madrona::render::RenderCamera,
    // All entities with the Renderable component will be drawn by the
    // viewer and batch renderer
    madrona::render::Renderable
> {};

// The MacGuffin object that must be moved to the goal.
// Since MacGuffinState is currently empty, this could really just be a PhysicsEntity,
// but we create a new archetype to separate them and allow for added state in the future.
struct MacGuffin : public madrona:: Archetype<
    MacGuffinState,
    RigidBody,
    EntityType,
    madrona::render::Renderable
> {};

// Archetype for the goal objects that the macguffin is moved onto
// Goal does't have collision but are rendered
struct Goal : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    ObjectID,
    EntityType,
    madrona::render::Renderable
> {};

// Generic archetype for entities that need physics but don't have custom
// logic associated with them.
struct PhysicsEntity : public madrona::Archetype<
    RigidBody,
    EntityType,
    madrona::render::Renderable
> {};

}
