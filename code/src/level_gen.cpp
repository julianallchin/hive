#include "level_gen.hpp"

namespace madEscape {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

static inline float randInRangeCentered(Engine &ctx, float range)
{
    return ctx.data().rng.sampleUniform() * range - range / 2.f;
}

static inline float randBetween(Engine &ctx, float min, float max)
{
    return ctx.data().rng.sampleUniform() * (max - min) + min;
}

// Initialize the basic components needed for physics rigid body entities
static inline void setupRigidBodyEntity(
    Engine &ctx,
    Entity e,
    Vector3 pos,
    Quat rot,
    SimObject sim_obj,
    EntityType entity_type,
    ResponseType response_type = ResponseType::Dynamic,
    Diag3x3 scale = {1, 1, 1})
{
    ObjectID obj_id { (int32_t)sim_obj };

    ctx.get<Position>(e) = pos;
    ctx.get<Rotation>(e) = rot;
    ctx.get<Scale>(e) = scale;
    ctx.get<ObjectID>(e) = obj_id;
    ctx.get<Velocity>(e) = {
        Vector3::zero(),
        Vector3::zero(),
    };
    ctx.get<ResponseType>(e) = response_type;
    ctx.get<ExternalForce>(e) = Vector3::zero();
    ctx.get<ExternalTorque>(e) = Vector3::zero();
    ctx.get<EntityType>(e) = entity_type;
}

// Register the entity with the broadphase system
// This is needed for every entity with all the physics components.
// Not registering an entity will cause a crash because the broadphase
// systems will still execute over entities with the physics components.
static void registerRigidBodyEntity(
    Engine &ctx,
    Entity e,
    SimObject sim_obj)
{
    ObjectID obj_id { (int32_t)sim_obj };
    ctx.get<broadphase::LeafID>(e) =
        PhysicsSystem::registerEntity(ctx, e, obj_id);
}

// Creates floor, outer walls, and agent entities.
// All these entities persist across all episodes.
void createPersistentEntities(Engine &ctx)
{
    // Create the floor entity, just a simple static plane.
    ctx.data().floorPlane = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().floorPlane,
        Vector3 { 0, 0, 0 },
        Quat { 1, 0, 0, 0 },
        SimObject::Plane,
        EntityType::None, // Floor plane type should never be queried
        ResponseType::Static);

    // Create the outer wall entities
    // Behind
    ctx.data().borders[0] = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().borders[0],
        Vector3 {
            0,
            -consts::wallWidth / 2.f,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::worldWidth + consts::wallWidth * 2,
            consts::wallWidth,
            2.f,
        });

    // Right
    ctx.data().borders[1] = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().borders[1],
        Vector3 {
            consts::worldWidth / 2.f + consts::wallWidth / 2.f,
            consts::worldLength / 2.f,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::wallWidth,
            consts::worldLength,
            2.f,
        });

    // Left
    ctx.data().borders[2] = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().borders[2],
        Vector3 {
            -consts::worldWidth / 2.f - consts::wallWidth / 2.f,
            consts::worldLength / 2.f,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::wallWidth,
            consts::worldLength,
            2.f,
        });

    // Top
    ctx.data().borders[3] = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().borders[3],
        Vector3 {
            0,
            consts::worldLength - consts::wallWidth / 2.0f,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::worldWidth,
            consts::wallWidth,
            2.f,
        });

    // initialized on reset
    ctx.data().episodeTracker = ctx.makeEntity<EpisodeTracker>();

    // MacGuffin
    ctx.data().macguffin = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().macguffin,
        Vector3 {0, 0, 0},  // position is reset on level start
        Quat {1, 0, 0, 0 },
        SimObject::MacGuffin,
        EntityType::MacGuffin,
        ResponseType::Dynamic,
        Diag3x3 {
            consts::macguffinSize,
            consts::macguffinSize,
            consts::macguffinSize
        });

    // Goal
    Entity goal = ctx.data().goal = ctx.makeRenderableEntity<Goal>();
    ctx.get<Rotation>(goal) = Quat {1, 0, 0, 0};
    ctx.get<Scale>(goal) = Diag3x3 { consts::goalSize, consts::goalSize, 0.1f };
    ctx.get<ObjectID>(goal) = ObjectID { (int32_t)SimObject::Goal };
    ctx.get<EntityType>(goal) = EntityType::Goal;
    // position to be initialized on level reset

    // Create agent entities. Note that this leaves a lot of components
    // uninitialized, these will be set during world generation, which is
    // called for every episode.
    for (CountT i = 0; i < consts::numAgents; ++i) {
        Entity agent = ctx.data().agents[i] =
            ctx.makeRenderableEntity<Agent>();

        // Create a render view for the agent
        if (ctx.data().enableRender) {
            render::RenderingSystem::attachEntityToView(ctx,
                    agent,
                    100.f, 0.001f,
                    1.5f * math::up);
        }

        ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
        ctx.get<ObjectID>(agent) = ObjectID { (int32_t)SimObject::Agent };
        ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
        ctx.get<GrabState>(agent).constraintEntity = Entity::none();
        ctx.get<EntityType>(agent) = EntityType::Agent;
    }
}

// Although agents and walls persist between episodes, we still need to
// re-register them with the broadphase system and, in the case of the agents,
// reset their positions.
static void resetPersistentEntities(Engine &ctx)
{
    registerRigidBodyEntity(ctx, ctx.data().floorPlane, SimObject::Plane);

    for (CountT i = 0; i < 4; i++) {
        Entity wall_entity = ctx.data().borders[i];
        registerRigidBodyEntity(ctx, wall_entity, SimObject::Wall);
    }
    // MacGuffin
    Entity macguffin = ctx.data().macguffin;
    Vector3 macguffin_pos{0.0, 0.0, 10.0};
    registerRigidBodyEntity(ctx, macguffin, SimObject::MacGuffin);
    ctx.get<Position>(macguffin) = macguffin_pos;
    ctx.get<Rotation>(macguffin) = Quat {1, 0, 0, 0};
    ctx.get<Velocity>(macguffin) = {
        Vector3::zero(),
        Vector3::zero(),
    };
    ctx.get<ExternalForce>(macguffin) = Vector3::zero();
    ctx.get<ExternalTorque>(macguffin) = Vector3::zero();

    // Goal
    Entity goal = ctx.data().goal;
    Vector3 goal_pos{ 15.0f, 15.0f, 0.0f};
    ctx.get<Position>(goal) = goal_pos;

    // Episode Tracker
    Entity episodeTracker = ctx.data().episodeTracker;
    ctx.get<StepsRemaining>(episodeTracker).t = consts::episodeLen;
    ctx.get<RewardHelper>(episodeTracker).starting_dist = -1.0f; // initialized in rewardsystem
    ctx.get<RewardHelper>(episodeTracker).prev_dist = -1.0f; // initialized in rewardsystem


    // Agents
    for (CountT i = 0; i < consts::numAgents; i++) {
        Entity agent_entity = ctx.data().agents[i];
        registerRigidBodyEntity(ctx, agent_entity, SimObject::Agent);

        // Place the agents near the starting wall
        Vector3 pos {
            randInRangeCentered(ctx, 
                consts::worldWidth / 2.f - 2.5f * consts::agentRadius),
            randBetween(ctx, consts::agentRadius * 1.1f,  2.f),
            0.f,
        };

        if (i % 2 == 0) {
            pos.x += consts::worldWidth / 4.f;
        } else {
            pos.x -= consts::worldWidth / 4.f;
        }

        ctx.get<Position>(agent_entity) = pos;
        ctx.get<Rotation>(agent_entity) = Quat::angleAxis(
            randInRangeCentered(ctx, math::pi / 4.f),
            math::up);

        auto &grab_state = ctx.get<GrabState>(agent_entity);
        if (grab_state.constraintEntity != Entity::none()) {
            ctx.destroyEntity(grab_state.constraintEntity);
            grab_state.constraintEntity = Entity::none();
        }

        ctx.get<Velocity>(agent_entity) = {
            Vector3::zero(),
            Vector3::zero(),
        };
        ctx.get<ExternalForce>(agent_entity) = Vector3::zero();
        ctx.get<ExternalTorque>(agent_entity) = Vector3::zero();
        ctx.get<Action>(agent_entity) = Action {
            .moveAmount = 0,
            .moveAngle = 0,
            .rotate = consts::numTurnBuckets / 2,
            .grab = 0,
        };
    }
}

static Entity makeCube(Engine &ctx,
                       float cube_x,
                       float cube_y,
                       float scale = 1.f)
{
    Entity cube = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        cube,
        Vector3 {
            cube_x,
            cube_y,
            1.f * scale,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Cube,
        EntityType::Cube,
        ResponseType::Dynamic,
        Diag3x3 {
            scale,
            scale,
            scale,
        });
    registerRigidBodyEntity(ctx, cube, SimObject::Cube);

    return cube;
}

enum class WallDirection : bool {
    Horizontal,
    Vertical
};

static Entity makeBarrier(Engine &ctx,
                       float x,
                       float y,
                       float length,
                       WallDirection isHorizontal)
{
    Entity barrier = ctx.makeRenderableEntity<PhysicsEntity>();
    float x_size;
    float y_size;
    if (isHorizontal == WallDirection::Horizontal) {
        x_size = length;
        y_size = consts::barrierWidth;
    }
    else {
        x_size = consts::barrierWidth;
        y_size = length;
    }
    setupRigidBodyEntity(
        ctx,
        barrier,
        Vector3 {
            x,
            y,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            x_size,
            y_size,
            consts::barrierHeight,
        });
    registerRigidBodyEntity(ctx, barrier, SimObject::Wall);
    return barrier;
}

static void generateLevel(Engine &ctx)
{
    // some cubes
    CountT numCubes = consts::maxCubes;
    for (CountT i = 0; i < numCubes; i++) {
        float x = randBetween(ctx, -10.0f, 10.0f);
        float y = 10.0f * i + 5.0f;
        float scale = 1.0f;
        Entity cube = makeCube(ctx, x, y, scale);
        ctx.data().cubes[i] = cube;
    }
    for (CountT i = numCubes; i < consts::maxCubes; i++) {
        ctx.data().cubes[i] = Entity::none();
    }


    // some barriers
    CountT numBarriers = consts::maxBarriers;
    for (CountT i = 0; i < numBarriers; i++) {
        float x = randBetween(ctx, -10.0f, 10.0f);
        float y = -10.0f * i - 5.0f;
        float length = 10.0f;
        Entity barrier = makeBarrier(ctx, x, y, length, WallDirection::Horizontal);
        ctx.data().barriers[i] = barrier;
    }
    for (CountT i = numBarriers; i < consts::maxBarriers; i++) {
        ctx.data().barriers[i] = Entity::none();
    }
}

// Randomly generate a new world for a training episode
void generateWorld(Engine &ctx)
{
    resetPersistentEntities(ctx);
    generateLevel(ctx);
}

}
