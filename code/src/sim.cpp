#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"
#include "level_gen.hpp"

#include <algorithm>

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace RenderingSystem = madrona::render::RenderingSystem;

namespace madEscape
{

    // Register all the ECS components and archetypes that will be
    // used in the simulation
    void Sim::registerTypes(ECSRegistry &registry, const Config &cfg)
    {
        base::registerTypes(registry);
        phys::PhysicsSystem::registerTypes(registry);

        RenderingSystem::registerTypes(registry, cfg.renderBridge);

        // Register individual components
        registry.registerComponent<AntAction>();
        registry.registerComponent<AntObservationComponent>();
        registry.registerComponent<HiveReward>();
        registry.registerComponent<HiveDone>();
        registry.registerComponent<GrabState>();
        registry.registerComponent<Lidar>();
        registry.registerComponent<StepsRemaining>();
        registry.registerComponent<EntityType>();

        // Register singleton components
        registry.registerSingleton<WorldReset>();
        registry.registerSingleton<LevelState>();
        registry.registerSingleton<HiveReward>();
        registry.registerSingleton<HiveDone>();
        registry.registerSingleton<StepsRemaining>();

        // Register archetypes
        registry.registerArchetype<Ant>();
        registry.registerArchetype<Macguffin>();
        registry.registerArchetype<Goal>();
        registry.registerArchetype<Wall>();
        registry.registerArchetype<MovableObject>();

        // Export interfaces for Python training code
        registry.exportSingleton<WorldReset>(
            (uint32_t)ExportID::Reset);
        registry.exportColumn<Ant, AntAction>(
            (uint32_t)ExportID::Action);
        registry.exportSingleton<HiveReward>(
            (uint32_t)ExportID::Reward);
        registry.exportSingleton<HiveDone>(
            (uint32_t)ExportID::Done);
        registry.exportColumn<Ant, AntObservationComponent>(
            (uint32_t)ExportID::SelfObservation);
        registry.exportColumn<Ant, Lidar>(
            (uint32_t)ExportID::Lidar);
        registry.exportSingleton<StepsRemaining>(
            (uint32_t)ExportID::StepsRemaining);
        
        // Note: We removed PartnerObservations, RoomEntityObservations, and DoorObservation exports
        // as they are not needed in the hive simulation
    }

    static inline void cleanupWorld(Engine &ctx)
    {
        // Destroy current level entities
        LevelState &level = ctx.singleton<LevelState>();

        // Clean up movable objects
        for (CountT i = 0; i < level.num_current_movable_objects; i++)
        {
            if (level.movable_objects[i] != Entity::none())
            {
                ctx.destroyRenderableEntity(level.movable_objects[i]);
            }
        }

        // Clean up walls (except border walls which are persistent)
        for (CountT i = 0; i < level.num_current_walls; i++)
        {
            if (level.walls[i] != Entity::none())
            {
                ctx.destroyRenderableEntity(level.walls[i]);
            }
        }

        // Clean up macguffin and goal
        if (level.macguffin != Entity::none())
        {
            ctx.destroyRenderableEntity(level.macguffin);
        }

        if (level.goal != Entity::none())
        {
            ctx.destroyRenderableEntity(level.goal);
        }
    }

    static inline void initWorld(Engine &ctx)
{
    phys::PhysicsSystem::reset(ctx);

    // Assign a new episode ID and update RNG
    ctx.data().rng = RNG(rand::split_i(ctx.data().initRandKey,
                                       ctx.data().curWorldEpisode++, (uint32_t)ctx.worldID().idx));

    // Initialize level state and singletons
    LevelState &level = ctx.singleton<LevelState>();
    level.macguffin = Entity::none();
    level.goal = Entity::none();
    level.num_current_movable_objects = 0;
    level.num_current_walls = 0;

    // Initialize hive reward and done state
    ctx.singleton<HiveReward>().v = 0.0f;
    ctx.singleton<HiveDone>().v = 0;

    // Set steps remaining to max episode length
    ctx.singleton<StepsRemaining>().t = consts::episodeLen;

    // Determine number of entities based on curriculum difficulty
    int currentDifficulty = ctx.data().curriculumDifficulty;
    
    // Calculate number of ants based on difficulty
    ctx.data().currentNumAnts = std::min(
        consts::defaultAnts + currentDifficulty * 5,
        (int)consts::maxAnts);
    
    // Calculate number of movable objects based on difficulty
    ctx.data().currentNumMovableObjects = std::min(
        consts::defaultMovableObjects + currentDifficulty / 2,
        (int)consts::maxMovableObjects);

    // Calculate number of interior walls based on difficulty
    ctx.data().currentNumInteriorWalls = std::min(
        consts::defaultWalls + currentDifficulty / 3,
        (int)consts::maxInteriorWalls);

    // Defined in src/level_gen.hpp / src/level_gen.cpp
    // This will generate the world using the parameters we've set
    generateWorld(ctx);
    
    // Every 10 episodes, increase the curriculum difficulty
    if (ctx.data().curWorldEpisode % 10 == 0 && ctx.data().curWorldEpisode > 0) {
        ctx.data().curriculumDifficulty++;
    }
    }

    // This system runs each frame and checks if the current episode is complete
    // or if code external to the application has forced a reset by writing to the
    // WorldReset singleton.
    //
    // If a reset is needed, cleanup the existing world and generate a new one.
    inline void resetSystem(Engine &ctx, WorldReset &reset)
    {
        int32_t should_reset = reset.reset;
        if (ctx.data().autoReset)
        {
            // Check if the hive's episode is done
            HiveDone &done = ctx.singleton<HiveDone>();
            if (done.v == 1)
            {
                should_reset = 1;
            }
        }
    }

    if (should_reset != 0)
    {
        reset.reset = 0;

        cleanupWorld(ctx);
        initWorld(ctx);
    }
}

// Translates discrete actions from the AntAction component to forces
// used by the physics simulation.
inline void antMovementSystem(Engine &,
                              AntAction &action,
                              Rotation &rot,
                              ExternalForce &external_force,
                              ExternalTorque &external_torque)
{
    constexpr float move_max = 1000;
    constexpr float turn_max = 320;

    Quat cur_rot = rot;

    float move_amount = action.move_amount_idx *
                        (move_max / (consts::numMoveAmountBuckets - 1));

    constexpr float move_angle_per_bucket =
        2.f * math::pi / float(consts::numMoveAngleBuckets);

    float move_angle = float(action.move_angle_idx) * move_angle_per_bucket;

    float f_x = move_amount * sinf(move_angle);
    float f_y = move_amount * cosf(move_angle);

    constexpr float turn_delta_per_bucket =
        turn_max / (consts::numTurnBuckets / 2);
    float t_z =
        turn_delta_per_bucket * (action.rotate_idx - consts::numTurnBuckets / 2);

    external_force = cur_rot.rotateVec({f_x, f_y, 0});
    external_torque = Vector3{0, 0, t_z};
}

// Implements the grab action by casting a short ray in front of the ant
// and creating a joint constraint if a grabbable entity is hit.
inline void antGrabSystem(Engine &ctx,
                          Entity e,
                          Position pos,
                          Rotation rot,
                          AntAction action,
                          GrabState &grab)
{
    if (action.grab_action == 0)
    {
        return;
    }

    // if a grab is currently in progress, triggering the grab action
    // just releases the object
    if (grab.constraintEntity != Entity::none())
    {
        ctx.destroyEntity(grab.constraintEntity);
        grab.constraintEntity = Entity::none();

        return;
    }

    // Get the per-world BVH singleton component
    auto &bvh = ctx.singleton<broadphase::BVH>();
    float hit_t;
    Vector3 hit_normal;

    Vector3 ray_o = pos + 0.5f * math::up;
    Vector3 ray_d = rot.rotateVec(math::fwd);

    // Short ray cast for grabbing
    Entity grab_entity =
        bvh.traceRay(ray_o, ray_d, &hit_t, &hit_normal, 2.0f);

    if (grab_entity == Entity::none())
    {
        return;
    }

    auto response_type = ctx.get<ResponseType>(grab_entity);
    if (response_type != ResponseType::Dynamic)
    {
        return;
    }

    auto entity_type = ctx.get<EntityType>(grab_entity);
    // Ants can't grab other ants
    if (entity_type == EntityType::Ant)
    {
        return;
    }

    Vector3 other_pos = ctx.get<Position>(grab_entity);
    Quat other_rot = ctx.get<Rotation>(grab_entity);

    // Use a shorter reach for ants
    Vector3 r1 = 0.25f * math::fwd + 0.25f * math::up;

    Vector3 hit_pos = ray_o + ray_d * hit_t;
    Vector3 r2 =
        other_rot.inv().rotateVec(hit_pos - other_pos);

    Quat attach1 = {1, 0, 0, 0};
    Quat attach2 = (other_rot.inv() * rot).normalize();

    float separation = hit_t - 0.25f;

    grab.constraintEntity = PhysicsSystem::makeFixedJoint(ctx,
                                                          e, grab_entity, attach1, attach2, r1, r2, separation);
}

// Make the ants easier to control by zeroing out their velocity
// after each step.
inline void antZeroVelSystem(Engine &,
                             Velocity &vel,
                             AntAction &)
{
    vel.linear.x = 0;
    vel.linear.y = 0;
    vel.linear.z = fminf(vel.linear.z, 0);

    vel.angular = Vector3::zero();
}

static inline float distObs(float v)
{
    return v / consts::worldLength;
}

static inline float globalPosObs(float v)
{
    return v / consts::worldLength;
}

// Computes the collective reward for the hive based on macguffin movement toward goal
// and goal achievement.
inline void hiveRewardSystem(Engine &ctx)
{
    LevelState &level = ctx.singleton<LevelState>();
    HiveReward &reward = ctx.singleton<HiveReward>();
    HiveDone &done = ctx.singleton<HiveDone>();
    StepsRemaining &steps = ctx.singleton<StepsRemaining>();

    // If done, don't update reward
    if (done.v == 1)
    {
        return;
    }

    // Get positions of macguffin and goal
    Vector3 macguffin_pos = ctx.get<Position>(level.macguffin);
    Vector3 goal_pos = ctx.get<Position>(level.goal);

    // Calculate 2D distance (ignore Z axis)
    Vector2 macguffin_pos_2d(macguffin_pos.x, macguffin_pos.y);
    Vector2 goal_pos_2d(goal_pos.x, goal_pos.y);
    float dist = (goal_pos_2d - macguffin_pos_2d).length();

    // Static variable to store previous distance
    static thread_local float prev_dist = -1.0f;

    // First time initialization
    if (prev_dist < 0)
    {
        prev_dist = dist;
        reward.v = 0.0f;
        return;
    }

    // Step reward based on distance reduction
    float step_reward = consts::distanceRewardScale * (prev_dist - dist);

    // Goal reward if close enough
    float goal_reward = 0.0f;
    if (dist <= consts::goalDistanceThreshold)
    {
        goal_reward = consts::goalReward;
        done.v = 1; // Episode complete on goal achievement
    }

    // Existential penalty per timestep
    float exist_penalty = consts::existentialPenalty;

    // Total reward for this step
    reward.v = step_reward + goal_reward + exist_penalty;

    // If steps remaining is zero, mark as done
    if (--steps.t <= 0)
    {
        done.v = 1;
    }

    // Store current distance for next step
    prev_dist = dist;
}

static inline float angleObs(float v)
{
    return v / math::pi;
}

// Translate xy delta to polar observations for learning.
static inline PolarObservation xyToPolar(Vector3 v)
{
    Vector2 xy{v.x, v.y};

    float r = xy.length();

    // Note that this is angle off y-forward
    float theta = atan2f(xy.x, xy.y);

    return PolarObservation{
        .r = distObs(r),
        .theta = angleObs(theta),
    };
}

static inline float encodeType(EntityType type)
{
    return (float)type / (float)EntityType::NumTypes;
}

static inline float computeZAngle(Quat q)
{
    float siny_cosp = 2.f * (q.w * q.z + q.x * q.y);
    float cosy_cosp = 1.f - 2.f * (q.y * q.y + q.z * q.z);
    return atan2f(siny_cosp, cosy_cosp);
}

// This system packages ant observations for the ant policy inputs.
// It collects self-state and relative polar coordinates to important objects.
inline void collectAntObservationsSystem(Engine &ctx,
                                         Position pos,
                                         Rotation rot,
                                         const GrabState &grab,
                                         AntObservationComponent &ant_obs)
{
    // Get level state to access macguffin and goal entities
    const LevelState &level = ctx.singleton<LevelState>();

    // Self state observations
    ant_obs.global_x = globalPosObs(pos.x);
    ant_obs.global_y = globalPosObs(pos.y);
    ant_obs.orientation_theta = angleObs(computeZAngle(rot));
    ant_obs.is_grabbing = grab.constraintEntity != Entity::none() ? 1.0f : 0.0f;

    // Get positions of important objects
    Vector3 macguffin_pos = ctx.get<Position>(level.macguffin);
    Vector3 goal_pos = ctx.get<Position>(level.goal);

    // Calculate vectors to important objects in world space
    Vector3 to_macguffin = macguffin_pos - pos;
    Vector3 to_goal = goal_pos - pos;

    // Convert to ant's local coordinate system
    Quat to_view = rot.inv();
    Vector3 local_to_macguffin = to_view.rotateVec(to_macguffin);
    Vector3 local_to_goal = to_view.rotateVec(to_goal);

    // Convert to polar coordinates for observations
    PolarObservation polar_to_macguffin = xyToPolar(local_to_macguffin);
    PolarObservation polar_to_goal = xyToPolar(local_to_goal);

    // Store polar observations
    ant_obs.polar_to_macguffin_r = polar_to_macguffin.r;
    ant_obs.polar_to_macguffin_theta = polar_to_macguffin.theta;
    ant_obs.polar_to_goal_r = polar_to_goal.r;
    ant_obs.polar_to_goal_theta = polar_to_goal.theta;
}

// Launches consts::numLidarSamples per agent.
// This system is specially optimized in the GPU version:
// a warp of threads is dispatched for each invocation of the system
// and each thread in the warp traces one lidar ray for the agent.
inline void lidarSystem(Engine &ctx,
                        Entity e,
                        Lidar &lidar)
{
    Vector3 pos = ctx.get<Position>(e);
    Quat rot = ctx.get<Rotation>(e);
    auto &bvh = ctx.singleton<broadphase::BVH>();

    Vector3 agent_fwd = rot.rotateVec(math::fwd);
    Vector3 right = rot.rotateVec(math::right);

    auto traceRay = [&](int32_t idx)
    {
        float theta = 2.f * math::pi * (float(idx) / float(consts::numLidarSamples)) + math::pi / 2.f;
        float x = cosf(theta);
        float y = sinf(theta);

        Vector3 ray_dir = (x * right + y * agent_fwd).normalize();

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity =
            bvh.traceRay(pos + 0.5f * math::up, ray_dir, &hit_t,
                         &hit_normal, 200.f);

        if (hit_entity == Entity::none())
        {
            lidar.samples[idx] = {
                .depth = 0.f,
                .encodedType = encodeType(EntityType::None),
            };
        }
        else
        {
            EntityType entity_type = ctx.get<EntityType>(hit_entity);

            lidar.samples[idx] = {
                .depth = distObs(hit_t),
                .encodedType = encodeType(entity_type),
            };
        }
    };

    // MADRONA_GPU_MODE guards GPU specific logic
#ifdef MADRONA_GPU_MODE
    // Can use standard cuda variables like threadIdx for
    // warp level programming
    int32_t idx = threadIdx.x % 32;

    if (idx < consts::numLidarSamples)
    {
        traceRay(idx);
    }
#else
    for (CountT i = 0; i < consts::numLidarSamples; i++)
    {
        traceRay(i);
    }
#endif
}

// Computes reward for each agent and keeps track of the max distance achieved
// so far through the challenge. Continuous reward is provided for any new
// distance achieved.
inline void rewardSystem(Engine &,
                         Position pos,
                         Progress &progress,
                         Reward &out_reward)
{
    // Just in case agents do something crazy, clamp total reward
    float reward_pos = fminf(pos.y, consts::worldLength * 2);

    float old_max_y = progress.maxY;

    float new_progress = reward_pos - old_max_y;

    float reward;
    if (new_progress > 0)
    {
        reward = new_progress * consts::rewardPerDist;
        progress.maxY = reward_pos;
    }
    else
    {
        reward = consts::slackReward;
    }

    out_reward.v = reward;
}

// Each agent gets a small bonus to it's reward if the other agent has
// progressed a similar distance, to encourage them to cooperate.
// This system reads the values of the Progress component written by
// rewardSystem for other agents, so it must run after.
inline void bonusRewardSystem(Engine &ctx,
                              OtherAgents &others,
                              Progress &progress,
                              Reward &reward)
{
    bool partners_close = true;
    for (CountT i = 0; i < consts::numAgents - 1; i++)
    {
        Entity other = others.e[i];
        Progress other_progress = ctx.get<Progress>(other);

        if (fabsf(other_progress.maxY - progress.maxY) > 2.f)
        {
            partners_close = false;
        }
    }

    if (partners_close && reward.v > 0.f)
    {
        reward.v *= 1.25f;
    }
}

// Keep track of the number of steps remaining in the episode and
// notify training that an episode has completed by
// setting done = 1 on the final step of the episode
inline void stepTrackerSystem(Engine &,
                              StepsRemaining &steps_remaining,
                              Done &done)
{
    int32_t num_remaining = --steps_remaining.t;
    if (num_remaining == consts::episodeLen - 1)
    {
        done.v = 0;
    }
    else if (num_remaining == 0)
    {
        done.v = 1;
    }
}

// Helper function for sorting nodes in the taskgraph.
// Sorting is only supported / required on the GPU backend,
// since the CPU backend currently keeps separate tables for each world.
// This will likely change in the future with sorting required for both
// environments
#ifdef MADRONA_GPU_MODE
template <typename ArchetypeT>
TaskGraph::NodeID queueSortByWorld(TaskGraph::Builder &builder,
                                   Span<const TaskGraph::NodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(
            deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}
#endif

// Build the task graph
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &cfg)
{
    TaskGraphBuilder &builder = taskgraph_mgr.init(TaskGraphID::Step);

    // Turn policy actions into movement
    auto move_sys = builder.addToGraph<ParallelForNode<Engine,
                                                       antMovementSystem,
                                                       AntAction,
                                                       Rotation,
                                                       ExternalForce,
                                                       ExternalTorque>>({});

    // Build BVH for broadphase / raycasting
    auto broadphase_setup_sys = phys::PhysicsSystem::setupBroadphaseTasks(
        builder, {move_sys});

    // Grab action, post BVH build to allow raycasting
    auto grab_sys = builder.addToGraph<ParallelForNode<Engine,
                                                       antGrabSystem,
                                                       Entity,
                                                       Position,
                                                       Rotation,
                                                       AntAction,
                                                       GrabState>>({broadphase_setup_sys});

    // Physics collision detection and solver
    auto substep_sys = phys::PhysicsSystem::setupPhysicsStepTasks(builder,
                                                                  {grab_sys}, consts::numPhysicsSubsteps);

    // Improve controllability of ants by setting their velocity to 0
    // after physics is done.
    auto ant_zero_vel = builder.addToGraph<ParallelForNode<Engine,
                                                           antZeroVelSystem, Velocity, AntAction>>(
        {substep_sys});

    // Finalize physics subsystem work
    auto phys_done = phys::PhysicsSystem::setupCleanupTasks(
        builder, {ant_zero_vel});

    // Compute hive reward based on macguffin position relative to goal
    auto hive_reward_sys = builder.addToGraph<SingletonNode<Engine,
                                                            hiveRewardSystem>>({phys_done});

    // Conditionally reset the world if the episode is over
    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
                                                        resetSystem,
                                                        WorldReset>>({hive_reward_sys});

    auto clear_tmp = builder.addToGraph<ResetTmpAllocNode>({reset_sys});
    (void)clear_tmp;

#ifdef MADRONA_GPU_MODE
    // RecycleEntitiesNode is required on the GPU backend in order to reclaim
    // deleted entity IDs.
    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({reset_sys});
    (void)recycle_sys;
#endif

    // This second BVH build is a limitation of the current taskgraph API.
    // It's only necessary if the world was reset, but we don't have a way
    // to conditionally queue taskgraph nodes yet.
    auto post_reset_broadphase = phys::PhysicsSystem::setupBroadphaseTasks(
        builder, {reset_sys});

    // Collect ant observations for the next step
    auto collect_ant_obs = builder.addToGraph<ParallelForNode<Engine,
                                                              collectAntObservationsSystem,
                                                              Position,
                                                              Rotation,
                                                              GrabState,
                                                              AntObservationComponent>>({post_reset_broadphase});

    // The lidar system
#ifdef MADRONA_GPU_MODE
    // Note the use of CustomParallelForNode to create a taskgraph node
    // that launches a warp of threads (32) for each invocation (1).
    // The 32, 1 parameters could be changed to 32, 32 to create a system
    // that cooperatively processes 32 entities within a warp.
    auto lidar = builder.addToGraph<CustomParallelForNode<Engine,
                                                          lidarSystem, 32, 1,
#else
    auto lidar = builder.addToGraph<ParallelForNode<Engine,
                                                    lidarSystem,
#endif
                                                          Entity,
                                                          Lidar>>({post_reset_broadphase});

    if (cfg.renderBridge)
    {
        RenderingSystem::setupTasks(builder, {reset_sys});
    }

#ifdef MADRONA_GPU_MODE
    // Sort entities by world for improved performance
    auto sort_ants = queueSortByWorld<Ant>(
        builder, {lidar, collect_ant_obs});
    auto sort_macguffin = queueSortByWorld<Macguffin>(
        builder, {sort_ants});
    auto sort_movable = queueSortByWorld<MovableObject>(
        builder, {sort_macguffin});
    auto sort_walls = queueSortByWorld<Wall>(
        builder, {sort_movable});
    (void)sort_walls;
#else
    (void)lidar;
    (void)collect_ant_obs;
#endif
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
    // Currently the physics system needs an upper bound on the number of
    // entities that will be stored in the BVH. We plan to fix this in
    // a future release.
    constexpr CountT max_total_entities = consts::maxAnts +
                                          consts::maxMovableObjects + consts::maxInteriorWalls +
                                          7; // 4 border walls + floor + macguffin + goal

    // Initialize physics system with gravity that only applies to the XY plane (no Z)
    phys::PhysicsSystem::init(ctx, cfg.rigidBodyObjMgr,
                              consts::deltaT, consts::numPhysicsSubsteps, {0.f, 0.f, 0.f},
                              max_total_entities);

    initRandKey = cfg.initRandKey;
    autoReset = cfg.autoReset;
    enableRender = cfg.renderBridge != nullptr;
    
    // Initialize curriculum difficulty to start at 0 (easiest level)
    curriculumDifficulty = 0;
    
    // Initialize ant count based on config or default
    currentNumAnts = cfg.numAnts > 0 ? cfg.numAnts : consts::defaultAnts;
    currentNumMovableObjects = cfg.numMovableObjects;
    currentNumInteriorWalls = cfg.numWalls;

    if (enableRender)
    {
        RenderingSystem::init(ctx, cfg.renderBridge);
    }

    // Initialize episode counter
    curWorldEpisode = 0;
    
    // Initialize ant, object, and wall arrays to Entity::none()
    for (CountT i = 0; i < consts::maxAnts; i++) {
        ants[i] = Entity::none();
    }
    
    for (CountT i = 0; i < consts::maxMovableObjects; i++) {
        movableObjects[i] = Entity::none();
    }
    
    for (CountT i = 0; i < consts::maxInteriorWalls; i++) {
        interiorWalls[i] = Entity::none();
    }

    // Create the persistent entities (borders, floor, etc).
    createPersistentEntities(ctx);

    // Generate initial world state with macguffin, goal, etc.
    initWorld(ctx);
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);
}
