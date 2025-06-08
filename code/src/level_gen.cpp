#include "level_gen.hpp"

namespace madEscape
{

    using namespace madrona;
    using namespace madrona::math;
    using namespace madrona::phys;

    static inline float randBetween(Engine &ctx, float min, float max)
    {
        return ctx.data().rng.sampleUniform() * (max - min) + min;
    }

    // Returns a random integer between min and max (inclusive)
    static inline int randIntBetween(Engine &ctx, int min, int max)
    {
        float rand_float = ctx.data().rng.sampleUniform() * (max + 1 - min) + min;
        return static_cast<int>(rand_float);
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
        ObjectID obj_id{(int32_t)sim_obj};

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
        ObjectID obj_id{(int32_t)sim_obj};
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
            Vector3{0, 0, 0},
            Quat{1, 0, 0, 0},
            SimObject::Plane,
            EntityType::None, // Floor plane type should never be queried
            ResponseType::Static);

        // Create the outer wall entities
        // Bottom
        ctx.data().borders[0] = ctx.makeRenderableEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            ctx.data().borders[0],
            Vector3{
                0,
                -consts::worldLength / 2.f,
                0,
            },
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::Wall,
            ResponseType::Static,
            Diag3x3{
                (consts::worldWidth + consts::borderWidth) / consts::wallMeshX, // add borderwidth for no jagged corners
                consts::borderWidth / consts::wallMeshY,
                consts::borderHeight / consts::wallMeshHeight});

        // Right
        ctx.data().borders[1] = ctx.makeRenderableEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            ctx.data().borders[1],
            Vector3{
                consts::worldWidth / 2.f,
                0,
                0,
            },
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::Wall,
            ResponseType::Static,
            Diag3x3{
                consts::borderWidth / consts::wallMeshX,
                (consts::worldLength + consts::borderWidth) / consts::wallMeshY, // add borderwidth for no jagged corners
                consts::borderHeight / consts::wallMeshHeight,
            });

        // Left
        ctx.data().borders[2] = ctx.makeRenderableEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            ctx.data().borders[2],
            Vector3{
                0,
                consts::worldLength / 2.f,
                0,
            },
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::Wall,
            ResponseType::Static,
            Diag3x3{
                (consts::worldWidth + consts::borderWidth) / consts::wallMeshX, // add borderwidth for no jagged corners
                consts::borderWidth / consts::wallMeshY,
                consts::borderHeight / consts::wallMeshHeight,
            });

        // Top
        ctx.data().borders[3] = ctx.makeRenderableEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            ctx.data().borders[3],
            Vector3{
                -consts::worldWidth / 2.f,
                0,
                0,
            },
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::Wall,
            ResponseType::Static,
            Diag3x3{
                consts::borderWidth / consts::wallMeshX,
                (consts::worldLength + consts::borderWidth) / consts::wallMeshY, // add borderwidth for no jagged corners
                consts::borderHeight / consts::wallMeshHeight,
            });

        // initialized on reset
        ctx.data().episodeTracker = ctx.makeEntity<EpisodeTracker>();

        // MacGuffin
        ctx.data().macguffin = ctx.makeRenderableEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            ctx.data().macguffin,
            Vector3{0, 0, 0}, // position is reset on level start
            Quat{1, 0, 0, 0},
            SimObject::MacGuffin,
            EntityType::MacGuffin,
            ResponseType::Dynamic,
            Diag3x3{
                consts::macguffinSize / consts::cubeMeshSize,
                consts::macguffinSize / consts::cubeMeshSize,
                consts::macguffinSize / consts::cubeMeshSize});

        // Goal
        Entity goal = ctx.data().goal = ctx.makeRenderableEntity<Goal>();
        ctx.get<Rotation>(goal) = Quat{1, 0, 0, 0};
        ctx.get<Scale>(goal) = Diag3x3{
            consts::goalSize / consts::cubeMeshSize,
            consts::goalSize / consts::cubeMeshSize,
            0.1f};
        ctx.get<ObjectID>(goal) = ObjectID{(int32_t)SimObject::Goal};
        ctx.get<EntityType>(goal) = EntityType::Goal;
        // position to be initialized on level reset

        // Agents
        // Note that this leaves a lot of components
        // uninitialized, these will be set during world generation, which is
        // called for every episode.
        // We create the max number, even if all are not used. This is because agents
        // must persist between episodes or else you can't export their data (for some reason)
        for (CountT i = 0; i < consts::maxAgents; ++i)
        {
            Entity agent = ctx.data().agents[i] =
                ctx.makeRenderableEntity<Agent>();

            // Create a render view for the agent
            if (ctx.data().enableRender)
            {
                render::RenderingSystem::attachEntityToView(ctx,
                                                            agent,
                                                            100.f,
                                                            .25f * consts::agentSize,
                                                            0.5 * consts::agentSize * math::up);
            }

            ctx.get<Scale>(agent) = Diag3x3{
                consts::agentSize / consts::agentMeshSize,
                consts::agentSize / consts::agentMeshSize,
                consts::agentSize / consts::agentMeshSize};
            ctx.get<ObjectID>(agent) = ObjectID{(int32_t)SimObject::Agent};
            ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
            ctx.get<GrabState>(agent).constraintEntity = Entity::none();
            ctx.get<EntityType>(agent) = EntityType::Agent;
        }
    }

    // Enum to identify which border/wall
    enum class Border
    {
        Top = 0,
        Right = 1,
        Bottom = 2,
        Left = 3
    };

    // Structure to hold barrier placement information
    struct BarrierPlacement
    {
        float x;
        float y;
        float width;
        float height;
    };

    struct CubePlacement
    {
        float x;
        float y;
        float scale;
    };

    struct AgentPlacement
    {
        float x;
        float y;
        float angle;
    };

    struct MacGuffinPlacement
    {
        float x;
        float y;
        Border wall; // Which wall/border it's placed along
    };

    struct GoalPlacement
    {
        float x;
        float y;
        Border wall; // Which wall/border it's placed along
    };

    struct LevelPlacements
    {
        BarrierPlacement barrierPlacements[consts::maxBarriers];
        int32_t numBarriers;

        CubePlacement cubePlacements[consts::maxCubes];
        int32_t numCubes;

        AgentPlacement agentPlacements[consts::maxAgents];
        int32_t numAgents;

        MacGuffinPlacement macguffinPlacement;

        GoalPlacement goalPlacement;
    };

    // Although agents and walls persist between episodes, we still need to
    // re-register them with the broadphase system and, in the case of the agents,
    // reset their positions.
    static void resetPersistentEntities(Engine &ctx,
                                        const LevelPlacements &levelPlacements)
    {
        // floor
        registerRigidBodyEntity(ctx, ctx.data().floorPlane, SimObject::Plane);

        // borders
        for (CountT i = 0; i < 4; i++)
        {
            Entity wall_entity = ctx.data().borders[i];
            registerRigidBodyEntity(ctx, wall_entity, SimObject::Wall);
        }

        // MacGuffin
        Entity macguffin = ctx.data().macguffin;
        Vector3 macguffin_pos{levelPlacements.macguffinPlacement.x, levelPlacements.macguffinPlacement.y, consts::macguffinSize / 2.f};
        registerRigidBodyEntity(ctx, macguffin, SimObject::MacGuffin);
        ctx.get<Position>(macguffin) = macguffin_pos;
        ctx.get<Rotation>(macguffin) = Quat{1, 0, 0, 0};
        ctx.get<Velocity>(macguffin) = {Vector3::zero(), Vector3::zero()};
        ctx.get<ExternalForce>(macguffin) = Vector3::zero();
        ctx.get<ExternalTorque>(macguffin) = Vector3::zero();

        // Goal
        Entity goal = ctx.data().goal;
        Vector3 goal_pos{levelPlacements.goalPlacement.x, levelPlacements.goalPlacement.y, 0.0f};
        ctx.get<Position>(goal) = goal_pos;

        // Episode Tracker
        Entity episodeTracker = ctx.data().episodeTracker;
        ctx.get<StepsRemaining>(episodeTracker).t = consts::episodeLen;
        ctx.get<RewardHelper>(episodeTracker).starting_dist = -1.0f; // initialized in rewardsystem
        ctx.get<RewardHelper>(episodeTracker).prev_dist = -1.0f;     // initialized in rewardsystem

        // NumAgents
        int32_t numAgents = levelPlacements.numAgents;

        // Agents
        for (int32_t i = 0; i < consts::maxAgents; i++)
        {
            // for every ant, set most things to the default values
            Entity agent_entity = ctx.data().agents[i];
            registerRigidBodyEntity(ctx, agent_entity, SimObject::Agent);
            auto &grab_state = ctx.get<GrabState>(agent_entity);
            if (grab_state.constraintEntity != Entity::none())
            {
                ctx.destroyEntity(grab_state.constraintEntity);
                grab_state.constraintEntity = Entity::none();
            }
            ctx.get<Velocity>(agent_entity) = {
                Vector3::zero(),
                Vector3::zero(),
            };
            ctx.get<ExternalForce>(agent_entity) = Vector3::zero();
            ctx.get<ExternalTorque>(agent_entity) = Vector3::zero();
            ctx.get<Action>(agent_entity) = Action{
                .moveAmount = 0,
                .moveAngle = 0,
                .rotate = consts::numTurnBuckets / 2,
                .grab = 0,
            };
            // set positions and alive status of the agent's we'll actually use
            if (i < numAgents)
            {
                // Place the agents near the starting wall
                ctx.get<Position>(agent_entity) = Vector3{
                    levelPlacements.agentPlacements[i].x,
                    levelPlacements.agentPlacements[i].y,
                    0.0f,
                };
                ctx.get<Rotation>(agent_entity) = Quat::angleAxis(
                    levelPlacements.agentPlacements[i].angle, math::up);
                ctx.get<Active>(agent_entity).v = 1;
            }
            // for the rest of the ants, they go to jail
            else
            {
                ctx.get<Position>(agent_entity) = Vector3{
                    consts::worldWidth / 2.0f + (50.0f + 2 * consts::agentSize) * i,
                    0.0f,
                    0.0f,
                };
                ctx.get<Rotation>(agent_entity) = Quat{1, 0, 0, 0};
                ctx.get<Active>(agent_entity).v = 0;
            }
        }
    }

    static Entity makeCube(Engine &ctx,
                           const CubePlacement &placement)
    {
        Entity cube = ctx.makeRenderableEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            cube,
            Vector3{
                placement.x,
                placement.y,
                consts::cubeSize / 2.f},
            Quat{1, 0, 0, 0},
            SimObject::Cube,
            EntityType::Cube,
            ResponseType::Dynamic,
            Diag3x3{
                consts::cubeSize * placement.scale / consts::cubeMeshSize,
                consts::cubeSize * placement.scale / consts::cubeMeshSize,
                consts::cubeSize * placement.scale / consts::cubeMeshSize});
        registerRigidBodyEntity(ctx, cube, SimObject::Cube);
        return cube;
    }

    static Entity makeBarrier(Engine &ctx,
                              const BarrierPlacement &placement)
    {
        Entity barrier = ctx.makeRenderableEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            barrier,
            Vector3{placement.x, placement.y, 0},
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::Wall,
            ResponseType::Static,
            Diag3x3{
                placement.width / consts::wallMeshX,
                placement.height / consts::wallMeshY,
                consts::barrierHeight / consts::wallMeshHeight});
        registerRigidBodyEntity(ctx, barrier, SimObject::Wall);
        return barrier;
    }

    // Determine a suitable position for the macguffin
    // nothing needs to have been previously placed
    static void determineMacGuffinPlacement(Engine &ctx, LevelPlacements &levelPlacements)
    {
        // Choose a border randomly
        Border border = static_cast<Border>(randIntBetween(ctx, 0, 3));

        // Buffer from the wall to ensure the macguffin doesn't clip into the border
        float buffer = 0.5f * consts::macguffinSize + 0.5f * consts::borderWidth;

        // Room boundaries accounting for the border walls
        float minX = -consts::worldWidth / 2.0f + buffer;
        float maxX = consts::worldWidth / 2.0f - buffer;
        float minY = -consts::worldLength / 2.0f + buffer;
        float maxY = consts::worldLength / 2.0f - buffer;

        // Starting position
        float x = 0.0f;
        float y = 0.0f;

        // Place the macguffin near the chosen border
        float offset = randBetween(ctx, consts::minBorderSpawnBuffer, consts::maxBorderSpawnBuffer);
        switch (border)
        {
        case Border::Top: // Top border
            x = randBetween(ctx, minX, maxX);
            y = maxY - offset; // 0-3 units from the border
            break;
        case Border::Right:    // Right border
            x = maxX - offset; // 0-3 units from the border
            y = randBetween(ctx, minY, maxY);
            break;
        case Border::Bottom: // Bottom border
            x = randBetween(ctx, minX, maxX);
            y = minY + offset; // 0-3 units from the border
            break;
        case Border::Left:     // Left border
            x = minX + offset; // 0-3 units from the border
            y = randBetween(ctx, minY, maxY);
            break;
        }

        levelPlacements.macguffinPlacement = MacGuffinPlacement{x, y, border};
    }

    // Determine a suitable position for the goal along the perimeter of the room
    // assumes macguffin has already been placed
    static void determineGoalPlacement(Engine &ctx, LevelPlacements &levelPlacements)
    {
        // Place goal on the opposite wall
        Border goalWall = static_cast<Border>((static_cast<int>(levelPlacements.macguffinPlacement.wall) + 2) % 4);

        // Buffer from the wall to ensure the goal doesn't clip into the border
        float buffer = 0.5f * consts::goalSize + 0.5f * consts::borderWidth;

        // Room boundaries accounting for the border walls
        float minX = -consts::worldWidth / 2.0f + buffer;
        float maxX = consts::worldWidth / 2.0f - buffer;
        float minY = -consts::worldLength / 2.0f + buffer;
        float maxY = consts::worldLength / 2.0f - buffer;

        // Starting position
        float x = 0.0f;
        float y = 0.0f;

        // Place the goal near the chosen border
        float offset = randBetween(ctx, consts::minBorderSpawnBuffer, consts::maxBorderSpawnBuffer);
        switch (goalWall)
        {
        case Border::Top: // Top border
            x = randBetween(ctx, minX, maxX);
            y = maxY - offset;
            break;
        case Border::Right: // Right border
            x = maxX - offset;
            y = randBetween(ctx, minY, maxY);
            break;
        case Border::Bottom: // Bottom border
            x = randBetween(ctx, minX, maxX);
            y = minY + offset;
            break;
        case Border::Left: // Left border
            x = minX + offset;
            y = randBetween(ctx, minY, maxY);
            break;
        }

        levelPlacements.goalPlacement = GoalPlacement{x, y, goalWall};
    }

    static void determineBarrierPlacements(Engine &ctx, const int32_t numBarriers, LevelPlacements &levelPlacements)
    {
        levelPlacements.numBarriers = 0;

        if (numBarriers <= 0)
        {
            return;
        }

        // Room boundaries
        float minX = -consts::worldWidth / 2.0f + consts::borderWidth / 2.0f;
        float maxX = consts::worldWidth / 2.0f - consts::borderWidth / 2.0f;
        float minY = -consts::worldLength / 2.0f + consts::borderWidth / 2.0f;
        float maxY = consts::worldLength / 2.0f - consts::borderWidth / 2.0f;

        // Try up to the maximum number of attempts for barrier placement
        int attempt = 0;
        levelPlacements.numBarriers = 0;

        while (levelPlacements.numBarriers < numBarriers && attempt < consts::maxBarrierPlacementAttempts)
        {
            attempt++;

            // Decide if barrier will be horizontal or vertical
            bool isHorizontal = randBetween(ctx, 0.0f, 1.0f) < 0.5f;

            // Barrier size
            float barrierLength = randBetween(ctx, consts::minBarrierLength, consts::maxBarrierLength);
            float barrierWidth = consts::borderWidth;

            // Determine barrier position
            float x = 0.0f;
            float y = 0.0f;
            float width = 0.0f;
            float height = 0.0f;

            if (isHorizontal)
            {
                // Horizontal barrier
                width = barrierLength;
                height = barrierWidth;

                // Random position within room bounds
                x = randBetween(ctx, minX + width / 2, maxX - width / 2);
                y = randBetween(ctx, minY + height / 2, maxY - height / 2);
            }
            else
            {
                // Vertical barrier
                width = barrierWidth;
                height = barrierLength;

                // Random position within room bounds
                x = randBetween(ctx, minX + width / 2, maxX - width / 2);
                y = randBetween(ctx, minY + height / 2, maxY - height / 2);
            }

            bool overlapsWithMacguffin =
                (std::abs(x - levelPlacements.macguffinPlacement.x) < (width / 2.0f + consts::macguffinSize / 2.0f) + consts::barrierMacguffinBuffer) &&
                (std::abs(y - levelPlacements.macguffinPlacement.y) < (height / 2.0f + consts::macguffinSize / 2.0f) + consts::barrierMacguffinBuffer);

            if (overlapsWithMacguffin)
            {
                continue; // Skip this attempt
            }

            bool overlapsWithGoal =
                (std::abs(x - levelPlacements.goalPlacement.x) < (width / 2.0f + consts::goalSize / 2.0f) + consts::barrierGoalBuffer) &&
                (std::abs(y - levelPlacements.goalPlacement.y) < (height / 2.0f + consts::goalSize / 2.0f) + consts::barrierGoalBuffer);

            if (overlapsWithGoal)
            {
                continue; // Skip this attempt
            }

            // Check if barrier overlaps with existing barriers
            bool overlapsWithBarrier = false;
            for (int32_t i = 0; i < levelPlacements.numBarriers; i++)
            {
                const BarrierPlacement &existingBarrier = levelPlacements.barrierPlacements[i];
                // Simple overlap check (approximate)
                if (std::abs(x - existingBarrier.x) < (width + existingBarrier.width) / 2 + consts::barrierBarrierBuffer &&
                    std::abs(y - existingBarrier.y) < (height + existingBarrier.height) / 2 + consts::barrierBarrierBuffer)
                {
                    overlapsWithBarrier = true;
                    break;
                }
            }

            if (overlapsWithBarrier)
            {
                break;
            }

            BarrierPlacement barrier;
            barrier.x = x;
            barrier.y = y;
            barrier.width = width;
            barrier.height = height;

            levelPlacements.barrierPlacements[levelPlacements.numBarriers] = barrier;
            levelPlacements.numBarriers++;
        }
    }

    static void determineCubePlacements(Engine &ctx, const int32_t numCubes, LevelPlacements &levelPlacements)
    {
        levelPlacements.numCubes = 0;

        if (numCubes <= 0)
        {
            return;
        }

        // Define world edges (absolute edges of the simulation area)
        const float worldEdgeMinX = -consts::worldWidth / 2.0f;
        const float worldEdgeMaxX = consts::worldWidth / 2.0f;
        const float worldEdgeMinY = -consts::worldLength / 2.0f;
        const float worldEdgeMaxY = consts::worldLength / 2.0f;

        // Define placement boundaries (inside the border walls)
        const float placementBoundMinX = worldEdgeMinX + consts::borderWidth / 2.0f;
        const float placementBoundMaxX = worldEdgeMaxX - consts::borderWidth / 2.0f;
        const float placementBoundMinY = worldEdgeMinY + consts::borderWidth / 2.0f;
        const float placementBoundMaxY = worldEdgeMaxY - consts::borderWidth / 2.0f;

        int attempt = 0;

        while (levelPlacements.numCubes < numCubes && attempt < consts::maxCubePlacementAttempts)
        {
            attempt++;

            // Random scale for the cube
            float scale = randBetween(ctx, consts::cubeMinScaleFactor, consts::cubeMaxScaleFactor);

            // Current cube's half-extent (assuming consts::cubeSize is base half-width/radius)
            float currentCubeHalfExtent = consts::cubeSize * scale / 2.0f;
            float macguffinHalfExtent = consts::macguffinSize / 2.0f;

            // Determine valid range for placing the center of the cube
            float placeableMinX = placementBoundMinX + currentCubeHalfExtent;
            float placeableMaxX = placementBoundMaxX - currentCubeHalfExtent;
            float placeableMinY = placementBoundMinY + currentCubeHalfExtent;
            float placeableMaxY = placementBoundMaxY - currentCubeHalfExtent;

            // If the placeable area is invalid (e.g., cube too large for the space), skip or break.
            if (placeableMinX >= placeableMaxX || placeableMinY >= placeableMaxY)
            {
                // This might happen if world is too small or objects too large relative to consts::borderWidth
                // Or if numCubes is very high and remaining space is fragmented.
                // Depending on desired behavior, could log an error or simply stop trying.
                break;
            }

            // Random position for the center of the cube
            float x = randBetween(ctx, placeableMinX, placeableMaxX);
            float y = randBetween(ctx, placeableMinY, placeableMaxY);

            // --- Overlap Checks using AABB ---

            // 1. Check overlap with Macguffin
            bool overlapsWithMacguffin =
                (std::abs(x - levelPlacements.macguffinPlacement.x) < (currentCubeHalfExtent + macguffinHalfExtent + consts::cubeMacguffinBuffer)) &&
                (std::abs(y - levelPlacements.macguffinPlacement.y) < (currentCubeHalfExtent + macguffinHalfExtent + consts::cubeMacguffinBuffer));

            if (overlapsWithMacguffin)
            {
                continue; // Skip this attempt, try a new placement
            }

            // its fine to overlap with goal

            // 3. Check overlap with Barriers
            bool objectOverlapsWithBarrier = false;
            for (int32_t i = 0; i < levelPlacements.numBarriers; i++)
            {
                const BarrierPlacement barrier = levelPlacements.barrierPlacements[i];

                float barrierHalfWidth = barrier.width / 2.0f;
                float barrierHalfHeight = barrier.height / 2.0f;

                if ((std::abs(x - barrier.x) < (currentCubeHalfExtent + barrierHalfWidth + consts::cubeBarrierBuffer)) &&
                    (std::abs(y - barrier.y) < (currentCubeHalfExtent + barrierHalfHeight + consts::cubeBarrierBuffer)))
                {
                    objectOverlapsWithBarrier = true;
                    break; // Found an overlap with a barrier
                }
            }
            if (objectOverlapsWithBarrier)
            {
                continue; // Skip this attempt
            }

            // 4. Check overlap with existing Cubes
            bool overlapsWithExistingCube = false;
            for (int32_t i = 0; i < levelPlacements.numCubes; i++)
            {
                const CubePlacement &existingCube = levelPlacements.cubePlacements[i];
                float existingCubeHalfExtent = consts::cubeSize * existingCube.scale;

                if ((std::abs(x - existingCube.x) < (currentCubeHalfExtent + existingCubeHalfExtent + consts::cubeCubeBuffer)) &&
                    (std::abs(y - existingCube.y) < (currentCubeHalfExtent + existingCubeHalfExtent + consts::cubeCubeBuffer)))
                {
                    overlapsWithExistingCube = true;
                    break; // Found an overlap with another cube
                }
            }
            if (overlapsWithExistingCube)
            {
                continue; // Skip this attempt
            }

            // If all checks passed, the placement is valid
            CubePlacement cubePlacement;
            cubePlacement.x = x;
            cubePlacement.y = y;
            cubePlacement.scale = scale;

            levelPlacements.cubePlacements[levelPlacements.numCubes] = cubePlacement;
            levelPlacements.numCubes++;
        }
    }

    static void determineAgentPlacements(Engine &ctx, int32_t numAgents, LevelPlacements &levelPlacements)
    {
        levelPlacements.numAgents = 0;

        // Get how many agents we need to place
        if (numAgents <= 0)
        {
            return;
        }

        // Room boundaries
        float minX = -consts::worldWidth / 2.0f;
        float maxX = consts::worldWidth / 2.0f;
        float minY = -consts::worldLength / 2.0f;
        float maxY = consts::worldLength / 2.0f;

        float borderBuffer = consts::borderWidth / 2.0f + consts::agentSize / 2.0f;

        // Adjusted room boundaries accounting for border walls
        float adjustedMinX = minX + borderBuffer;
        float adjustedMaxX = maxX - borderBuffer;
        float adjustedMinY = minY + borderBuffer;
        float adjustedMaxY = maxY - borderBuffer;

        // Try up to the maximum number of attempts per agent

        // Try to place each agent
        for (int32_t agentIndex = 0; agentIndex < numAgents; agentIndex++)
        {
            int attempts = 0;
            bool placedSuccessfully = false;

            while (!placedSuccessfully && attempts < consts::maxAgentPlacementAttemptsPerAgent)
            {
                attempts++;

                // Random position within adjusted room bounds
                float x = randBetween(ctx, adjustedMinX, adjustedMaxX);
                float y = randBetween(ctx, adjustedMinY, adjustedMaxY);

                // Random angle (orientation)
                float angle = randBetween(ctx, 0.0f, 2.0f * 3.14159f); // 0 to 2π radians

                // Check if agent overlaps with macguffin
                bool overlapsWithMacguffin = false;
                if (std::abs(x - levelPlacements.macguffinPlacement.x) < (consts::agentSize / 2.0f + consts::macguffinSize / 2.0f + consts::agentMacguffinBuffer) &&
                    std::abs(y - levelPlacements.macguffinPlacement.y) < (consts::agentSize / 2.0f + consts::macguffinSize / 2.0f + consts::agentMacguffinBuffer))
                {
                    overlapsWithMacguffin = true;
                }

                if (overlapsWithMacguffin)
                {
                    continue; // Skip this attempt
                }

                // its's fine if they spawn on the goal; don't check that

                // Check if agent overlaps with walls
                bool overlapsWithBarrier = false;
                for (int32_t i = 0; i < levelPlacements.numBarriers; i++)
                {
                    const BarrierPlacement &barrier = levelPlacements.barrierPlacements[i];
                    // Simple box-based distance check
                    if (std::abs(x - barrier.x) < (consts::agentSize / 2.0f + barrier.width / 2.0f + consts::agentBarrierBuffer) &&
                        std::abs(y - barrier.y) < (consts::agentSize / 2.0f + barrier.height / 2.0f + consts::agentBarrierBuffer))
                    {
                        overlapsWithBarrier = true;
                        break;
                    }
                }

                if (overlapsWithBarrier)
                {
                    continue; // Skip this attempt
                }

                // Check if agent overlaps with cubes
                bool overlapsWithCube = false;
                // for (const auto &cube : cubePlacements) {
                for (int32_t i = 0; i < levelPlacements.numCubes; i++)
                {
                    const CubePlacement &cube = levelPlacements.cubePlacements[i];
                    // Simple box-based distance check
                    if (std::abs(x - cube.x) < (consts::agentSize / 2.0f + consts::movableObjectSize * cube.scale / 2.0f + consts::agentBarrierBuffer) &&
                        std::abs(y - cube.y) < (consts::agentSize / 2.0f + consts::movableObjectSize * cube.scale / 2.0f + consts::agentBarrierBuffer))
                    {
                        overlapsWithCube = true;
                        break;
                    }
                }

                if (overlapsWithCube)
                {
                    continue; // Skip this attempt
                }

                // Check if agent overlaps with already placed agents
                bool overlapsWithAgent = false;
                // for (const auto &existingAgent : agentPlacements) {
                for (int32_t i = 0; i < levelPlacements.numAgents; i++)
                {
                    const AgentPlacement &existingAgent = levelPlacements.agentPlacements[i];

                    float distToAgent = std::sqrt(
                        std::pow(x - existingAgent.x, 2) +
                        std::pow(y - existingAgent.y, 2));

                    if (distToAgent < consts::agentSize + consts::agentAgentBuffer)
                    {
                        overlapsWithAgent = true;
                        break;
                    }
                }

                if (overlapsWithAgent)
                {
                    continue;
                }

                // Agent position is valid, add it
                AgentPlacement agent;
                agent.x = x;
                agent.y = y;
                agent.angle = angle;
                levelPlacements.agentPlacements[levelPlacements.numAgents] = agent;
                levelPlacements.numAgents++;
                placedSuccessfully = true;
            }

            // If we couldn't place this agent after all attempts, stop trying to place more
            if (!placedSuccessfully)
            {
                break;
            }
        }
    }

    static void generateLevel(Engine &ctx, const LevelPlacements &levelPlacements)
    {
        // some cubes
        for (CountT i = 0; i < levelPlacements.numCubes; i++)
        {
            Entity cube = makeCube(ctx, levelPlacements.cubePlacements[i]);
            ctx.data().cubes[i] = cube;
        }
        for (int32_t i = levelPlacements.numCubes; i < consts::maxCubes; i++)
        {
            ctx.data().cubes[i] = Entity::none();
        }

        // some barriers
        for (int32_t i = 0; i < levelPlacements.numBarriers; i++)
        {
            Entity barrier = makeBarrier(ctx, levelPlacements.barrierPlacements[i]);
            ctx.data().barriers[i] = barrier;
        }
        for (int32_t i = levelPlacements.numBarriers; i < consts::maxBarriers; i++)
        {
            ctx.data().barriers[i] = Entity::none();
        }
    }

    // Randomly generate a new world for a training episode
    void generateWorld(Engine &ctx)
    {
        int32_t attemptedNumCubes = randIntBetween(ctx, consts::minCubes, consts::maxCubes);
        int32_t attemptedNumBarriers = randIntBetween(ctx, consts::minBarriers, consts::maxBarriers);
        int32_t attemptedNumAgents = randIntBetween(ctx, consts::minAgents, consts::maxAgents);
        // note: these counts may be inaccurate after we try to place objects

        // determine placements of objects
        LevelPlacements levelPlacements;
        determineMacGuffinPlacement(ctx, levelPlacements);
        determineGoalPlacement(ctx, levelPlacements);
        determineBarrierPlacements(ctx, attemptedNumBarriers, levelPlacements);
        determineCubePlacements(ctx, attemptedNumCubes, levelPlacements);
        determineAgentPlacements(ctx, attemptedNumAgents, levelPlacements);

        // actual make/reset entities
        resetPersistentEntities(ctx, levelPlacements);
        generateLevel(ctx, levelPlacements);
    }

}
