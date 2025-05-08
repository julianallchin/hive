#include "level_gen.hpp"

namespace madEscape
{

    using namespace madrona;
    using namespace madrona::math;
    using namespace madrona::phys;

    // Helper functions for random number generation

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

    // Creates floor and outer walls entities.
    // These entities persist across all episodes.
    void createPersistentEntities(Engine &ctx)
    {
        // Create the floor entity, just a simple static plane.
        ctx.data().floorPlane = ctx.makeRenderableEntity<Plane>();
        setupRigidBodyEntity(
            ctx,
            ctx.data().floorPlane,
            Vector3{0, 0, 0},
            Quat{1, 0, 0, 0},
            SimObject::Plane,
            EntityType::None, // Floor plane type should never be queried
            ResponseType::Static);

        // Create the outer wall entities
        // Bottom wall
        ctx.data().borders[0] = ctx.makeRenderableEntity<Wall>();
        setupRigidBodyEntity(
            ctx,
            ctx.data().borders[0],
            Vector3{
                0,
                -consts::worldWidth / 2.f + consts::wallWidth / 2.f,
                0,
            },
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::Wall,
            ResponseType::Static,
            Diag3x3{
                consts::worldWidth,
                consts::wallWidth,
                2.f,
            });

        // Right wall
        ctx.data().borders[1] = ctx.makeRenderableEntity<Wall>();
        setupRigidBodyEntity(
            ctx,
            ctx.data().borders[1],
            Vector3{
                consts::worldWidth / 2.f - consts::wallWidth / 2.f,
                0,
                0,
            },
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::Wall,
            ResponseType::Static,
            Diag3x3{
                consts::wallWidth,
                consts::worldWidth,
                2.f,
            });

        // Top wall
        ctx.data().borders[2] = ctx.makeRenderableEntity<Wall>();
        setupRigidBodyEntity(
            ctx,
            ctx.data().borders[2],
            Vector3{
                0,
                consts::worldWidth / 2.f - consts::wallWidth / 2.f,
                0,
            },
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::Wall,
            ResponseType::Static,
            Diag3x3{
                consts::worldWidth,
                consts::wallWidth,
                2.f,
            });

        // Left wall
        ctx.data().borders[3] = ctx.makeRenderableEntity<Wall>();
        setupRigidBodyEntity(
            ctx,
            ctx.data().borders[3],
            Vector3{
                -consts::worldWidth / 2.f + consts::wallWidth / 2.f,
                0,
                0,
            },
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::Wall,
            ResponseType::Static,
            Diag3x3{
                consts::wallWidth,
                consts::worldWidth,
                2.f,
            });
    }

    // Persistent entities (walls) need to be re-registered with the broadphase system.
    void resetPersistentEntities(Engine &ctx)
    {
        // Register the floor and all border walls
        registerRigidBodyEntity(ctx, ctx.data().floorPlane, SimObject::Plane);

        for (CountT i = 0; i < 4; i++)
        {
            Entity wall_entity = ctx.data().borders[i];
            registerRigidBodyEntity(ctx, wall_entity, SimObject::Wall);
        }
    }

    // Creates the macguffin (object that ants need to move)
    static Entity createMacguffin(Engine &ctx, float x, float y)
    {
        Entity macguffin = ctx.makeRenderableEntity<Macguffin>();
        setupRigidBodyEntity(
            ctx,
            macguffin,
            Vector3{x, y, consts::macguffinRadius},
            Quat{1, 0, 0, 0},
            SimObject::Macguffin,
            EntityType::Macguffin,
            ResponseType::Dynamic,
            Diag3x3{
                consts::macguffinRadius * 2,
                consts::macguffinRadius * 2,
                consts::macguffinRadius * 2});
        registerRigidBodyEntity(ctx, macguffin, SimObject::Macguffin);

        return macguffin;
    }

    // Creates the goal (target location for the macguffin)
    static Entity createGoal(Engine &ctx, float x, float y)
    {
        // Goal is non-physical, just a marker
        Entity goal = ctx.makeRenderableEntity<Goal>();

        ctx.get<Position>(goal) = Vector3{x, y, 0.1f};
        ctx.get<Rotation>(goal) = Quat{1, 0, 0, 0};
        ctx.get<Scale>(goal) = Diag3x3{
            consts::goalRadius * 2,
            consts::goalRadius * 2,
            0.1f};
        ctx.get<ObjectID>(goal) = ObjectID{(int32_t)SimObject::Goal};
        ctx.get<EntityType>(goal) = EntityType::Goal;

        return goal;
    }

    // Creates a movable obstacle object
    static Entity createMovableObject(Engine &ctx, float x, float y, float scale = 1.0f)
    {
        Entity obj = ctx.makeRenderableEntity<MovableObject>();
        setupRigidBodyEntity(
            ctx,
            obj,
            Vector3{x, y, consts::movableObjectRadius * scale},
            Quat{1, 0, 0, 0},
            SimObject::MovableObject,
            EntityType::MovableObject,
            ResponseType::Dynamic,
            Diag3x3{
                consts::movableObjectRadius * 2 * scale,
                consts::movableObjectRadius * 2 * scale,
                consts::movableObjectRadius * 2 * scale});
        registerRigidBodyEntity(ctx, obj, SimObject::MovableObject);

        return obj;
    }

    // Creates an additional wall obstacle
    static Entity createWall(Engine &ctx, float x, float y, float width, float height)
    {
        Entity wall = ctx.makeRenderableEntity<Wall>();
        setupRigidBodyEntity(
            ctx,
            wall,
            Vector3{x, y, 0},
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::Wall,
            ResponseType::Static,
            Diag3x3{
                width,
                height,
                2.0f});
        registerRigidBodyEntity(ctx, wall, SimObject::Wall);

        return wall;
    }

    // Place the goal at a random location along the perimeter of the room
    static Entity placeGoal(Engine &ctx)
    {
        // Choose a side of the room (0=top, 1=right, 2=bottom, 3=left)
        int side = ctx.data().rng.sampleI32(0, 4);
        float padding = consts::goalRadius * 2.0f;
        float halfWidth = consts::worldWidth / 2.0f - padding;

        float x = 0.0f;
        float y = 0.0f;

        switch (side)
        {
        case 0: // Top
            x = randInRangeCentered(ctx, halfWidth);
            y = halfWidth;
            break;
        case 1: // Right
            x = halfWidth;
            y = randInRangeCentered(ctx, halfWidth);
            break;
        case 2: // Bottom
            x = randInRangeCentered(ctx, halfWidth);
            y = -halfWidth;
            break;
        case 3: // Left
            x = -halfWidth;
            y = randInRangeCentered(ctx, halfWidth);
            break;
        }

        return createGoal(ctx, x, y);
    }

    // Place the macguffin in a random position
    static Entity placeMacguffin(Engine &ctx)
    {
        // Place macguffin opposite of the goal
        // Get goal position
        LevelState &level = ctx.singleton<LevelState>();
        Vector3 goalPos = ctx.get<Position>(level.goal);

        // Place macguffin on opposite side, with some randomness
        float angle = atan2f(goalPos.y, goalPos.x) + math::pi +
                      randInRangeCentered(ctx, math::pi / 4.0f);

        // Distance should be a good portion of world size
        float dist = consts::worldWidth * 0.3f * ctx.data().rng.sampleUniform();

        float x = cosf(angle) * dist;
        float y = sinf(angle) * dist;

        return createMacguffin(ctx, x, y);
    }

    // Place movable objects in the world
    static void placeMovableObjects(Engine &ctx, CountT numObjects)
    {
        LevelState &level = ctx.singleton<LevelState>();

        // Get positions of goal and macguffin to avoid placing objects too close
        Vector3 goalPos = ctx.get<Position>(level.goal);
        Vector3 macguffinPos = ctx.get<Position>(level.macguffin);

        float minDistance = consts::worldWidth * 0.15f; // Minimum distance from goal/macguffin
        float maxAttempts = 30;                         // Max attempts to place an object

        for (CountT i = 0; i < numObjects; i++)
        {
            float x, y;
            bool validPosition = false;
            CountT attempts = 0;

            // Try to find a valid position
            while (!validPosition && attempts < maxAttempts)
            {
                // Random position within world bounds
                x = randInRangeCentered(ctx, consts::worldWidth * 0.4f);
                y = randInRangeCentered(ctx, consts::worldWidth * 0.4f);

                // Check if far enough from goal and macguffin
                float distToGoal = sqrtf(powf(x - goalPos.x, 2) + powf(y - goalPos.y, 2));
                float distToMacguffin = sqrtf(powf(x - macguffinPos.x, 2) + powf(y - macguffinPos.y, 2));

                if (distToGoal > minDistance && distToMacguffin > minDistance)
                {
                    validPosition = true;
                }

                attempts++;
            }

            if (validPosition)
            {
                // Random size variation
                float scale = randBetween(ctx, 0.8f, 1.2f);
                Entity obj = createMovableObject(ctx, x, y, scale);
                level.movable_objects[i] = obj;
            }
            else
            {
                // If we can't find a valid position, don't create the object
                level.movable_objects[i] = Entity::none();
            }
        }

        level.num_current_movable_objects = numObjects;
    }

    // Place additional walls in the world
    static void placeWalls(Engine &ctx, CountT numWalls)
    {
        LevelState &level = ctx.singleton<LevelState>();

        // Get positions of goal and macguffin to avoid blocking paths
        Vector3 goalPos = ctx.get<Position>(level.goal);
        Vector3 macguffinPos = ctx.get<Position>(level.macguffin);

        // Calculate vector from macguffin to goal
        float dx = goalPos.x - macguffinPos.x;
        float dy = goalPos.y - macguffinPos.y;
        float pathAngle = atan2f(dy, dx);

        for (CountT i = 0; i < numWalls; i++)
        {
            // Place walls perpendicular to the path from macguffin to goal
            float wallAngle = pathAngle + math::pi / 2.0f + randInRangeCentered(ctx, math::pi / 4.0f);

            // Calculate position - somewhere between macguffin and goal, but offset
            float t = ctx.data().rng.sampleUniform(); // Position along path (0 to 1)
            float pathOffset = consts::worldWidth * 0.2f * ctx.data().rng.sampleUniform();

            float centerX = macguffinPos.x + dx * t + cosf(wallAngle) * pathOffset;
            float centerY = macguffinPos.y + dy * t + sinf(wallAngle) * pathOffset;

            // Make sure wall is within bounds
            centerX = std::max(std::min(centerX, consts::worldWidth / 2.0f - consts::wallWidth),
                               -consts::worldWidth / 2.0f + consts::wallWidth);
            centerY = std::max(std::min(centerY, consts::worldWidth / 2.0f - consts::wallWidth),
                               -consts::worldWidth / 2.0f + consts::wallWidth);

            // Wall dimensions
            float wallLength = randBetween(ctx, consts::worldWidth * 0.1f, consts::worldWidth * 0.25f);
            float wallWidth = consts::wallWidth;

            // Rotate dimensions based on angle
            float width, height;
            if (fabsf(fmodf(wallAngle, math::pi)) < math::pi / 4.0f ||
                fabsf(fmodf(wallAngle, math::pi)) > 3.0f * math::pi / 4.0f)
            {
                // More horizontal
                width = wallLength;
                height = wallWidth;
            }
            else
            {
                // More vertical
                width = wallWidth;
                height = wallLength;
            }

            Entity wall = createWall(ctx, centerX, centerY, width, height);
            level.walls[i] = wall;
        }

        level.num_current_walls = numWalls;
    }

    // Place random number of ants in the world
    static void placeAnts(Engine &ctx, CountT numAnts)
    {
        LevelState &level = ctx.singleton<LevelState>();

        // Store the current number of ants
        ctx.data().numAnts = numAnts;
        level.num_current_ants = numAnts;

        float worldHalfWidth = consts::worldWidth / 2.f - consts::antRadius * 2.f;

        // Create and place ants
        for (CountT i = 0; i < numAnts; ++i)
        {
            Entity ant = ctx.makeRenderableEntity<Ant>();

            // Random position
            float x = randInRangeCentered(ctx, worldHalfWidth);
            float y = randInRangeCentered(ctx, worldHalfWidth);

            // Random rotation
            float angle = ctx.data().rng.sampleUniform() * 2.f * math::pi;
            Quat rot = Quat::angleAxis(angle, math::up);

            setupRigidBodyEntity(
                ctx,
                ant,
                Vector3{x, y, consts::antRadius},
                rot,
                SimObject::Ant,
                EntityType::Ant);

            // Create a render view for a few ants, not for all (performance reasons)
            if (ctx.data().enableRender && i % 10 == 0)
            {
                render::RenderingSystem::attachEntityToView(ctx,
                                                            ant,
                                                            100.f, 0.001f,
                                                            0.5f * math::up);
            }

            ctx.get<Scale>(ant) = Diag3x3{
                consts::antRadius * 2,
                consts::antRadius * 2,
                consts::antRadius * 2};
            ctx.get<GrabState>(ant).constraintEntity = Entity::none();
            ctx.get<GrabState>(ant).grabTarget = Entity::none();

            registerRigidBodyEntity(ctx, ant, SimObject::Ant);

            // Store ant entity in level state
            level.ants[i] = ant;
        }
    }

    // Generate the hive simulation world
    void generateWorld(Engine &ctx, CountT numMovableObjects, CountT numWalls)
    {
        resetPersistentEntities(ctx);

        LevelState &level = ctx.singleton<LevelState>();

        // Create goal first to establish a target location
        level.goal = placeGoal(ctx);

        // Then place macguffin
        level.macguffin = placeMacguffin(ctx);

        // Place movable objects
        placeMovableObjects(ctx, numMovableObjects);

        // Place additional walls
        placeWalls(ctx, numWalls);

        // Randomly determine number of ants within the specified range
        CountT numAnts = ctx.data().rng.sampleI32(consts::minAnts, consts::maxAnts + 1);

        // Place ants
        placeAnts(ctx, numAnts);
    }

}
