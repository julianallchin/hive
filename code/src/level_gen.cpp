#include "level_gen.hpp"

#include <algorithm>

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

    // Creates floor and borders
    // These entities persist across all episodes.
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
        // Bottom wall
        ctx.data().borders[0] = ctx.makeRenderableEntity<PhysicsEntity>();
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
            EntityType::PhysicsEntity,
            ResponseType::Static,
            Diag3x3{
                consts::worldWidth,
                consts::wallWidth,
                2.f,
            });

        // Right wall
        ctx.data().borders[1] = ctx.makeRenderableEntity<PhysicsEntity>();
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
            EntityType::PhysicsEntity,
            ResponseType::Static,
            Diag3x3{
                consts::wallWidth,
                consts::worldWidth,
                2.f,
            });

        // Top wall
        ctx.data().borders[2] = ctx.makeRenderableEntity<PhysicsEntity>();
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
            EntityType::PhysicsEntity,
            ResponseType::Static,
            Diag3x3{
                consts::worldWidth,
                consts::wallWidth,
                2.f,
            });

        // Left wall
        ctx.data().borders[3] = ctx.makeRenderableEntity<PhysicsEntity>();
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
            EntityType::PhysicsEntity,
            ResponseType::Static,
            Diag3x3{
                consts::wallWidth,
                consts::worldWidth,
                2.f,
            });
    }

    // Persistent entities need to be re-registered with the broadphase system.
    void resetPersistentEntities(Engine &ctx)
    {
        // Register the floor
        registerRigidBodyEntity(ctx, ctx.data().floorPlane, SimObject::Plane);

        // Register the borders
        for (CountT i = 0; i < 4; i++)
        {
            registerRigidBodyEntity(ctx, ctx.data().borders[i], SimObject::Plane);
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

        ctx.data().macguffin = macguffin;
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

        ctx.data().goal = goal;
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
        Entity wall = ctx.makeRenderableEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            wall,
            Vector3{x, y, 0},
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::PhysicsEntity,
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
    static Entity placeMacguffin(Engine &ctx, Vector3 goalPos)
    {
        // Place macguffin on opposite side, with some randomness
        float angle = atan2f(goalPos.y, goalPos.x) + math::pi +
                      randInRangeCentered(ctx, math::pi / 4.0f);

        // Distance should be a good portion of world size
        float dist = consts::worldWidth * 0.3f * ctx.data().rng.sampleUniform();

        float x = cosf(angle) * dist;
        float y = sinf(angle) * dist;

        return createMacguffin(ctx, x, y);
    }

    // Checks if a position is within world bounds with a safety margin
    static bool isWithinWorldBounds(float x, float y, float radius) {
        float worldHalfWidth = consts::worldWidth / 2.0f - radius;
        return (x >= -worldHalfWidth && x <= worldHalfWidth &&
                y >= -worldHalfWidth && y <= worldHalfWidth);
    }

    // Check if a position is far enough from specific objects
    static bool isFarEnoughFrom(const Vector3 &pos, const Vector3 &targetPos, float minDistance) {
        float distSq = (pos.x - targetPos.x) * (pos.x - targetPos.x) + 
                       (pos.y - targetPos.y) * (pos.y - targetPos.y);
        return distSq > minDistance * minDistance;
    }

    // Place movable objects in the world
    static void placeMovableObjects(Engine &ctx, CountT numObjects)
    {
        // Ensure we don't exceed the maximum number of movable objects
        if (numObjects > consts::maxMovableObjects) {
            printf("Warning: Requested %d movable objects, but max allowed is %d.\n",
                   (int)numObjects, (int)consts::maxMovableObjects);
            numObjects = consts::maxMovableObjects;
        }
        CountT numToCreate = std::min(numObjects, (CountT)consts::maxMovableObjects);

        if (numToCreate == 0) return;

        // Get positions of goal and macguffin to avoid placing objects too close
        Vector3 goalPos = ctx.get<Position>(ctx.data().goal);
        Vector3 macguffinPos = ctx.get<Position>(ctx.data().macguffin);

        float minDistance = consts::worldWidth * 0.15f;
        float maxAttempts = 30;

        // Divide the world into a grid
        const int gridSize = std::ceil(sqrtf((float)numToCreate)) + 1; // +1 for extra buffer
        float cellWidth = consts::worldWidth * 0.8f / gridSize;
        float worldStart = -consts::worldWidth * 0.4f;

        CountT objectsCreated = 0;
        for (int gridY = 0; gridY < gridSize && objectsCreated < numToCreate; gridY++) {
            for (int gridX = 0; gridX < gridSize && objectsCreated < numToCreate; gridX++) {
                float baseX = worldStart + gridX * cellWidth + cellWidth * 0.5f;
                float baseY = worldStart + gridY * cellWidth + cellWidth * 0.5f;

                float x, y;
                bool validPosition = false;
                CountT attempts = 0;

                // Try to find a valid position within this grid cell
                while (!validPosition && attempts < maxAttempts) {
                    // Add some jitter within the cell
                    x = baseX + randInRangeCentered(ctx, cellWidth * 0.5f);
                    y = baseY + randInRangeCentered(ctx, cellWidth * 0.5f);

                    // Ensure within world bounds
                    if (!isWithinWorldBounds(x, y, consts::movableObjectRadius)) {
                        attempts++;
                        continue;
                    }

                    // Check if far enough from goal and macguffin
                    Vector3 objPos(x, y, 0.0f);
                    if (!isFarEnoughFrom(objPos, goalPos, minDistance) ||
                        !isFarEnoughFrom(objPos, macguffinPos, minDistance)) {
                        attempts++;
                        continue;
                    }

                    validPosition = true;
                    attempts++;
                }

                if (validPosition) {
                    // Random size variation
                    float scale = randBetween(ctx, 0.8f, 1.2f);
                    Entity obj = createMovableObject(ctx, x, y, scale);
                    ctx.data().movableObjects[objectsCreated] = obj;
                    objectsCreated++;
                }
            }
        }

        // If we couldn't place all objects with the grid approach, try random placement for the rest
        for (CountT i = objectsCreated; i < numToCreate; i++) {
            float x, y;
            bool validPosition = false;
            CountT attempts = 0;

            while (!validPosition && attempts < maxAttempts) {
                // Random position within world bounds
                x = randInRangeCentered(ctx, consts::worldWidth * 0.4f);
                y = randInRangeCentered(ctx, consts::worldWidth * 0.4f);

                // Ensure within world bounds
                if (!isWithinWorldBounds(x, y, consts::movableObjectRadius)) {
                    attempts++;
                    continue;
                }

                // Check if far enough from goal and macguffin
                Vector3 objPos(x, y, 0.0f);
                if (!isFarEnoughFrom(objPos, goalPos, minDistance) ||
                    !isFarEnoughFrom(objPos, macguffinPos, minDistance)) {
                    attempts++;
                    continue;
                }

                validPosition = true;
                attempts++;
            }

            if (validPosition) {
                float scale = randBetween(ctx, 0.8f, 1.2f);
                Entity obj = createMovableObject(ctx, x, y, scale);
                ctx.data().movableObjects[i] = obj;
            } else {
                ctx.data().movableObjects[i] = Entity::none();
            }
        }

        // Store the actual number of objects created
        ctx.data().numMovableObjects = numToCreate;
    }

    // Place additional walls in the world
    static void placeWalls(Engine &ctx, CountT numWalls)
    {
        if (numWalls == 0) {
            return;
        }
        
        // Get positions of goal and macguffin to avoid blocking paths
        Vector3 goalPos = ctx.get<Position>(ctx.data().goal);
        Vector3 macguffinPos = ctx.get<Position>(ctx.data().macguffin);

        // Calculate vector from macguffin to goal
        float dx = goalPos.x - macguffinPos.x;
        float dy = goalPos.y - macguffinPos.y;
        float pathAngle = atan2f(dy, dx);
        float pathLength = sqrtf(dx*dx + dy*dy);

        // Divide the path into segments for wall placement
        float segmentLength = pathLength / (numWalls + 1); // +1 to avoid walls at exactly goal/macguffin
        float pathMinDistance = consts::worldWidth * 0.1f; // Min distance from the direct path
        
        for (CountT i = 0; i < numWalls; i++)
        {
            // Place walls perpendicular to the path from macguffin to goal
            // Use deterministic spacing along the path based on index
            float t = (i + 1.0f) / (numWalls + 1.0f); // Position along path (avoids ends)
            
            // Add some randomization to the angle and offset
            float wallAngle = pathAngle + math::pi / 2.0f + randInRangeCentered(ctx, math::pi / 6.0f);
            float pathOffset = pathMinDistance + consts::worldWidth * 0.15f * ctx.data().rng.sampleUniform();

            // Calculate position with offset perpendicular to path
            float centerX = macguffinPos.x + dx * t + cosf(wallAngle) * pathOffset;
            float centerY = macguffinPos.y + dy * t + sinf(wallAngle) * pathOffset;

            // Make sure wall is within bounds with safety margin
            float safetyMargin = consts::wallWidth * 1.5f;
            float worldHalfWidth = consts::worldWidth / 2.0f - safetyMargin;
            centerX = std::max(std::min(centerX, worldHalfWidth), -worldHalfWidth);
            centerY = std::max(std::min(centerY, worldHalfWidth), -worldHalfWidth);

            // Determine wall dimensions - make sure they're appropriate for world size
            float maxWallLength = consts::worldWidth * 0.25f;
            float minWallLength = consts::worldWidth * 0.1f;
            float wallLength = randBetween(ctx, minWallLength, maxWallLength);
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

            // Make sure wall is fully in bounds
            if (centerX - width/2 < -consts::worldWidth/2 + safetyMargin) {
                centerX = -consts::worldWidth/2 + width/2 + safetyMargin;
            }
            if (centerX + width/2 > consts::worldWidth/2 - safetyMargin) {
                centerX = consts::worldWidth/2 - width/2 - safetyMargin;
            }
            if (centerY - height/2 < -consts::worldWidth/2 + safetyMargin) {
                centerY = -consts::worldWidth/2 + height/2 + safetyMargin;
            }
            if (centerY + height/2 > consts::worldWidth/2 - safetyMargin) {
                centerY = consts::worldWidth/2 - height/2 - safetyMargin;
            }

            Entity wall = createWall(ctx, centerX, centerY, width, height);
            ctx.data().walls[i] = wall;
        }
    }

    // Place random number of ants in the world using grid-based distribution
    static void placeAnts(Engine &ctx, CountT numAnts)
    {
        if (numAnts == 0) {
            ctx.singleton<NumAnts>().count = 0;
            return;
        }

        // Account for the ant radius in our calculations to keep ants in bounds
        float worldHalfWidth = consts::worldWidth / 2.f - consts::antRadius * 2.f;
        float worldSize = worldHalfWidth * 2;

        // Create a grid for even distribution of ants
        // Determine grid dimensions based on number of ants
        int gridSize = std::ceil(sqrtf((float)numAnts)) + 1; // +1 for margin
        float cellSize = worldSize / gridSize;
        
        // Get goal and macguffin positions to avoid placing ants too close
        Vector3 goalPos = ctx.get<Position>(ctx.data().goal);
        Vector3 macguffinPos = ctx.get<Position>(ctx.data().macguffin);
        float minDistanceToObjects = consts::antRadius * 10.0f; // Keep ants away from key objects
        
        // Place ants in a grid pattern with small random offsets
        for (CountT i = 0; i < numAnts; ++i) {
            Entity ant = ctx.makeRenderableEntity<Ant>();
            
            // Calculate grid position for this ant
            int gridX = i % gridSize;
            int gridY = i / gridSize;
            
            // Calculate base position in world coordinates
            float baseX = -worldHalfWidth + gridX * cellSize + cellSize/2;
            float baseY = -worldHalfWidth + gridY * cellSize + cellSize/2;
            
            // Add small random offset within the cell
            float offsetRange = cellSize * 0.4f; // Keep within 40% of cell size to avoid overlap
            float x = baseX + randInRangeCentered(ctx, offsetRange);
            float y = baseY + randInRangeCentered(ctx, offsetRange);
            
            // Ensure position is within world bounds
            x = std::max(std::min(x, worldHalfWidth), -worldHalfWidth);
            y = std::max(std::min(y, worldHalfWidth), -worldHalfWidth);
            
            // Check distance from goal and macguffin
            Vector3 antPos(x, y, 0);
            float distToGoal = (antPos - goalPos).length2D();
            float distToMacguffin = (antPos - macguffinPos).length2D();
            
            // If too close to goal or macguffin, adjust position
            if (distToGoal < minDistanceToObjects || distToMacguffin < minDistanceToObjects) {
                // Move the ant away from the object it's too close to
                Vector3 awayDir;
                if (distToGoal < distToMacguffin) {
                    awayDir = antPos - goalPos;
                } else {
                    awayDir = antPos - macguffinPos;
                }
                
                // Normalize and move away
                if (awayDir.length2D() > 0.001f) {
                    awayDir = awayDir.normalized2D() * minDistanceToObjects;
                    x = (distToGoal < distToMacguffin ? goalPos.x : macguffinPos.x) + awayDir.x;
                    y = (distToGoal < distToMacguffin ? goalPos.y : macguffinPos.y) + awayDir.y;
                    
                    // Ensure still within bounds
                    x = std::max(std::min(x, worldHalfWidth), -worldHalfWidth);
                    y = std::max(std::min(y, worldHalfWidth), -worldHalfWidth);
                }
            }
            
            // Random rotation for the ant
            float angle = ctx.data().rng.sampleUniform() * 2.f * math::pi;
            Quat rot = Quat::angleAxis(angle, math::up);

            // Setup the ant entity
            setupRigidBodyEntity(
                ctx,
                ant,
                Vector3{x, y, consts::antRadius},
                rot,
                SimObject::Ant,
                EntityType::Ant);

            // Create a render view for a few ants, not for all (performance reasons)
            if (ctx.data().enableRender) {
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

            registerRigidBodyEntity(ctx, ant, SimObject::Ant);

            ctx.data().ants[i] = ant;
        }
        
        ctx.singleton<NumAnts>().count = numAnts;
    }

    // Generate the hive simulation world
    void generateWorld(Engine &ctx)
    {
        resetPersistentEntities(ctx);

        // Create goal first to establish a target location
        placeGoal(ctx);

        // Then place macguffin
        placeMacguffin(ctx, ctx.get<Position>(ctx.data().goal));

        // Place movable objects - using the random count from Sim
        placeMovableObjects(ctx, ctx.data().numMovableObjects);

        // Place additional walls - using the random count from Sim
        placeWalls(ctx, ctx.data().numWalls);

        // Place ants - using the random count from Sim
        placeAnts(ctx, ctx.singleton<NumAnts>().count);
    }

}
