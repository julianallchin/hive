#include "level_gen.hpp"

#include <algorithm>
#include <vector>
#include <cmath>

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

    // Determine a suitable position for the goal along the perimeter of the room
    static Vector3 determineGoalPos(Engine &ctx) {
        // Place goal along one of the walls, but not in a corner
        float wallChoice = ctx.data().rng.sampleUniform();
        float x, y;

        if (wallChoice < 0.25f) {
            // Bottom wall
            x = randInRangeCentered(ctx, consts::worldWidth * 0.5f);
            y = -consts::worldWidth / 2.f + consts::wallWidth * 1.5f; // Slightly inside the wall
        } else if (wallChoice < 0.5f) {
            // Right wall
            x = consts::worldWidth / 2.f - consts::wallWidth * 1.5f; // Slightly inside the wall
            y = randInRangeCentered(ctx, consts::worldWidth * 0.5f);
        } else if (wallChoice < 0.75f) {
            // Top wall
            x = randInRangeCentered(ctx, consts::worldWidth * 0.5f);
            y = consts::worldWidth / 2.f - consts::wallWidth * 1.5f; // Slightly inside the wall
        } else {
            // Left wall
            x = -consts::worldWidth / 2.f + consts::wallWidth * 1.5f; // Slightly inside the wall
            y = randInRangeCentered(ctx, consts::worldWidth * 0.5f);
        }

        return Vector3(x, y, 0.0f);
    }

    // Place the goal at a position determined by determineGoalPos
    static Entity placeGoal(Engine &ctx)
    {
        Vector3 goalPos = determineGoalPos(ctx);
        Entity goal = createGoal(ctx, goalPos.x, goalPos.y);
        return goal;
    }

    // Determine a suitable position for the macguffin based on goal position
    static Vector3 determineMacguffinPos(Engine &ctx, const Vector3 &goalPos) {
        // Place macguffin on opposite side, with some randomness
        float angle = atan2f(goalPos.y, goalPos.x) + math::pi +
                       randInRangeCentered(ctx, math::pi / 4.0f);
        float dist = consts::worldWidth * 0.3f;

        float x = cosf(angle) * dist;
        float y = sinf(angle) * dist;
        
        // Ensure within world bounds
        float safetyMargin = consts::macguffinRadius * 2.0f;
        float worldEdge = consts::worldWidth / 2.0f - safetyMargin;
        x = std::max(std::min(x, worldEdge), -worldEdge);
        y = std::max(std::min(y, worldEdge), -worldEdge);
        
        return Vector3(x, y, consts::macguffinRadius);
    }
    
    // Place the macguffin using the position determined by determineMacguffinPos
    static Entity placeMacguffin(Engine &ctx, Vector3 goalPos)
    {
        Vector3 macguffinPos = determineMacguffinPos(ctx, goalPos);
        return createMacguffin(ctx, macguffinPos.x, macguffinPos.y);
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
    
    // Structure to hold wall placement information
    struct WallPlacement {
        float x;
        float y;
        float width;
        float height;
    };
    
    // Structure to hold positions for all entities
    struct LevelPositions {
        Vector3 macguffinPos;
        Vector3 goalPos;
        std::vector<Vector3> movableObjectPositions;
        std::vector<WallPlacement> wallPlacements;
        std::vector<Vector3> antPositions;
    };
    
    // Generate deterministic failsafe position for an entity type based on its index
    static Vector3 generateDeterministicFailsafePosition(Engine &ctx, EntityType entityType, int index) {
        // Define fixed parameters for the deterministic pattern
        const float worldHalfWidth = consts::worldWidth / 2.f * 0.7f; // Use 70% of world size for safety
        const float baseZ = 0.0f;
        
        // Define starting angles for different entity types to keep them separated
        float baseAngle = 0.0f;
        float radiusFactor = 0.7f; // Default radius factor
        
        switch (entityType) {
            case EntityType::Macguffin:
                baseAngle = 0.0f;
                radiusFactor = 0.8f; // Macguffin closer to edge
                break;
            case EntityType::Goal:
                baseAngle = math::pi; // Opposite of macguffin
                radiusFactor = 0.8f;
                break;
            case EntityType::MovableObject:
                baseAngle = math::pi / 4.0f;
                radiusFactor = 0.5f; // Movable objects more in the middle
                break;
            case EntityType::Wall:
                baseAngle = 3.0f * math::pi / 4.0f;
                radiusFactor = 0.6f;
                break;
            case EntityType::Ant:
                baseAngle = 5.0f * math::pi / 4.0f;
                radiusFactor = 0.4f; // Ants closer to the center
                break;
            default:
                baseAngle = 0.0f;
                break;
        }
        
        // Create a spiral pattern based on index
        // Use golden angle to create an even distribution
        const float goldenAngle = math::pi * (3.0f - sqrtf(5.0f)); // ~2.4 radians
        float angle = baseAngle + index * goldenAngle;
        
        // Radius increases with index for spacing
        float radius = worldHalfWidth * radiusFactor * (0.2f + 0.1f * sqrtf((float)index));
        if (radius > worldHalfWidth * radiusFactor) {
            radius = worldHalfWidth * radiusFactor;
        }
        
        // Calculate position using polar coordinates
        float x = radius * cosf(angle);
        float y = radius * sinf(angle);
        
        return Vector3(x, y, baseZ);
    }

    // Determine all placement positions for entities in the level together
    static LevelPositions determinePlacementPositions(Engine &ctx, CountT numMovableObjects, CountT numWalls, CountT numAnts) {
        LevelPositions positions;
        
        // Calculate world bounds
        float worldHalfWidth = consts::worldWidth / 2.f - consts::movableObjectRadius;
        float minDistance = consts::movableObjectRadius * 4.0f; // Minimum distance between objects
        
        // 1. First, determine macguffin position - it should be near a wall
        // Try to place near a wall, with failsafe available
        bool foundValidMacguffinPos = false;
        
        // Try to find a position near the wall
        for (int attempt = 0; attempt < 20 && !foundValidMacguffinPos; attempt++) {
            // Select a random wall (top, right, bottom, left)
            int wallIdx = ctx.data().rng.sampleUniformInt(0, 3);
            float wallX = 0.0f, wallY = 0.0f;
            
            switch (wallIdx) {
                case 0: // Top wall
                    wallX = randInRange(ctx, -worldHalfWidth, worldHalfWidth);
                    wallY = worldHalfWidth;
                    break;
                case 1: // Right wall
                    wallX = worldHalfWidth;
                    wallY = randInRange(ctx, -worldHalfWidth, worldHalfWidth);
                    break;
                case 2: // Bottom wall
                    wallX = randInRange(ctx, -worldHalfWidth, worldHalfWidth);
                    wallY = -worldHalfWidth;
                    break;
                case 3: // Left wall
                    wallX = -worldHalfWidth;
                    wallY = randInRange(ctx, -worldHalfWidth, worldHalfWidth);
                    break;
            }
            
            // Move slightly inward from the wall
            const float inwardOffset = consts::movableObjectRadius * 2.0f;
            if (wallIdx == 0) wallY -= inwardOffset;
            else if (wallIdx == 1) wallX -= inwardOffset;
            else if (wallIdx == 2) wallY += inwardOffset;
            else if (wallIdx == 3) wallX += inwardOffset;
            
            positions.macguffinPos = Vector3(wallX, wallY, 0.0f);
            foundValidMacguffinPos = true; // We can assume this is valid since it's our first placement
        }
        
        // If we couldn't find a valid position, use deterministic failsafe
        if (!foundValidMacguffinPos) {
            positions.macguffinPos = generateDeterministicFailsafePosition(ctx, EntityType::Macguffin, 0);
            if (ctx.data().printDebugOutput) {
                printf("Using failsafe position for macguffin: (%f, %f)\n", 
                       positions.macguffinPos.x, positions.macguffinPos.y);
            }
        }
        positions.macguffinPos.z = consts::macguffinRadius;
        
        // 2. Place goal away from macguffin
        float minGoalDistance = consts::worldWidth * 0.4f; // Minimum distance from macguffin
        int attempts = 0;
        const int maxAttempts = 50;
        
        do {
            // Choose a random position
            positions.goalPos.x = randInRangeCentered(ctx, consts::worldWidth * 0.7f);
            positions.goalPos.y = randInRangeCentered(ctx, consts::worldWidth * 0.7f);
            float angle = atan2f(positions.macguffinPos.y, positions.macguffinPos.x) + math::pi;
            float distance = randInRange(ctx, worldHalfWidth * 0.5f, worldHalfWidth * 0.9f);
            
            float goalX = distance * cosf(angle);
            float goalY = distance * sinf(angle);
            
            Vector3 potentialGoalPos = Vector3(goalX, goalY, 0.0f);
            
            // Goal needs to be far enough from macguffin but doesn't need to avoid other objects
            if ((potentialGoalPos - positions.macguffinPos).length2D() > worldHalfWidth * 0.6f) {
                positions.goalPos = potentialGoalPos;
                foundValidGoalPos = true;
            }
        }
        
        // If we couldn't find a valid position, use deterministic failsafe
        if (!foundValidGoalPos) {
            positions.goalPos = generateDeterministicFailsafePosition(ctx, EntityType::Goal, 0);
            if (ctx.data().printDebugOutput) {
                printf("Using failsafe position for goal: (%f, %f)\n", 
                       positions.goalPos.x, positions.goalPos.y);
            }
        }
        
        positions.goalPos.z = consts::goalRadius;
        
        // Make sure goal is within bounds
        float safetyMargin = 1.0f; // Goal doesn't have collision but still needs to be inside visually
        float goalEdge = worldHalfWidth - safetyMargin;
        positions.goalPos.x = std::max(std::min(positions.goalPos.x, goalEdge), -goalEdge);
        positions.goalPos.y = std::max(std::min(positions.goalPos.y, goalEdge), -goalEdge);
        
        // 3. Determine positions for movable objects
        positions.movableObjectPositions.reserve(numMovableObjects);
        if (numMovableObjects > 0) {
            // Create a grid for even distribution
            int gridSize = (int)std::ceil(sqrtf((float)numMovableObjects)) + 1;
            float cellSize = consts::worldWidth * 0.8f / gridSize;
            float startOffset = -consts::worldWidth * 0.4f;
            
            // Minimum distance from important objects
            float minObjDistance = consts::worldWidth * 0.12f;
            
            for (CountT i = 0; i < numMovableObjects; i++) {
                // Calculate grid position
                int gridX = i % gridSize;
                int gridY = i / gridSize;
                
                // Base position in grid
                float baseX = startOffset + gridX * cellSize + cellSize * 0.5f;
                float baseY = startOffset + gridY * cellSize + cellSize * 0.5f;
                
                // Add some randomness within the cell
                float x = baseX + randInRangeCentered(ctx, cellSize * 0.4f);
                float y = baseY + randInRangeCentered(ctx, cellSize * 0.4f);
                Vector3 objPos(x, y, 0.0f);
                
                // Check if far enough from goal and macguffin
                bool validPosition = isWithinWorldBounds(x, y, consts::movableObjectRadius) &&
                                    isFarEnoughFrom(objPos, positions.goalPos, minObjDistance) &&
                                    isFarEnoughFrom(objPos, positions.macguffinPos, minObjDistance);
                
                // If not valid after grid placement, try with deterministic attempts
                if (!validPosition) {
                    // Try a few systematic positions with increasing angles around a circle
                    for (int attempt = 0; attempt < 8 && !validPosition; attempt++) {
                        // Use evenly distributed angles around a circle at different distances
                        float angle = math::pi / 4.0f * attempt;
                        float distance = worldHalfWidth * (0.4f + 0.1f * (attempt % 3));
                        
                        x = distance * cosf(angle);
                        y = distance * sinf(angle);
                        objPos = Vector3(x, y, 0.0f);
                        
                        validPosition = isWithinWorldBounds(x, y, consts::movableObjectRadius) &&
                                        isFarEnoughFrom(objPos, positions.goalPos, minObjDistance) &&
                                        isFarEnoughFrom(objPos, positions.macguffinPos, minObjDistance);
                    }
                    
                    // If still not valid, use deterministic failsafe
                    if (!validPosition) {
                        objPos = generateDeterministicFailsafePosition(ctx, EntityType::MovableObject, i);
                        if (ctx.data().printDebugOutput) {
                            printf("Using failsafe position for movable object %d: (%f, %f)\n", 
                                   i, objPos.x, objPos.y);
                        }
                    }
                }
                
                // Ensure within bounds
                float objSafetyMargin = consts::movableObjectRadius * 2.0f;
                float objEdge = worldHalfWidth - objSafetyMargin;
                x = std::max(std::min(x, objEdge), -objEdge);
                y = std::max(std::min(y, objEdge), -objEdge);
                
                // Create the position
                Vector3 finalObjPos(x, y, consts::movableObjectRadius);
                
                // Check if too close to goal or macguffin
                if (!isFarEnoughFrom(finalObjPos, positions.macguffinPos, minObjDistance) || 
                    !isFarEnoughFrom(finalObjPos, positions.goalPos, minObjDistance)) {
                    // Try to shift away from the closest object
                    Vector3 awayDir;
                    float distToMacguffin = (finalObjPos - positions.macguffinPos).length2D();
                    float distToGoal = (finalObjPos - positions.goalPos).length2D();
                    
                    if (distToMacguffin < distToGoal) {
                        awayDir = finalObjPos - positions.macguffinPos;
                    } else {
                        awayDir = finalObjPos - positions.goalPos;
                    }
                    
                    if (awayDir.length2D() > 0.001f) {
                        awayDir = awayDir.normalized2D() * minObjDistance * 0.6f;
                        x += awayDir.x;
                        y += awayDir.y;
                        
                        // Re-ensure within bounds
                        x = std::max(std::min(x, objEdge), -objEdge);
                        y = std::max(std::min(y, objEdge), -objEdge);
                        
                        objPos = Vector3(x, y, consts::movableObjectRadius);
                    }
                }
                
                positions.movableObjectPositions.push_back(objPos);
            }
        }
        
        // 4. Determine positions for walls
        positions.wallPlacements.reserve(numWalls);
        if (numWalls > 0) {
            // Calculate vector from macguffin to goal
            float dx = positions.goalPos.x - positions.macguffinPos.x;
            float dy = positions.goalPos.y - positions.macguffinPos.y;
            float pathAngle = atan2f(dy, dx);
            float pathLength = sqrtf(dx*dx + dy*dy);
            
            // Minimum distance from path
            float pathMinDistance = consts::worldWidth * 0.1f;
            
            for (CountT i = 0; i < numWalls; i++) {
                // Distribute along path
                float t = (i + 1.0f) / (numWalls + 1.0f); 
                
                // Use deterministic angle and offset instead of random
                float deterministic_value = (float)i / numWalls; // Value between 0 and almost 1.0 based on index
                float wallAngle = pathAngle + math::pi/2.0f + (deterministic_value - 0.5f) * math::pi/3.0f;
                float pathOffset = pathMinDistance + consts::worldWidth * 0.1f * (0.5f + deterministic_value * 0.5f);
                
                // Position along path
                float centerX = positions.macguffinPos.x + dx * t + cosf(wallAngle) * pathOffset;
                float centerY = positions.macguffinPos.y + dy * t + sinf(wallAngle) * pathOffset;
                
                // Check if position is valid (within world bounds)
                if (!isWithinWorldBounds(centerX, centerY, wallLength * 0.5f)) {
                    // If not valid, use deterministic failsafe
                    Vector3 failsafePos = generateDeterministicFailsafePosition(ctx, EntityType::Wall, i);
                    centerX = failsafePos.x;
                    centerY = failsafePos.y;
                    
                    if (ctx.data().printDebugOutput) {
                        printf("Using failsafe position for wall %d: (%f, %f)\n", (int)i, centerX, centerY);
                    }
                }
                
                // Wall dimensions
                float wallLength = consts::worldWidth * (0.1f + (i % 3) * 0.05f);
                float wallWidth = consts::wallWidth;
                
                // Determine orientation based on angle
                float width, height;
                if (fabsf(fmodf(wallAngle, math::pi)) < math::pi/4.0f || 
                    fabsf(fmodf(wallAngle, math::pi)) > 3.0f*math::pi/4.0f) {
                    // More horizontal
                    width = wallLength;
                    height = wallWidth;
                } else {
                    // More vertical
                    width = wallWidth;
                    height = wallLength;
                }
                
                // Ensure wall is within bounds
                float wallSafetyMargin = std::max(width, height) * 0.5f + wallThickness;
                float wallEdge = worldHalfWidth - wallSafetyMargin;
                centerX = std::max(std::min(centerX, wallEdge), -wallEdge);
                centerY = std::max(std::min(centerY, wallEdge), -wallEdge);
                
                // Add to placements
                WallPlacement wall;
                wall.x = centerX;
                wall.y = centerY;
                wall.width = width;
                wall.height = height;
                positions.wallPlacements.push_back(wall);
            }
        }
        
        // 5. Determine positions for ants
        positions.antPositions.reserve(numAnts);
        if (numAnts > 0) {
            // Create a larger grid for even distribution of ants
            int gridSize = (int)std::ceil(sqrtf((float)numAnts * 1.5f)); // More cells than ants for spacing
            float cellSize = consts::worldWidth * 0.8f / gridSize;
            float worldStart = -consts::worldWidth * 0.4f;
            float minDistanceToObjects = consts::worldWidth * 0.08f; // Smaller than for movable objects
            
            for (CountT i = 0; i < numAnts; i++) {
                // Calculate grid cell for this ant
                int gridX = i % gridSize;
                int gridY = i / gridSize;
                
                // Base position in the cell
                float baseX = worldStart + gridX * cellSize + cellSize * 0.5f;
                float baseY = worldStart + gridY * cellSize + cellSize * 0.5f;
                
                // Use deterministic offsets instead of randomization
                // This uses the ant index to create a predictable offset pattern within each cell
                float deterministicOffsetX = ((float)(i % 3) / 2.0f - 0.5f) * cellSize * 0.6f;
                float deterministicOffsetY = ((float)((i / 3) % 3) / 2.0f - 0.5f) * cellSize * 0.6f;
                float x = baseX + deterministicOffsetX;
                float y = baseY + deterministicOffsetY;
                
                // Ensure within bounds
                float antSafetyMargin = consts::antRadius * 2.0f;
                float antEdge = worldHalfWidth - antSafetyMargin;
                x = std::max(std::min(x, antEdge), -antEdge);
                y = std::max(std::min(y, antEdge), -antEdge);
                
                // Create position
                Vector3 antPos(x, y, consts::antRadius);
                
                // Check if too close to important objects
                bool validPosition = isWithinWorldBounds(x, y, consts::antRadius) &&
                                   isFarEnoughFrom(antPos, positions.macguffinPos, minDistanceToObjects) &&
                                   isFarEnoughFrom(antPos, positions.goalPos, minDistanceToObjects);
                
                // If not valid, use a deterministic failsafe position
                if (!validPosition) {
                    Vector3 failsafePos = generateDeterministicFailsafePosition(ctx, EntityType::Ant, i);
                    antPos = Vector3(failsafePos.x, failsafePos.y, consts::antRadius);
                    
                    if (ctx.data().printDebugOutput) {
                        printf("Using failsafe position for ant %d: (%f, %f)\n", 
                               (int)i, failsafePos.x, failsafePos.y);
                    }
                }
                
                positions.antPositions.push_back(antPos);
            }
        }
        
        return positions;
    }
    
    // Generate the hive simulation world using the unified positioning system
    void generateWorld(Engine &ctx)
    {
        resetPersistentEntities(ctx);
        
        // Determine all entity positions together
        CountT numMovableObjects = ctx.data().numMovableObjects;
        CountT numWalls = ctx.data().numWalls;
        CountT numAnts = ctx.singleton<NumAnts>().count;
        
        // Calculate all positions in one pass to ensure no overlaps
        LevelPositions positions = determinePlacementPositions(
            ctx, numMovableObjects, numWalls, numAnts);
        
        // Create entities using the calculated positions
        placeMacguffin(ctx, positions.macguffinPos);
        placeGoal(ctx, positions.goalPos);
        placeMovableObjects(ctx, positions.movableObjectPositions);
        placeWalls(ctx, positions.wallPlacements);
        placeAnts(ctx, positions.antPositions);
    }

    // Place the macguffin based on the provided position
    static Entity placeMacguffin(Engine &ctx, const Vector3 &position)
    {
        Entity macguffin = createMacguffin(ctx, position.x, position.y);
        ctx.data().macguffin = macguffin;
        return macguffin;
    }
    
    // Place the goal based on the provided position
    static Entity placeGoal(Engine &ctx, const Vector3 &position)
    {
        Entity goal = createGoal(ctx, position.x, position.y);
        ctx.data().goal = goal;
        return goal;
    }
    
    // Place movable objects at the provided positions
    static void placeMovableObjects(Engine &ctx, const std::vector<Vector3> &positions)
    {
        CountT numObjects = std::min(positions.size(), (size_t)consts::maxMovableObjects);
        
        if (numObjects == 0) return;
        
        // Create each object at its position
        for (CountT i = 0; i < numObjects; i++) {
            const Vector3 &pos = positions[i];
            Entity object = createMovableObject(ctx, pos.x, pos.y);
            
            // Store reference to the entity
            ctx.data().movableObjects[i] = object;
        }
        
        // Reset any unused slots
        for (CountT i = numObjects; i < consts::maxMovableObjects; i++) {
            ctx.data().movableObjects[i] = Entity::none();
        }
        
        // Store the actual number of objects created
        ctx.data().numMovableObjects = numObjects;
    }
    // Place walls at the provided positions
    static void placeWalls(Engine &ctx, const std::vector<WallPlacement> &wallPlacements)
    {
        CountT numWalls = wallPlacements.size();
        
        if (numWalls == 0) return;
        
        // Create each wall at its position with specified dimensions
        for (CountT i = 0; i < numWalls; i++) {
            const WallPlacement &wall = wallPlacements[i];
            Entity wallEntity = createWall(ctx, wall.x, wall.y, wall.width, wall.height);
            
            if (ctx.data().printDebugOutput) {
                printf("Created wall %d at (%f, %f) with dimensions %f x %f\n",
                       (int)i, wall.x, wall.y, wall.width, wall.height);
            }
        }
    }
    // Place ants at the provided positions
    static void placeAnts(Engine &ctx, const std::vector<Vector3> &antPositions)
    {
        CountT numAnts = antPositions.size();
        
        if (numAnts == 0) return;
        
        // Create each ant at its position
        for (CountT i = 0; i < numAnts; ++i) {
            const Vector3 &pos = antPositions[i];
            
            // Create ant entity
            Entity ant = ctx.makeRenderableEntity<Ant>();
            
            // Random rotation for the ant
            float angle = ctx.data().rng.sampleUniform() * 2.f * math::pi;
            Quat rot = Quat::angleAxis(angle, math::up);
            
            // Setup the ant entity
            setupRigidBodyEntity(
                ctx,
                ant,
                Vector3{pos.x, pos.y, consts::antRadius},
                rot,
                SimObject::Ant,
                EntityType::Ant);
                
            // Create a render view for few ants if rendering is enabled
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
            
            if (ctx.data().printDebugOutput) {
                printf("Created ant %d at (%f, %f)\n", (int)i, pos.x, pos.y);
            }
        }
        
        // Update the ant count
        ctx.singleton<NumAnts>().count = numAnts;
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
            throw std::runtime_error("Trying to place 0 ants! numAnts must be greater than 0");
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

            // Create a render view for few ants
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
