#include "level_gen.hpp"

#include <algorithm>
#include <vector>

namespace madEscape
{

    using namespace madrona;
    using namespace madrona::math;
    using namespace madrona::phys;

    // Helper functions for random number generation

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
                -consts::worldLength / 2.f,
                0,
            },
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::PhysicsEntity,
            ResponseType::Static,
            Diag3x3{
                (consts::worldWidth + consts::borderWidth) / consts::borderMeshX, // add borderwidth for no jagged corners
                consts::borderWidth / consts::borderMeshY,
                consts::borderHeight / consts::borderMeshHeight,
            });

        // Right wall
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
            EntityType::PhysicsEntity,
            ResponseType::Static,
            Diag3x3{
                consts::borderWidth / consts::borderMeshX,
                (consts::worldLength + consts::borderWidth) / consts::borderMeshY, // add borderwidth for no jagged corners
                consts::borderHeight / consts::borderMeshHeight,
            });

        // Top wall
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
            EntityType::PhysicsEntity,
            ResponseType::Static,
            Diag3x3{
                (consts::worldWidth + consts::borderWidth) / consts::borderMeshX, // add borderwidth for no jagged corners
                consts::borderWidth / consts::borderMeshY,
                consts::borderHeight / consts::borderMeshHeight,
            });

        // Left wall
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
            EntityType::PhysicsEntity,
            ResponseType::Static,
            Diag3x3{
                consts::borderWidth / consts::borderMeshX,
                (consts::worldLength + consts::borderWidth) / consts::borderMeshY, // add borderwidth for no jagged corners
                consts::borderHeight / consts::borderMeshHeight,
            });

        for (CountT i = 0; i < consts::maxAnts; i++)
        {
            Entity ant = ctx.makeRenderableEntity<Ant>();
            if (ctx.data().enableRender) {
                render::RenderingSystem::attachEntityToView(ctx,
                                                        ant,
                                                        100.f, 0.001f,
                                                        0.5f * math::up);
            }

            // rigidbody including: pos, rot, scale, obj_id, vel, response_type, force, torque, entity_type
            setupRigidBodyEntity(
                ctx,
                ant,
                Vector3{0, 0, 0}, // overwritten on reset
                Quat::angleAxis(placement.angle, math::up),
                SimObject::Ant,
                EntityType::Ant,
                ResponseType::Dynamic,
                Diag3x3 {
                    consts::antSize / consts::antMeshSize,
                    consts::antSize / consts::antMeshSize,
                    consts::antSize / consts::antMeshSize}
            );
            
            ctx.data().ants[i] = ant;
        }
    }

    // Persistent entities need to be re-registered with the broadphase system.
    void resetPersistentEntities(Engine &ctx, AllPlacements &placements)
    {
        // Register the floor
        registerRigidBodyEntity(ctx, ctx.data().floorPlane, SimObject::Plane);

        // Register the borders
        for (CountT i = 0; i < 4; i++)
        {
            registerRigidBodyEntity(ctx, ctx.data().borders[i], SimObject::Plane);
        }

        for (CountT i = 0; i < consts::max::Ants; i++) {
            Entity ant = ctx.data().ants[i];
            AntPlacement placement = placements.ants[i];
            ctx.get<Position>(ant) = Vector3{placement.x, placement.y, 0.0f};
            ctx.get<GrabState>(ant).constraintEntity = Entity::none();
            // no need to init Lidar or Observation; they're set during simulation
            ctx.get<Action>(ant) = Action {
                .moveAmount = 0,
                .moveAngle = 0,
                .rotate = consts::numTurnBuckets / 2,
                .grab = 0,
            };
            registerRigidBodyEntity(ctx, ant, SimObject::Ant);
        }
    }


    // Enum to identify which border/wall
    enum class Border {
        Top = 0,
        Right = 1,
        Bottom = 2,
        Left = 3
    };

    // Structure to hold wall placement information
    struct WallPlacement {
        float x;
        float y;
        float width;
        float height;
    };

    struct MovableObjectPlacement {
        float x;
        float y;
        float scale;
    };

    struct AntPlacement {
        float x;
        float y;
        float angle;
    };
    
    struct MacguffinPlacement {
        float x;
        float y;
        Border wall; // Which wall/border it's placed along
        
        Vector3 toVector3() const {
            return Vector3{x, y, 0.0f};
        }
    };

    struct GoalPlacement {
        float x;
        float y;
        Border wall; // Which wall/border it's placed along
        
        Vector3 toVector3() const {
            return Vector3{x, y, 0.0f};
        }
    };

    struct AllPlacements {
        MacguffinPlacement macguffin;
        GoalPlacement goal;
        std::vector<WallPlacement> walls;
        std::vector<MovableObjectPlacement> movableObjects;
        std::vector<AntPlacement> ants;
    };

    static Entity createLevelState(Engine &ctx)
    {
        Entity levelState = ctx.makeEntity<LevelState>();
        ctx.get<Reward>(levelState).v = 0.0f;
        ctx.get<RewardHelperVars>(levelState).prev_dist = -1.0f;
        ctx.get<RewardHelperVars>(levelState).original_dist = -1.0f;
        ctx.get<HiveDone>(levelState).v = 0; // 0 = not done
        ctx.get<StepsRemaining>(levelState).t = consts::episodeLen;

        ctx.data().levelState = levelState;
        return levelState;
    }


    // Creates the macguffin (object that ants need to move)
    static Entity createMacguffin(Engine &ctx, MacguffinPlacement placement)
    {
        Entity macguffin = ctx.makeRenderableEntity<Macguffin>();
        setupRigidBodyEntity(
            ctx,
            macguffin,
            placement.toVector3(),
            Quat{1, 0, 0, 0},
            SimObject::Macguffin,
            EntityType::Macguffin,
            ResponseType::Dynamic,
            Diag3x3{
                consts::macguffinSize / consts::macguffinMeshSize,
                consts::macguffinSize / consts::macguffinMeshSize,
                consts::macguffinSize / consts::macguffinMeshSize});
        registerRigidBodyEntity(ctx, macguffin, SimObject::Macguffin);
        ctx.data().macguffin = macguffin;
        return macguffin;
    }

    // Creates the goal (target location for the macguffin)
    static Entity createGoal(Engine &ctx, GoalPlacement placement)
    {
        // Goal is non-physical, just a marker
        Entity goal = ctx.makeRenderableEntity<Goal>();

        ctx.get<Position>(goal) = placement.toVector3();
        ctx.get<Rotation>(goal) = Quat{1, 0, 0, 0};
        ctx.get<Scale>(goal) = Diag3x3{
            consts::goalSize / consts::goalMeshSize,
            consts::goalSize / consts::goalMeshSize,
            0.01f};
        ctx.get<ObjectID>(goal) = ObjectID{(int32_t)SimObject::Goal};
        ctx.get<EntityType>(goal) = EntityType::Goal;
        ctx.data().goal = goal;
        return goal;
    }

    // Creates a movable obstacle object
    static Entity createMovableObject(Engine &ctx, struct MovableObjectPlacement placement, CountT index)
    {
        Entity obj = ctx.makeRenderableEntity<MovableObject>();
        setupRigidBodyEntity(
            ctx,
            obj,
            Vector3{placement.x, placement.y, 0},
            Quat{1, 0, 0, 0},
            SimObject::MovableObject,
            EntityType::MovableObject,
            ResponseType::Dynamic,
            Diag3x3{
                consts::movableObjectSize * placement.scale / consts::movableObjectMeshSize,
                consts::movableObjectSize * placement.scale / consts::movableObjectMeshSize,
                consts::movableObjectSize * placement.scale / consts::movableObjectMeshSize});
        registerRigidBodyEntity(ctx, obj, SimObject::MovableObject);
        ctx.data().movableObjects[index] = obj;
        return obj;
    }

    // Creates an additional wall obstacle
    static Entity createWall(Engine &ctx, const WallPlacement &placement, CountT index)
    {
        Entity wall = ctx.makeRenderableEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            wall,
            Vector3{placement.x, placement.y, 0},
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::PhysicsEntity,
            ResponseType::Static,
            Diag3x3{
                placement.width / consts::wallMeshX,
                placement.height / consts::wallMeshY,
                consts::wallHeight / consts::wallMeshHeight});
        registerRigidBodyEntity(ctx, wall, SimObject::Wall);
        ctx.data().walls[index] = wall;
        return wall;
    }
    

    // Determine a suitable position for the macguffin
    static MacguffinPlacement determineMacguffinPlacement(Engine &ctx) {
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
        switch (border) {
            case Border::Top: // Top border
                x = randBetween(ctx, minX, maxX);
                y = maxY - offset; // 0-3 units from the border
                break;
            case Border::Right: // Right border
                x = maxX - offset; // 0-3 units from the border
                y = randBetween(ctx, minY, maxY);
                break;
            case Border::Bottom: // Bottom border
                x = randBetween(ctx, minX, maxX);
                y = minY + offset; // 0-3 units from the border
                break;
            case Border::Left: // Left border
                x = minX + offset; // 0-3 units from the border
                y = randBetween(ctx, minY, maxY);
                break;
        }
        
        return MacguffinPlacement{x, y, border};
    }

    // Determine a suitable position for the goal along the perimeter of the room
    static GoalPlacement determineGoalPlacement(Engine &ctx, const MacguffinPlacement &macguffinPlacement) {
        // Place goal on the opposite wall
        Border goalWall = static_cast<Border>((static_cast<int>(macguffinPlacement.wall) + 2) % 4);
        
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
        switch (goalWall) {
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
        
        return GoalPlacement{x, y, goalWall};
    }

    static std::vector<WallPlacement> determineWallPlacements(Engine &ctx, const MacguffinPlacement &macguffinPlacement, const GoalPlacement &goalPlacement) {
        std::vector<WallPlacement> wallPlacements;
        
        // Return empty vector if no walls needed
        if (ctx.data().numWalls <= 0) {
            return wallPlacements;
        }
        
        // Room boundaries
        float minX = -consts::worldWidth / 2.0f + consts::borderWidth / 2.0f;
        float maxX = consts::worldWidth / 2.0f - consts::borderWidth / 2.0f;
        float minY = -consts::worldLength / 2.0f + consts::borderWidth / 2.0f;
        float maxY = consts::worldLength / 2.0f - consts::borderWidth / 2.0f;
        
        // Try up to the maximum number of attempts for wall placement
        int attempt = 0;
        
        while (wallPlacements.size() < ctx.data().numWalls && attempt < consts::maxWallPlacementAttempts) {
            attempt++;
            
            // Decide if wall will be horizontal or vertical
            bool isHorizontal = randBetween(ctx, 0.0f, 1.0f) < 0.5f;
            
            // Wall size
            float wallLength = randBetween(ctx, consts::minWallLength, consts::maxWallLength);
            float wallWidth = consts::borderWidth;
            
            // Determine wall position
            float x = 0.0f;
            float y = 0.0f;
            float width = 0.0f;
            float height = 0.0f;
            
            if (isHorizontal) {
                // Horizontal wall
                width = wallLength;
                height = wallWidth;
                
                // Random position within room bounds
                x = randBetween(ctx, minX + width/2, maxX - width/2);
                y = randBetween(ctx, minY + height/2, maxY - height/2);
            } else {
                // Vertical wall
                width = wallWidth;
                height = wallLength;
                
                // Random position within room bounds
                x = randBetween(ctx, minX + width/2, maxX - width/2);
                y = randBetween(ctx, minY + height/2, maxY - height/2);
            }
            
            bool overlapsWithMacguffin =
            (std::abs(x - macguffinPlacement.x) < (width / 2.0f + consts::macguffinSize / 2.0f) + consts::wallMacguffinBuffer) &&
            (std::abs(y - macguffinPlacement.y) < (height / 2.0f + consts::macguffinSize / 2.0f) + consts::wallMacguffinBuffer);

            
            if (overlapsWithMacguffin) {
                continue; // Skip this attempt
            }
            
            bool overlapsWithGoal =
            (std::abs(x - goalPlacement.x) < (width / 2.0f + consts::goalSize / 2.0f) + consts::wallGoalBuffer) &&
            (std::abs(y - goalPlacement.y) < (height / 2.0f + consts::goalSize / 2.0f) + consts::wallGoalBuffer);
            
            if (overlapsWithGoal) {
                continue; // Skip this attempt
            }
            
            // Check if wall overlaps with existing walls
            bool overlapsWithWall = false;
            for (const auto &existingWall : wallPlacements) {
                // Simple overlap check (approximate)
                if (std::abs(x - existingWall.x) < (width + existingWall.width) / 2 + consts::wallWallBuffer&&
                    std::abs(y - existingWall.y) < (height + existingWall.height) / 2 + consts::wallWallBuffer) {
                    overlapsWithWall = true;
                    break;
                }
            }

            if (overlapsWithWall) {
                break;
            }
            
            WallPlacement wall;
            wall.x = x;
            wall.y = y;
            wall.width = width;
            wall.height = height;
            wallPlacements.push_back(wall);
            
        }
        // record the actual number of walls
        ctx.data().numWalls = wallPlacements.size();
        return wallPlacements;
    }

    static std::vector<MovableObjectPlacement> determineMovableObjectPlacements(
    Engine &ctx, 
    const MacguffinPlacement &macguffinPlacement, 
    const GoalPlacement &_, 
    const std::vector<WallPlacement> &wallPlacements) {
    
        std::vector<MovableObjectPlacement> objectPlacements;
        
        // Return empty vector if no movable objects needed
        if (ctx.data().numMovableObjects <= 0) {
            return objectPlacements;
        }

        // Define world edges (absolute edges of the simulation area)
        const float worldEdgeMinX = -consts::worldWidth / 2.0f;
        const float worldEdgeMaxX =  consts::worldWidth / 2.0f;
        const float worldEdgeMinY = -consts::worldLength / 2.0f;
        const float worldEdgeMaxY =  consts::worldLength / 2.0f;

        // Define placement boundaries (inside the border walls)
        const float placementBoundMinX = worldEdgeMinX + consts::borderWidth / 2.0f;
        const float placementBoundMaxX = worldEdgeMaxX - consts::borderWidth / 2.0f;
        const float placementBoundMinY = worldEdgeMinY + consts::borderWidth / 2.0f;
        const float placementBoundMaxY = worldEdgeMaxY - consts::borderWidth / 2.0f;
        
        int attempt = 0;
        // const int maxMovableObjectPlacementAttempts = 200; // Assuming this is in consts

        while (objectPlacements.size() < static_cast<size_t>(ctx.data().numMovableObjects) && attempt < consts::maxMovableObjectPlacementAttempts) {
            attempt++;
            
            // Random scale for the movable object
            float scale = randBetween(ctx, consts::movableObjectMinScale, consts::movableObjectMaxScale);
            
            // Current movable object's half-extent (assuming consts::movableObjectSize is base half-width/radius)
            float currentObjectHalfExtent = consts::movableObjectSize * scale / 2.0f;
            float macguffinHalfExtent = consts::macguffinSize / 2.0f;
            
            // Determine valid range for placing the center of the object
            float placeableMinX = placementBoundMinX + currentObjectHalfExtent;
            float placeableMaxX = placementBoundMaxX - currentObjectHalfExtent;
            float placeableMinY = placementBoundMinY + currentObjectHalfExtent;
            float placeableMaxY = placementBoundMaxY - currentObjectHalfExtent;

            // If the placeable area is invalid (e.g., object too large for the space), skip or break.
            if (placeableMinX >= placeableMaxX || placeableMinY >= placeableMaxY) {
                // This might happen if world is too small or objects too large relative to consts::borderWidth
                // Or if numMovableObjects is very high and remaining space is fragmented.
                // Depending on desired behavior, could log an error or simply stop trying.
                break; 
            }
            
            // Random position for the center of the movable object
            float x = randBetween(ctx, placeableMinX, placeableMaxX);
            float y = randBetween(ctx, placeableMinY, placeableMaxY);
            
            // --- Overlap Checks using AABB ---

            // 1. Check overlap with Macguffin
            bool overlapsWithMacguffin =
                (std::abs(x - macguffinPlacement.x) < (currentObjectHalfExtent + macguffinHalfExtent + consts::movableObjectMacguffinBuffer)) &&
                (std::abs(y - macguffinPlacement.y) < (currentObjectHalfExtent + macguffinHalfExtent + consts::movableObjectMacguffinBuffer));
            
            if (overlapsWithMacguffin) {
                continue; // Skip this attempt, try a new placement
            }

            // its fine to overlap with goal
            
            // 3. Check overlap with Walls
            bool objectOverlapsWithWall = false;
            for (const auto &wall : wallPlacements) {
                float wallHalfWidth = wall.width / 2.0f;
                float wallHalfHeight = wall.height / 2.0f;
                
                if ((std::abs(x - wall.x) < (currentObjectHalfExtent + wallHalfWidth + consts::movableObjectWallBuffer)) &&
                    (std::abs(y - wall.y) < (currentObjectHalfExtent + wallHalfHeight + consts::movableObjectWallBuffer))) {
                    objectOverlapsWithWall = true;
                    break; // Found an overlap with a wall
                }
            }
            if (objectOverlapsWithWall) {
                continue; // Skip this attempt
            }
            
            // 4. Check overlap with existing Movable Objects
            bool overlapsWithExistingObject = false;
            for (const auto &existingObject : objectPlacements) {
                float existingObjectHalfExtent = consts::movableObjectSize * existingObject.scale;
                
                if ((std::abs(x - existingObject.x) < (currentObjectHalfExtent + existingObjectHalfExtent + consts::movableObjectObjectBuffer)) &&
                    (std::abs(y - existingObject.y) < (currentObjectHalfExtent + existingObjectHalfExtent + consts::movableObjectObjectBuffer))) {
                    overlapsWithExistingObject = true;
                    break; // Found an overlap with another movable object
                }
            }
            if (overlapsWithExistingObject) {
                continue; // Skip this attempt
            }
            
            // If all checks passed, the placement is valid
            MovableObjectPlacement newObj;
            newObj.x = x;
            newObj.y = y;
            newObj.scale = scale;
            objectPlacements.push_back(newObj);
        }
        
        // Record the actual number of movable objects successfully placed
        ctx.data().numMovableObjects = objectPlacements.size();
        return objectPlacements;
    }
    
    
    static std::vector<AntPlacement> determineAntPlacements(Engine &ctx, const MacguffinPlacement &macguffinPlacement, const GoalPlacement &_, const std::vector<WallPlacement> &wallPlacements, const std::vector<MovableObjectPlacement> &movableObjectPlacements) {
        std::vector<AntPlacement> antPlacements;
        
        // Get how many ants we need to place
        int numAnts = ctx.singleton<NumAnts>().count;
        if (numAnts <= 0) {
            return antPlacements;
        }
        
        // Room boundaries
        float minX = -consts::worldWidth / 2.0f;
        float maxX = consts::worldWidth / 2.0f;
        float minY = -consts::worldLength / 2.0f;
        float maxY = consts::worldLength / 2.0f;
        
        float borderBuffer = consts::borderWidth / 2.0f + consts::antSize / 2.0f;

        
        // Adjusted room boundaries accounting for border walls
        float adjustedMinX = minX + borderBuffer;
        float adjustedMaxX = maxX - borderBuffer;
        float adjustedMinY = minY + borderBuffer;
        float adjustedMaxY = maxY - borderBuffer;
        
        // Try up to the maximum number of attempts per ant
        
        // Try to place each ant
        for (int antIndex = 0; antIndex < numAnts; antIndex++) {
            int attempts = 0;
            bool placedSuccessfully = false;
            
            while (!placedSuccessfully && attempts < consts::maxAntPlacementAttemptsPerAnt) {
                attempts++;
                
                // Random position within adjusted room bounds
                float x = randBetween(ctx, adjustedMinX, adjustedMaxX);
                float y = randBetween(ctx, adjustedMinY, adjustedMaxY);
                
                // Random angle (orientation)
                float angle = randBetween(ctx, 0.0f, 2.0f * 3.14159f);  // 0 to 2π radians
                
                // Check if ant overlaps with macguffin
                bool overlapsWithMacguffin = false;
                if (std::abs(x - macguffinPlacement.x) < (consts::antSize / 2.0f + consts::macguffinSize / 2.0f + consts::antMacguffinBuffer) &&
                    std::abs(y - macguffinPlacement.y) < (consts::antSize / 2.0f + consts::macguffinSize / 2.0f + consts::antMacguffinBuffer)) {
                    overlapsWithMacguffin = true;
                }
                
                if (overlapsWithMacguffin) {
                    continue; // Skip this attempt
                }

                // its's fine if they spawn on the goal; don't check that
                
                // Check if ant overlaps with walls
                bool overlapsWithWall = false;
                for (const auto &wall : wallPlacements) {
                    // Simple box-based distance check
                    if (std::abs(x - wall.x) < (consts::antSize / 2.0f + wall.width / 2.0f + consts::antWallBuffer) &&
                        std::abs(y - wall.y) < (consts::antSize / 2.0f + wall.height / 2.0f + consts::antWallBuffer)) {
                        overlapsWithWall = true;
                        break;
                    }
                }
                
                if (overlapsWithWall) {
                    continue; // Skip this attempt
                }
                
                // Check if ant overlaps with movable objects
                bool overlapsWithObject = false;
                for (const auto &obj : movableObjectPlacements) {
                    // Simple box-based distance check
                    if (std::abs(x - obj.x) < (consts::antSize / 2.0f + consts::movableObjectSize * obj.scale / 2.0f + consts::antWallBuffer) &&
                        std::abs(y - obj.y) < (consts::antSize / 2.0f + consts::movableObjectSize * obj.scale / 2.0f + consts::antWallBuffer)) {
                        overlapsWithObject = true;
                        break;
                    }
                }
                
                if (overlapsWithObject) {
                    continue; // Skip this attempt
                }
                
                // Check if ant overlaps with already placed ants
                bool overlapsWithAnt = false;
                for (const auto &existingAnt : antPlacements) {
                    float distToAnt = std::sqrt(
                        std::pow(x - existingAnt.x, 2) + 
                        std::pow(y - existingAnt.y, 2));
                    
                    if (distToAnt < consts::antSize + consts::antAntBuffer) {
                        overlapsWithAnt = true;
                        break;
                    }
                }
                
                if (overlapsWithAnt) {
                    continue;
                }
                
                // Ant position is valid, add it
                AntPlacement ant;
                ant.x = x;
                ant.y = y;
                ant.angle = angle;
                antPlacements.push_back(ant);
                placedSuccessfully = true;
            }
            
            // If we couldn't place this ant after all attempts, stop trying to place more
            if (!placedSuccessfully) {
                break;
            }
        }
        ctx.singleton<NumAnts>().count = antPlacements.size();
        return antPlacements;
    }


    // Generate the hive simulation world. called each episode.
    void generateWorld(Engine &ctx)
    {
        const auto &macguffinPlacement = determineMacguffinPlacement(ctx);
        const auto &goalPlacement = determineGoalPlacement(ctx, macguffinPlacement);
        const auto &wallPlacements = determineWallPlacements(ctx, macguffinPlacement, goalPlacement);
        const auto &movableObjectPlacements = determineMovableObjectPlacements(ctx, macguffinPlacement, goalPlacement, wallPlacements);
        const auto &antPlacements = determineAntPlacements(ctx, macguffinPlacement, goalPlacement, wallPlacements, movableObjectPlacements);
        const AllPlacements placements = {
            macguffinPlacement,
            goalPlacement,
            wallPlacements,
            movableObjectPlacements,
            antPlacements
        };
        resetPersistentEntities(ctx, placements);

        createLevelState(ctx);

        // Place macguffin first
        createMacguffin(ctx, macguffinPlacement);

        // goal
        createGoal(ctx, goalPlacement);

        // walls
        for (size_t i = 0; i < ctx.data().numWalls; i++) {
            createWall(ctx, wallPlacements[i], i);
        }

        // movable objects
        for (size_t i = 0; i < ctx.data().numMovableObjects; i++) {
            createMovableObject(ctx, movableObjectPlacements[i], i);
        }
    }

}
