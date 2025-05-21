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
                -consts::worldWidth / 2.f + consts::borderWidth / 2.f,
                0,
            },
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::PhysicsEntity,
            ResponseType::Static,
            Diag3x3{
                consts::worldWidth,
                consts::borderWidth,
                2.f,
            });

        // Right wall
        ctx.data().borders[1] = ctx.makeRenderableEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            ctx.data().borders[1],
            Vector3{
                consts::worldWidth / 2.f - consts::borderWidth / 2.f,
                0,
                0,
            },
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::PhysicsEntity,
            ResponseType::Static,
            Diag3x3{
                consts::borderWidth,
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
                consts::worldWidth / 2.f - consts::borderWidth / 2.f,
                0,
            },
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::PhysicsEntity,
            ResponseType::Static,
            Diag3x3{
                consts::worldWidth,
                consts::borderWidth,
                2.f,
            });

        // Left wall
        ctx.data().borders[3] = ctx.makeRenderableEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            ctx.data().borders[3],
            Vector3{
                -consts::worldWidth / 2.f + consts::borderWidth / 2.f,
                0,
                0,
            },
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::PhysicsEntity,
            ResponseType::Static,
            Diag3x3{
                consts::borderWidth,
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
    
    // Structure to hold macguffin or goal placement information
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
                consts::macguffinRadius * 2,
                consts::macguffinRadius * 2,
                consts::macguffinRadius * 2});
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
            consts::goalRadius * 2,
            consts::goalRadius * 2,
            0.1f};
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
            Vector3{placement.x, placement.y, consts::movableObjectRadius * placement.scale},
            Quat{1, 0, 0, 0},
            SimObject::MovableObject,
            EntityType::MovableObject,
            ResponseType::Dynamic,
            Diag3x3{
                consts::movableObjectRadius * 2 * placement.scale,
                consts::movableObjectRadius * 2 * placement.scale,
                consts::movableObjectRadius * 2 * placement.scale});
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
                placement.width,
                placement.height,
                2.0f});
        registerRigidBodyEntity(ctx, wall, SimObject::Wall);
        ctx.data().walls[index] = wall;
        return wall;
    }
    

    Entity createAnt(Engine &ctx, const AntPlacement &placement, CountT index)
    {
        Entity ant = ctx.makeRenderableEntity<Ant>();
        setupRigidBodyEntity(
            ctx,
            ant,
            Vector3{placement.x, placement.y, consts::antRadius},
            Quat::angleAxis(placement.angle, math::up),
            SimObject::Ant,
            EntityType::Ant);
            
        // Create a render view for the ant if rendering is enabled
        if (ctx.data().enableRender) {
            render::RenderingSystem::attachEntityToView(ctx,
                                                       ant,
                                                       100.f, 0.001f,
                                                       0.5f * math::up);
        }

        // Set the scale of the ant
        ctx.get<Scale>(ant) = Diag3x3{
            consts::antRadius * 2,
            consts::antRadius * 2,
            consts::antRadius * 2};

        ctx.get<GrabState>(ant).constraintEntity = Entity::none();
        
        registerRigidBodyEntity(ctx, ant, SimObject::Ant);
        ctx.data().ants[index] = ant;
        return ant;
    }

    // Determine a suitable position for the macguffin
    static MacguffinPlacement determineMacguffinPlacement(Engine &ctx) {
        // Choose a border randomly
        Border border = static_cast<Border>(randIntBetween(ctx, 0, 3));
        
        // Buffer from the wall to ensure the macguffin doesn't clip into the border
        // (Add a small margin to macguffin radius for safety)
        float buffer = 2 * consts::macguffinRadius + consts::borderWidth;
        
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
        float buffer = 2 * consts::goalRadius + consts::borderWidth;
        
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
        float minX = -consts::worldWidth / 2.0f;
        float maxX = consts::worldWidth / 2.0f;
        float minY = -consts::worldLength / 2.0f;
        float maxY = consts::worldLength / 2.0f;
        
        // Buffer to keep objects from overlapping
        float macguffinBuffer = consts::macguffinRadius + 1.0f;
        float goalBuffer = consts::goalRadius + 1.0f;
        
        // Try up to the maximum number of attempts for wall placement
        int attempt = 0;
        
        while (wallPlacements.size() < ctx.data().numWalls && attempt < consts::maxWallPlacementAttempts) {
            attempt++;
            
            // Decide if wall will be horizontal or vertical
            bool isHorizontal = randBetween(ctx, 0.0f, 1.0f) < 0.5f;
            
            // Wall size
            float wallLength = randBetween(ctx, 5.0f, 15.0f);
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
            
            // Check if wall overlaps with macguffin
            float distToMacguffin = std::sqrt(
                std::pow(x - macguffinPlacement.x, 2) + 
                std::pow(y - macguffinPlacement.y, 2));
            
            if (distToMacguffin < macguffinBuffer + std::max(width, height) / 2) {
                continue; // Skip this attempt
            }
            
            // Check if wall overlaps with goal
            float distToGoal = std::sqrt(
                std::pow(x - goalPlacement.x, 2) + 
                std::pow(y - goalPlacement.y, 2));
            
            if (distToGoal < goalBuffer + std::max(width, height) / 2) {
                continue; // Skip this attempt
            }
            
            // Check if wall overlaps with existing walls
            bool overlapsWithWall = false;
            for (const auto &existingWall : wallPlacements) {
                // Simple overlap check (approximate)
                if (std::abs(x - existingWall.x) < (width + existingWall.width) / 2 &&
                    std::abs(y - existingWall.y) < (height + existingWall.height) / 2) {
                    overlapsWithWall = true;
                    break;
                }
            }
            
            if (!overlapsWithWall) {
                // Wall is valid, add it
                WallPlacement wall;
                wall.x = x;
                wall.y = y;
                wall.width = width;
                wall.height = height;
                wallPlacements.push_back(wall);
            }
        }
        // record the actual number of walls
        ctx.data().numWalls = wallPlacements.size();
        return wallPlacements;
    }

    static std::vector<MovableObjectPlacement> determineMovableObjectPlacements(Engine &ctx, const MacguffinPlacement &macguffinPlacement, const GoalPlacement &goalPlacement, const std::vector<WallPlacement> &wallPlacements) {
        std::vector<MovableObjectPlacement> objectPlacements;
        
        // Return empty vector if no movable objects needed
        if (ctx.data().numMovableObjects <= 0) {
            return objectPlacements;
        }
        
        // Room boundaries
        float minX = -consts::worldWidth / 2.0f;
        float maxX = consts::worldWidth / 2.0f;
        float minY = -consts::worldLength / 2.0f;
        float maxY = consts::worldLength / 2.0f;
        
        // Buffers to keep objects from overlapping
        float macguffinBuffer = consts::macguffinRadius + consts::movableObjectRadius + 1.0f;
        float goalBuffer = consts::goalRadius + consts::movableObjectRadius + 1.0f;
        float objectBuffer = consts::movableObjectRadius * 2.0f + 1.0f;
        float wallBuffer = consts::borderWidth + consts::movableObjectRadius + 0.5f;
        float borderBuffer = consts::borderWidth + consts::movableObjectRadius;
        
        // Adjusted room boundaries accounting for border walls
        float adjustedMinX = minX + borderBuffer;
        float adjustedMaxX = maxX - borderBuffer;
        float adjustedMinY = minY + borderBuffer;
        float adjustedMaxY = maxY - borderBuffer;
        
        // Try up to the maximum number of attempts for movable object placement
        int attempt = 0;
        
        while (objectPlacements.size() < ctx.data().numMovableObjects && attempt < consts::maxMovableObjectPlacementAttempts) {
            attempt++;
            
            // Random scale for the movable object (0.8 to 1.2 times base size)
            float scale = randBetween(ctx, 0.8f, 1.2f);
            
            // Random position within adjusted room bounds
            float x = randBetween(ctx, adjustedMinX, adjustedMaxX);
            float y = randBetween(ctx, adjustedMinY, adjustedMaxY);
            
            // Check if object overlaps with macguffin
            float distToMacguffin = std::sqrt(
                std::pow(x - macguffinPlacement.x, 2) + 
                std::pow(y - macguffinPlacement.y, 2));
            
            if (distToMacguffin < macguffinBuffer) {
                continue; // Skip this attempt
            }
            
            // Check if object overlaps with goal
            float distToGoal = std::sqrt(
                std::pow(x - goalPlacement.x, 2) + 
                std::pow(y - goalPlacement.y, 2));
            
            if (distToGoal < goalBuffer) {
                continue; // Skip this attempt
            }
            
            // Check if object overlaps with walls
            bool overlapsWithWall = false;
            for (const auto &wall : wallPlacements) {
                // Check if object is too close to a wall
                // Simple box overlap check
                if (std::abs(x - wall.x) < (consts::movableObjectRadius * scale + wall.width / 2 + wallBuffer) &&
                    std::abs(y - wall.y) < (consts::movableObjectRadius * scale + wall.height / 2 + wallBuffer)) {
                    overlapsWithWall = true;
                    break;
                }
            }
            
            if (overlapsWithWall) {
                continue; // Skip this attempt
            }
            
            // Check if object overlaps with existing movable objects
            bool overlapsWithObject = false;
            for (const auto &existingObject : objectPlacements) {
                float distToObject = std::sqrt(
                    std::pow(x - existingObject.x, 2) + 
                    std::pow(y - existingObject.y, 2));
                
                if (distToObject < objectBuffer) {
                    overlapsWithObject = true;
                    break;
                }
            }
            
            if (!overlapsWithObject) {
                // Movable object is valid, add it
                MovableObjectPlacement obj;
                obj.x = x;
                obj.y = y;
                obj.scale = scale;
                objectPlacements.push_back(obj);
            }
        }
        
        // Record the actual number of movable objects
        ctx.data().numMovableObjects = objectPlacements.size();
        return objectPlacements;
    }
    
    
    static std::vector<AntPlacement> determineAntPlacements(Engine &ctx, const MacguffinPlacement &macguffinPlacement, const GoalPlacement &goalPlacement, const std::vector<WallPlacement> &wallPlacements, const std::vector<MovableObjectPlacement> &movableObjectPlacements) {
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
        
        // Buffers to keep ants from overlapping
        float macguffinBuffer = consts::macguffinRadius + consts::antRadius + 0.5f;
        float goalBuffer = consts::goalRadius + consts::antRadius + 0.5f;
        float wallBuffer = consts::borderWidth + consts::antRadius + 0.2f;
        float objectBuffer = consts::movableObjectRadius + consts::antRadius + 0.5f;
        float antBuffer = consts::antRadius * 2.0f + 0.2f;
        float borderBuffer = consts::borderWidth + consts::antRadius;
        
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
                float distToMacguffin = std::sqrt(
                    std::pow(x - macguffinPlacement.x, 2) + 
                    std::pow(y - macguffinPlacement.y, 2));
                
                if (distToMacguffin < macguffinBuffer) {
                    continue; // Skip this attempt
                }
                
                // Check if ant overlaps with goal
                float distToGoal = std::sqrt(
                    std::pow(x - goalPlacement.x, 2) + 
                    std::pow(y - goalPlacement.y, 2));
                
                if (distToGoal < goalBuffer) {
                    continue; // Skip this attempt
                }
                
                // Check if ant overlaps with walls
                bool overlapsWithWall = false;
                for (const auto &wall : wallPlacements) {
                    // Simple box-based distance check
                    if (std::abs(x - wall.x) < (consts::antRadius + wall.width / 2 + wallBuffer) &&
                        std::abs(y - wall.y) < (consts::antRadius + wall.height / 2 + wallBuffer)) {
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
                    float distToObject = std::sqrt(
                        std::pow(x - obj.x, 2) + 
                        std::pow(y - obj.y, 2));
                    
                    // Account for object scale
                    if (distToObject < objectBuffer * obj.scale) {
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
                    
                    if (distToAnt < antBuffer) {
                        overlapsWithAnt = true;
                        break;
                    }
                }
                
                if (!overlapsWithAnt) {
                    // Ant position is valid, add it
                    AntPlacement ant;
                    ant.x = x;
                    ant.y = y;
                    ant.angle = angle;
                    antPlacements.push_back(ant);
                    placedSuccessfully = true;
                }
            }
            
            // If we couldn't place this ant after all attempts, stop trying to place more
            if (!placedSuccessfully) {
                break;
            }
        }
        ctx.singleton<NumAnts>().count = antPlacements.size();
        return antPlacements;
    }


    // Generate the hive simulation world
    void generateWorld(Engine &ctx)
    {
        resetPersistentEntities(ctx);

        // Place macguffin first
        const auto &macguffinPlacement = determineMacguffinPlacement(ctx);
        createMacguffin(ctx, macguffinPlacement);

        // goal
        const auto &goalPlacement = determineGoalPlacement(ctx, macguffinPlacement);
        createGoal(ctx, goalPlacement);

        // walls
        const auto &wallPlacements = determineWallPlacements(ctx, macguffinPlacement, goalPlacement);
        for (size_t i = 0; i < ctx.data().numWalls; i++) {
            createWall(ctx, wallPlacements[i], i);
        }

        // movable objects
        const auto &movableObjectPlacements = determineMovableObjectPlacements(ctx, macguffinPlacement, goalPlacement, wallPlacements);
        for (size_t i = 0; i < ctx.data().numMovableObjects; i++) {
            createMovableObject(ctx, movableObjectPlacements[i], i);
        }

        // ants
        const auto &antPlacements = determineAntPlacements(ctx, macguffinPlacement, goalPlacement, wallPlacements, movableObjectPlacements);
        for (int32_t i = 0; i < ctx.singleton<NumAnts>().count; ++i) {
            createAnt(ctx, antPlacements[i], i);
        }
    }

}
