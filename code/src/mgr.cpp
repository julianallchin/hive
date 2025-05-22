#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>
#include <madrona/render/api.hpp>

#include <array>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::py;

namespace madEscape
{

    struct RenderGPUState
    {
        render::APILibHandle apiLib;
        render::APIManager apiMgr;
        render::GPUHandle gpu;
    };

    static inline Optional<RenderGPUState> initRenderGPUState(
        const Manager::Config &mgr_cfg)
    {
        if (mgr_cfg.extRenderDev || !mgr_cfg.enableBatchRenderer)
        {
            return Optional<RenderGPUState>::none();
        }

        auto render_api_lib = render::APIManager::loadDefaultLib();
        render::APIManager render_api_mgr(render_api_lib.lib());
        render::GPUHandle gpu = render_api_mgr.initGPU(mgr_cfg.gpuID);

        return RenderGPUState{
            .apiLib = std::move(render_api_lib),
            .apiMgr = std::move(render_api_mgr),
            .gpu = std::move(gpu),
        };
    }

    static inline Optional<render::RenderManager> initRenderManager(
        const Manager::Config &mgr_cfg,
        const Optional<RenderGPUState> &render_gpu_state)
    {
        if (!mgr_cfg.extRenderDev && !mgr_cfg.enableBatchRenderer)
        {
            return Optional<render::RenderManager>::none();
        }

        render::APIBackend *render_api;
        render::GPUDevice *render_dev;

        if (render_gpu_state.has_value())
        {
            render_api = render_gpu_state->apiMgr.backend();
            render_dev = render_gpu_state->gpu.device();
        }
        else
        {
            render_api = mgr_cfg.extRenderAPI;
            render_dev = mgr_cfg.extRenderDev;
        }

        return render::RenderManager(render_api, render_dev, {
                                                                 .enableBatchRenderer = mgr_cfg.enableBatchRenderer,
                                                                 .renderMode = render::RenderManager::Config::RenderMode::RGBD,
                                                                 .agentViewWidth = mgr_cfg.batchRenderViewWidth,
                                                                 .agentViewHeight = mgr_cfg.batchRenderViewHeight,
                                                                 .numWorlds = mgr_cfg.numWorlds,
                                                                 .maxViewsPerWorld = consts::maxAnts,
                                                                 .maxInstancesPerWorld = 1000,
                                                                 .execMode = mgr_cfg.execMode,
                                                                 .voxelCfg = {},
                                                             });
    }

    struct Manager::Impl
    {
        Config cfg;
        PhysicsLoader physicsLoader;
        WorldReset *worldResetBuffer;
        Action *actionsBuffer;
        Optional<RenderGPUState> renderGPUState;
        Optional<render::RenderManager> renderMgr;

        inline Impl(const Manager::Config &mgr_cfg,
                    PhysicsLoader &&phys_loader,
                    WorldReset *reset_buffer,
                    Action *actionsBuffer,
                    Optional<RenderGPUState> &&render_gpu_state,
                    Optional<render::RenderManager> &&render_mgr)
            : cfg(mgr_cfg),
              physicsLoader(std::move(phys_loader)),
              worldResetBuffer(reset_buffer),
              actionsBuffer(actionsBuffer),
              renderGPUState(std::move(render_gpu_state)),
              renderMgr(std::move(render_mgr))
        {
        }

        inline virtual ~Impl() {}

        virtual void run() = 0;

        virtual Tensor exportTensor(ExportID slot,
                                    TensorElementType type,
                                    madrona::Span<const int64_t> dimensions) const = 0;

        static inline Impl *init(const Config &cfg);
    };

    struct Manager::CPUImpl final : Manager::Impl
    {
        using TaskGraphT =
            TaskGraphExecutor<Engine, Sim, Sim::Config, Sim::WorldInit>;

        TaskGraphT cpuExec;

        inline CPUImpl(const Manager::Config &mgr_cfg,
                       PhysicsLoader &&phys_loader,
                       WorldReset *reset_buffer,
                       Action *actionsBuffer,
                       Optional<RenderGPUState> &&render_gpu_state,
                       Optional<render::RenderManager> &&render_mgr,
                       TaskGraphT &&cpu_exec)
            : Impl(mgr_cfg, std::move(phys_loader),
                   reset_buffer, actionsBuffer,
                   std::move(render_gpu_state), std::move(render_mgr)),
              cpuExec(std::move(cpu_exec))
        {
        }

        inline virtual ~CPUImpl() final {}

        inline virtual void run()
        {
            cpuExec.run();
        }

        virtual inline Tensor exportTensor(ExportID slot,
                                           TensorElementType type,
                                           madrona::Span<const int64_t> dims) const final
        {
            void *dev_ptr = cpuExec.getExported((uint32_t)slot);
            return Tensor(dev_ptr, type, dims, Optional<int>::none());
        }
    };

#ifdef MADRONA_CUDA_SUPPORT
    struct Manager::CUDAImpl final : Manager::Impl
    {
        MWCudaExecutor gpuExec;
        MWCudaLaunchGraph stepGraph;

        inline CUDAImpl(const Manager::Config &mgr_cfg,
                        PhysicsLoader &&phys_loader,
                        WorldReset *reset_buffer,
                        Action *actionsBuffer,
                        Optional<RenderGPUState> &&render_gpu_state,
                        Optional<render::RenderManager> &&render_mgr,
                        MWCudaExecutor &&gpu_exec)
            : Impl(mgr_cfg, std::move(phys_loader),
                   reset_buffer, actionsBuffer,
                   std::move(render_gpu_state), std::move(render_mgr)),
              gpuExec(std::move(gpu_exec)),
              stepGraph(gpuExec.buildLaunchGraphAllTaskGraphs())
        {
        }

        inline virtual ~CUDAImpl() final {}

        inline virtual void run()
        {
            gpuExec.run(stepGraph);
        }

        virtual inline Tensor exportTensor(ExportID slot,
                                           TensorElementType type,
                                           madrona::Span<const int64_t> dims) const final
        {
            void *dev_ptr = gpuExec.getExported((uint32_t)slot);
            return Tensor(dev_ptr, type, dims, cfg.gpuID);
        }
    };
#endif

    static void loadRenderObjects(render::RenderManager &render_mgr)
    {
        StackAlloc tmp_alloc;

        std::array<std::string, (size_t)SimObject::NumObjects> render_asset_paths;
        render_asset_paths[(size_t)SimObject::MovableObject] =
            (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
        render_asset_paths[(size_t)SimObject::Wall] =
            (std::filesystem::path(DATA_DIR) / "wall_render.obj").string();
        render_asset_paths[(size_t)SimObject::Macguffin] =
            (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
        render_asset_paths[(size_t)SimObject::Ant] =
            (std::filesystem::path(DATA_DIR) / "agent_render.obj").string();
        render_asset_paths[(size_t)SimObject::Goal] =
            (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
        render_asset_paths[(size_t)SimObject::Plane] =
            (std::filesystem::path(DATA_DIR) / "plane.obj").string();

        std::array<const char *, (size_t)SimObject::NumObjects> render_asset_cstrs;
        for (size_t i = 0; i < render_asset_paths.size(); i++)
        {
            render_asset_cstrs[i] = render_asset_paths[i].c_str();
        }

        imp::AssetImporter importer;

        std::array<char, 1024> import_err;
        auto render_assets = importer.importFromDisk(
            render_asset_cstrs, Span<char>(import_err.data(), import_err.size()));

        if (!render_assets.has_value())
        {
            FATAL("Failed to load render assets: %s", import_err);
        }

        auto materials = std::to_array<imp::SourceMaterial>({
            {render::rgb8ToFloat(130, 82, 1), -1, 0.8f, 0.2f}, // Brown for movable objects
            {
                math::Vector4{0.4f, 0.4f, 0.4f, 0.0f},
                -1,
                0.8f,
                0.2f,
            }, // Gray for walls
            {
                math::Vector4{0.1f, 0.1f, 0.1f, 0.0f},
                1,
                0.5f,
                1.0f,
            },                                                    // Black for ants
            {render::rgb8ToFloat(230, 230, 230), -1, 0.8f, 1.0f}, // White for ant details
            {
                math::Vector4{0.5f, 0.3f, 0.3f, 0.0f},
                0,
                0.8f,
                0.2f,
            },                                                  // Earth for plane
            {render::rgb8ToFloat(230, 20, 20), -1, 0.8f, 1.0f}, // Red for macguffin
            {render::rgb8ToFloat(20, 230, 20), -1, 0.8f, 1.0f}, // Green for goal
        });

        // Override materials
        render_assets->objects[(CountT)SimObject::MovableObject].meshes[0].materialIDX = 0;
        render_assets->objects[(CountT)SimObject::Wall].meshes[0].materialIDX = 1;
        render_assets->objects[(CountT)SimObject::Macguffin].meshes[0].materialIDX = 5;
        render_assets->objects[(CountT)SimObject::Ant].meshes[0].materialIDX = 2;
        render_assets->objects[(CountT)SimObject::Ant].meshes[1].materialIDX = 3;
        render_assets->objects[(CountT)SimObject::Ant].meshes[2].materialIDX = 3;
        render_assets->objects[(CountT)SimObject::Goal].meshes[0].materialIDX = 6;
        render_assets->objects[(CountT)SimObject::Plane].meshes[0].materialIDX = 4;

        imp::ImageImporter img_importer;
        Span<imp::SourceTexture> imported_textures = img_importer.importImages(
            tmp_alloc, {
                           (std::filesystem::path(DATA_DIR) /
                            "green_grid.png")
                               .string()
                               .c_str(),
                           (std::filesystem::path(DATA_DIR) /
                            "smile.png")
                               .string()
                               .c_str(),
                       });

        render_mgr.loadObjects(
            render_assets->objects, materials, imported_textures);

        render_mgr.configureLighting({{true, math::Vector3{1.0f, 1.0f, -2.0f}, math::Vector3{1.0f, 1.0f, 1.0f}}});
    }

    static void loadPhysicsObjects(PhysicsLoader &loader)
    {
        // Set up asset paths for each SimObject
        // We need to map our SimObject enum to the appropriate collision meshes
        // Original mapping:
        // - Cube -> MovableObject
        // - Wall -> PhysicsEntity
        // - Door -> PhysicsEntity (using wall collision)
        // - Agent -> Ant
        // - Button -> Macguffin
        // - Plane -> Plane (handled separately)
        
        // We'll use a vector to store only the paths we need
        std::vector<std::string> asset_paths;
        std::vector<SimObject> path_mapping;
        
        // Define the mapping from SimObject to asset paths
        // The order here determines the order in the asset_paths array
        std::vector<std::pair<SimObject, std::string>> object_paths = {
            {SimObject::Ant, "agent_collision_simplified.obj"},
            {SimObject::MovableObject, "cube_collision.obj"},
            {SimObject::Wall, "wall_collision.obj"},
            {SimObject::Macguffin, "cube_collision.obj"},
            {SimObject::Goal, "cube_collision.obj"}
        };
        
        // Initialize the mapping and paths
        for (const auto &[obj, path] : object_paths) {
            path_mapping.push_back(obj);
            asset_paths.push_back((std::filesystem::path(DATA_DIR) / path).string());
        }

        // Convert paths to C strings for the importer
        std::vector<const char *> asset_cstrs;
        for (const auto &path : asset_paths) {
            asset_cstrs.push_back(path.c_str());
        }

        // Import the collision meshes
        imp::AssetImporter importer;
        char import_err_buffer[4096];
        

        auto imported_src_hulls = importer.importFromDisk(
            asset_cstrs, import_err_buffer, true);

        if (!imported_src_hulls.has_value()) {
            printf("Failed to import source hulls: %s\n", import_err_buffer);
            FATAL("%s", import_err_buffer);
        }

        DynArray<imp::SourceMesh> src_convex_hulls(
            imported_src_hulls->objects.size());

        DynArray<DynArray<SourceCollisionPrimitive>> prim_arrays(0);
        HeapArray<SourceCollisionObject> src_objs(
            (CountT)SimObject::NumObjects);

        // Create a mapping from SimObject to imported object index
        std::unordered_map<SimObject, size_t> obj_to_import_idx;
        for (size_t i = 0; i < path_mapping.size(); i++) {
            obj_to_import_idx[path_mapping[i]] = i;
        }

        auto setupHull = [&](SimObject obj_id,
                             float inv_mass,
                             RigidBodyFrictionData friction) {
            // Skip if this object type doesn't have a corresponding imported mesh
            if (obj_to_import_idx.find(obj_id) == obj_to_import_idx.end()) {
                printf("Warning: No mesh for SimObject %d\n", (int)obj_id);
                return;
            }

            size_t import_idx = obj_to_import_idx[obj_id];
            auto meshes = imported_src_hulls->objects[import_idx].meshes;
            DynArray<SourceCollisionPrimitive> prims(meshes.size());

            for (const imp::SourceMesh &mesh : meshes) {
                src_convex_hulls.push_back(mesh);
                prims.push_back({
                    .type = CollisionPrimitive::Type::Hull,
                    .hullInput = {
                        .hullIDX = uint32_t(src_convex_hulls.size() - 1),
                    },
                });
            }

            prim_arrays.emplace_back(std::move(prims));

            src_objs[(CountT)obj_id] = SourceCollisionObject{
                .prims = Span<const SourceCollisionPrimitive>(prim_arrays.back()),
                .invMass = inv_mass,
                .friction = friction,
            };
        };

        setupHull(SimObject::MovableObject, 0.1f, {
                                                      .muS = 0.5f,
                                                      .muD = 0.75f,
                                                  });

        setupHull(SimObject::Wall, 0.f, {
            .muS = 0.5f,
            .muD = 0.5f,
        });

        setupHull(SimObject::Macguffin, 0.03f, {
                                                   // Macguffin is harder to move, meant for multiple ants
                                                   .muS = 0.6f,
                                                   .muD = 0.7f,
        });

        setupHull(SimObject::Ant, 1.f, {
            .muS = 0.5f,
            .muD = 0.5f,
        });

        setupHull(SimObject::Goal, 0.f, {
                                            // Goal has no mass, it's just visual            .muS = 0.0f,
            .muS = 0.0f,
            .muD = 0.0f
        });

        SourceCollisionPrimitive plane_prim{
            .type = CollisionPrimitive::Type::Plane,
            .plane = {},
        };

        src_objs[(CountT)SimObject::Plane] = {
            .prims = Span<const SourceCollisionPrimitive>(&plane_prim, 1),
            .invMass = 0.f,
            .friction = {
                .muS = 0.5f,
                .muD = 0.5f,
            },
        };

        StackAlloc tmp_alloc;
        RigidBodyAssets rigid_body_assets;
        CountT num_rigid_body_data_bytes;
        void *rigid_body_data = RigidBodyAssets::processRigidBodyAssets(
            src_convex_hulls,
            src_objs,
            false,
            tmp_alloc,
            &rigid_body_assets,
            &num_rigid_body_data_bytes);

        if (rigid_body_data == nullptr)
        {
            FATAL("Invalid collision hull input");
        }

        // This is a bit hacky, but in order to make sure the ants
        // remain controllable by the policy, they are only allowed to
        // rotate around the Z axis (infinite inertia in x & y axes)
        rigid_body_assets.metadatas[(CountT)SimObject::Ant].mass.invInertiaTensor.x = 0.f;
        rigid_body_assets.metadatas[(CountT)SimObject::Ant].mass.invInertiaTensor.y = 0.f;

        loader.loadRigidBodies(rigid_body_assets);
        free(rigid_body_data);
    }

    Manager::Impl *Manager::Impl::init(
        const Manager::Config &mgr_cfg)
    {
        Sim::Config sim_cfg;
        sim_cfg.autoReset = mgr_cfg.autoReset;
        sim_cfg.initRandKey = rand::initKey(mgr_cfg.randSeed);

        // Pass randomization parameters
        // validate ant params
        assert(mgr_cfg.minAntsRand <= mgr_cfg.maxAntsRand);
        assert(mgr_cfg.minAntsRand >= consts::minAnts);
        assert(mgr_cfg.maxAntsRand <= consts::maxAnts);
        // validate movable object params
        assert(mgr_cfg.minMovableObjectsRand <= mgr_cfg.maxMovableObjectsRand);
        assert(mgr_cfg.minMovableObjectsRand >= consts::minMovableObjects);
        assert(mgr_cfg.maxMovableObjectsRand <= consts::maxMovableObjects);
        // validate wall params
        assert(mgr_cfg.minWallsRand <= mgr_cfg.maxWallsRand);
        assert(mgr_cfg.minWallsRand >= consts::minWalls);
        assert(mgr_cfg.maxWallsRand <= consts::maxWalls);
        // pass to sim
        sim_cfg.minAntsRand = mgr_cfg.minAntsRand;
        sim_cfg.maxAntsRand = mgr_cfg.maxAntsRand;
        sim_cfg.minMovableObjectsRand = mgr_cfg.minMovableObjectsRand;
        sim_cfg.maxMovableObjectsRand = mgr_cfg.maxMovableObjectsRand;
        sim_cfg.minWallsRand = mgr_cfg.minWallsRand;
        sim_cfg.maxWallsRand = mgr_cfg.maxWallsRand;

        switch (mgr_cfg.execMode)
        {
        case ExecMode::CUDA:
        {
#ifdef MADRONA_CUDA_SUPPORT
            CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);

            PhysicsLoader phys_loader(ExecMode::CUDA, 10);
            loadPhysicsObjects(phys_loader);

            ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();
            sim_cfg.rigidBodyObjMgr = phys_obj_mgr;

            Optional<RenderGPUState> render_gpu_state =
                initRenderGPUState(mgr_cfg);

            Optional<render::RenderManager> render_mgr =
                initRenderManager(mgr_cfg, render_gpu_state);

            if (render_mgr.has_value())
            {
                loadRenderObjects(*render_mgr);
                sim_cfg.renderBridge = render_mgr->bridge();
            }
            else
            {
                sim_cfg.renderBridge = nullptr;
            }

            HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

            MWCudaExecutor gpu_exec({
                                        .worldInitPtr = world_inits.data(),
                                        .numWorldInitBytes = sizeof(Sim::WorldInit),
                                        .userConfigPtr = (void *)&sim_cfg,
                                        .numUserConfigBytes = sizeof(Sim::Config),
                                        .numWorldDataBytes = sizeof(Sim),
                                        .worldDataAlignment = alignof(Sim),
                                        .numWorlds = mgr_cfg.numWorlds,
                                        .numTaskGraphs = 1,
                                        .numExportedBuffers = (uint32_t)ExportID::NumExports,
                                    },
                                    {
                                        {GPU_HIDESEEK_SRC_LIST},
                                        {GPU_HIDESEEK_COMPILE_FLAGS},
                                        CompileConfig::OptMode::LTO,
                                    },
                                    cu_ctx);

            WorldReset *world_reset_buffer =
                (WorldReset *)gpu_exec.getExported((uint32_t)ExportID::Reset);

            Action *agent_actions_buffer =
                (Action *)gpu_exec.getExported((uint32_t)ExportID::Action);

            return new CUDAImpl{
                mgr_cfg,
                std::move(phys_loader),
                world_reset_buffer,
                agent_actions_buffer,
                std::move(render_gpu_state),
                std::move(render_mgr),
                std::move(gpu_exec),
            };
#else
            FATAL("Madrona was not compiled with CUDA support");
#endif
        }
        break;
        case ExecMode::CPU:
        {
            PhysicsLoader phys_loader(ExecMode::CPU, 10);
            loadPhysicsObjects(phys_loader);

            ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();
            sim_cfg.rigidBodyObjMgr = phys_obj_mgr;

            Optional<RenderGPUState> render_gpu_state =
                initRenderGPUState(mgr_cfg);

            Optional<render::RenderManager> render_mgr =
                initRenderManager(mgr_cfg, render_gpu_state);

            if (render_mgr.has_value())
            {
                loadRenderObjects(*render_mgr);
                sim_cfg.renderBridge = render_mgr->bridge();
            }
            else
            {
                sim_cfg.renderBridge = nullptr;
            }

            HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

            CPUImpl::TaskGraphT cpu_exec{
                ThreadPoolExecutor::Config{
                    .numWorlds = mgr_cfg.numWorlds,
                    .numExportedBuffers = (uint32_t)ExportID::NumExports,
                },
                sim_cfg,
                world_inits.data(),
                (uint32_t)TaskGraphID::NumTaskGraphs,
            };

            WorldReset *world_reset_buffer =
                (WorldReset *)cpu_exec.getExported((uint32_t)ExportID::Reset);

            Action *agent_actions_buffer =
                (Action *)cpu_exec.getExported((uint32_t)ExportID::Action);

            auto cpu_impl = new CPUImpl{
                mgr_cfg,
                std::move(phys_loader),
                world_reset_buffer,
                agent_actions_buffer,
                std::move(render_gpu_state),
                std::move(render_mgr),
                std::move(cpu_exec),
            };

            return cpu_impl;
        }
        break;
        default:
            MADRONA_UNREACHABLE();
        }
    }

    Manager::Manager(const Config &cfg)
        : impl_(Impl::init(cfg))
    {
        // Currently, there is no way to populate the initial set of observations
        // without stepping the simulations in order to execute the taskgraph.
        // Therefore, after setup, we step all the simulations with a forced reset
        // that ensures the first real step will have valid observations at the
        // start of a fresh episode in order to compute actions.
        //
        // This will be improved in the future with support for multiple task
        // graphs, allowing a small task graph to be executed after initialization.

        for (int32_t i = 0; i < (int32_t)cfg.numWorlds; i++)
        {
            triggerReset(i);
        }

        step();
    }

    Manager::~Manager() {}

    void Manager::step()
    {
        impl_->run();

        if (impl_->renderMgr.has_value())
        {
            impl_->renderMgr->readECS();
        }

        if (impl_->cfg.enableBatchRenderer)
        {
            impl_->renderMgr->batchRender();
        }
    }

    Tensor Manager::resetTensor() const
    {
        return impl_->exportTensor(ExportID::Reset,
                                   TensorElementType::Int32,
                                   {
                                       impl_->cfg.numWorlds,
                                       1,
                                   });
    }

    Tensor Manager::actionTensor() const
    {
        return impl_->exportTensor(ExportID::Action, TensorElementType::Int32,
                                   {
                                       impl_->cfg.numWorlds,
                                       consts::maxAnts,
                                       4,
                                   });
    }

    Tensor Manager::rewardTensor() const
    {
        return impl_->exportTensor(ExportID::Reward, TensorElementType::Float32,
                                   {
                                       impl_->cfg.numWorlds,
                                       1,
                                   });
    }

    Tensor Manager::doneTensor() const
    {
        return impl_->exportTensor(ExportID::Done, TensorElementType::Int32,
                                   {
                                       impl_->cfg.numWorlds,
                                       1,
                                   });
    }

    Tensor Manager::observationTensor() const
    {
        return impl_->exportTensor(ExportID::Observation,
                                   TensorElementType::Float32,
                                   {
                                       impl_->cfg.numWorlds,
                                       consts::maxAnts,
                                       8,
                                   });
    }

    Tensor Manager::numAntsTensor() const
    {
        // Return information about how many ants are active in each world
        // This is used by the Python code to mask observations/actions for inactive ants
        return impl_->exportTensor(ExportID::NumAnts,
                                  TensorElementType::Int32,
                                  {
                                      impl_->cfg.numWorlds,
                                      1,
                                  });
    }

    Tensor Manager::lidarTensor() const
    {
        return impl_->exportTensor(ExportID::Lidar, TensorElementType::Float32,
                                   {
                                       impl_->cfg.numWorlds,
                                       consts::maxAnts,
                                       consts::numLidarSamples,
                                       2,
                                   });
    }

    Tensor Manager::stepsRemainingTensor() const
    {
        return impl_->exportTensor(ExportID::StepsRemaining,
                                   TensorElementType::Int32,
                                   {
                                       impl_->cfg.numWorlds,
                                       consts::maxAnts,
                                       1,
                                   });
    }

    Tensor Manager::rgbTensor() const
    {
        const uint8_t *rgb_ptr = impl_->renderMgr->batchRendererRGBOut();

        return Tensor((void *)rgb_ptr, TensorElementType::UInt8, {
                                                                     impl_->cfg.numWorlds,
                                                                     consts::maxAnts,
                                                                     impl_->cfg.batchRenderViewHeight,
                                                                     impl_->cfg.batchRenderViewWidth,
                                                                     4,
                                                                 },
                      impl_->cfg.gpuID);
    }

    Tensor Manager::depthTensor() const
    {
        const float *depth_ptr = impl_->renderMgr->batchRendererDepthOut();

        return Tensor((void *)depth_ptr, TensorElementType::Float32, {
                                                                         impl_->cfg.numWorlds,
                                                                         consts::maxAnts,
                                                                         impl_->cfg.batchRenderViewHeight,
                                                                         impl_->cfg.batchRenderViewWidth,
                                                                         1,
                                                                     },
                      impl_->cfg.gpuID);
    }

    void Manager::triggerReset(int32_t world_idx)
    {
        WorldReset reset{
            1,
        };

        auto *reset_ptr = impl_->worldResetBuffer + world_idx;

        if (impl_->cfg.execMode == ExecMode::CUDA)
        {
#ifdef MADRONA_CUDA_SUPPORT
            cudaMemcpy(reset_ptr, &reset, sizeof(WorldReset),
                       cudaMemcpyHostToDevice);
#endif
        }
        else
        {
            *reset_ptr = reset;
        }
    }

    void Manager::setAction(int32_t world_idx,
                            int32_t agent_idx,
                            int32_t move_amount,
                            int32_t move_angle,
                            int32_t rotate,
                            int32_t grab)
    {
        Action action{
            .moveAmount = move_amount,
            .moveAngle = move_angle,
            .rotate = rotate,
            .grab = grab,
        };

        auto *action_ptr = impl_->actionsBuffer +
                           world_idx * consts::maxAnts + agent_idx;

        if (impl_->cfg.execMode == ExecMode::CUDA)
        {
#ifdef MADRONA_CUDA_SUPPORT
            cudaMemcpy(action_ptr, &action, sizeof(Action),
                       cudaMemcpyHostToDevice);
#endif
        }
        else
        {
            *action_ptr = action;
        }
    }

    render::RenderManager &Manager::getRenderManager()
    {
        return *impl_->renderMgr;
    }

}
