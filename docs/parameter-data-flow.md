To pass randomization parameters for the number of ants (minAntsRand, maxAntsRand, etc.) from Python to your C++ simulation, you need to create a flow of information from your Python training script all the way to the Sim::Config structure. Here's how to do it:

1. Update Manager::Config to Include Randomization Parameters
First, you need to add these parameters to the Manager::Config structure in mgr.hpp:

cpp
CopyInsert
struct Config {
    madrona::ExecMode execMode;
    int gpuID;
    uint32_t numWorlds;
    uint32_t numAnts; // This should be the maximum number of ants
    uint32_t randSeed;
    bool autoReset;
    // Add these parameters
    uint32_t minAntsRand;
    uint32_t maxAntsRand;
    uint32_t minMovableObjectsRand;
    uint32_t maxMovableObjectsRand;
    uint32_t minWallsRand;
    uint32_t maxWallsRand;
    float worldSize;
    uint32_t maxSteps;
    bool enableBatchRenderer;
    // ... rest of config
};
2. Pass These Parameters to Sim::Config in Manager::Impl::init
In the mgr.cpp file, find the Manager::Impl::init function, which creates the Sim::Config structure. You need to pass your randomization parameters from Manager::Config to Sim::Config:

cpp
CopyInsert
// In Manager::Impl::init function
// ...
Sim::Config sim_cfg {
    .autoReset = mgr_cfg.autoReset,
    .initRandKey = RandKey {
        (uint32_t)mgr_cfg.randSeed,
    },
    .rigidBodyObjMgr = phys_objects.get(),
    .renderBridge = render_mgr.hasValue() ? &render_mgr->getBridge() : nullptr,
    // Add these parameters from mgr_cfg to sim_cfg
    .minAntsRand = mgr_cfg.minAntsRand,
    .maxAntsRand = mgr_cfg.maxAntsRand,
    .minMovableObjectsRand = mgr_cfg.minMovableObjectsRand,
    .maxMovableObjectsRand = mgr_cfg.maxMovableObjectsRand,
    .minWallsRand = mgr_cfg.minWallsRand,
    .maxWallsRand = mgr_cfg.maxWallsRand,
};
3. Update Python Bindings for SimManager Constructor
The C++ Manager constructor parameters need to be exposed to Python through bindings. This is typically done in a file like bindings.cpp. You need to update the SimManager constructor binding to accept these additional parameters:

cpp
CopyInsert
// In bindings.cpp
py::class_<madHive::Manager>(m, "SimManager")
    .def(py::init<madHive::Manager::Config>())
    .def("step", &madHive::Manager::step)
    // ... other bindings ...

py::class_<madHive::Manager::Config>(m, "SimManagerConfig")
    .def(py::init<>())
    .def_readwrite("exec_mode", &madHive::Manager::Config::execMode)
    .def_readwrite("gpu_id", &madHive::Manager::Config::gpuID)
    .def_readwrite("num_worlds", &madHive::Manager::Config::numWorlds)
    .def_readwrite("num_ants", &madHive::Manager::Config::numAnts)
    .def_readwrite("min_ants_rand", &madHive::Manager::Config::minAntsRand)
    .def_readwrite("max_ants_rand", &madHive::Manager::Config::maxAntsRand)
    .def_readwrite("min_movable_objects_rand", &madHive::Manager::Config::minMovableObjectsRand)
    .def_readwrite("max_movable_objects_rand", &madHive::Manager::Config::maxMovableObjectsRand)
    .def_readwrite("min_walls_rand", &madHive::Manager::Config::minWallsRand)
    .def_readwrite("max_walls_rand", &madHive::Manager::Config::maxWallsRand)
    .def_readwrite("rand_seed", &madHive::Manager::Config::randSeed)
    .def_readwrite("auto_reset", &madHive::Manager::Config::autoReset)
    .def_readwrite("world_size", &madHive::Manager::Config::worldSize)
    .def_readwrite("max_steps", &madHive::Manager::Config::maxSteps)
    // ... other config parameters ...
4. Update Python Training Script to Set These Parameters
Finally, update your train.py script to set these parameters:

python
CopyInsert
# Create and configure SimManager
config = madrona_hive.SimManagerConfig()
config.exec_mode = madrona_hive.madrona.ExecMode.CUDA if args.gpu_sim else madrona_hive.madrona.ExecMode.CPU
config.gpu_id = args.gpu_id
config.num_worlds = args.num_worlds
config.num_ants = args.max_ants  # Maximum possible ants (for tensor allocation)
config.min_ants_rand = args.min_ants
config.max_ants_rand = args.max_ants
config.min_movable_objects_rand = args.min_movable_objects
config.max_movable_objects_rand = args.max_movable_objects
config.min_walls_rand = args.min_walls
config.max_walls_rand = args.max_walls
config.rand_seed = 5
config.auto_reset = True
config.world_size = args.world_size
config.max_steps = args.max_steps
config.enable_batch_renderer = False

# Create simulation manager with config
sim = madrona_hive.SimManager(config)
5. Add Command Line Arguments for These Parameters
Add command line arguments to your Python script to easily configure these parameters:

python
CopyInsert
arg_parser.add_argument('--min-ants', type=int, default=80)
arg_parser.add_argument('--max-ants', type=int, default=120)
arg_parser.add_argument('--min-movable-objects', type=int, default=5)
arg_parser.add_argument('--max-movable-objects', type=int, default=15)
arg_parser.add_argument('--min-walls', type=int, default=1)
arg_parser.add_argument('--max-walls', type=int, default=3)
arg_parser.add_argument('--world-size', type=float, default=20.0)
arg_parser.add_argument('--max-steps', type=int, default=1000)
6. Update the Level Generation Code
Finally, make sure your level generation code in the ECS systems uses these parameters to randomize the number of ants:

cpp
CopyInsert
// In your level generation system
void GenerateLevel::update(ECSContext &ctx) {
    // ...
    
    Sim &sim = ctx.data();
    
    // Get randomization parameters from Sim::Config
    uint32_t min_ants = sim.config.minAntsRand;
    uint32_t max_ants = sim.config.maxAntsRand;
    
    // Randomly decide how many ants to spawn
    uint32_t num_ants = ctx.randInt(min_ants, max_ants);
    
    // Generate that many ants
    for (uint32_t i = 0; i < num_ants; i++) {
        // Create ant entity
        // ...
    }
    
    // Similar code for movable objects and walls
    // ...
}
With these changes, you should be able to control the randomization parameters for the number of ants, movable objects, and walls from your Python training script using command line arguments.

The key is establishing the complete path from Python script → SimManager → Manager::Config → Sim::Config → Level generation system, ensuring that the parameters flow through each stage correctly.