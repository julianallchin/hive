#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>
#include <madrona/window.hpp>

#include "sim.hpp"
#include "mgr.hpp"
#include "types.hpp"
#include "consts.hpp"

#include <filesystem>
#include <fstream>

using namespace madrona;
using namespace madrona::viz;

static HeapArray<int32_t> readReplayLog(const char *path)
{
    std::ifstream replay_log(path, std::ios::binary);
    replay_log.seekg(0, std::ios::end);
    int64_t size = replay_log.tellg();
    replay_log.seekg(0, std::ios::beg);

    HeapArray<int32_t> log(size / sizeof(int32_t));

    replay_log.read((char *)log.data(), (size / sizeof(int32_t)) * sizeof(int32_t));

    return log;
}

int main(int argc, char *argv[])
{
    using namespace madEscape;

    // Read command line arguments
    uint32_t num_ants = 1;
    if (argc >= 2) {
        num_ants = (uint32_t)atoi(argv[1]);
    }
    
    uint32_t num_worlds = 1;
    if (argc >= 3) {
        num_worlds = (uint32_t)atoi(argv[2]);
    }

    ExecMode exec_mode = ExecMode::CPU;
    if (argc >= 4) {
        if (!strcmp("--cpu", argv[3])) {
            exec_mode = ExecMode::CPU;
        } else if (!strcmp("--cuda", argv[3])) {
            exec_mode = ExecMode::CUDA;
        }
    }

    // Setup replay log
    const char *replay_log_path = nullptr;
    if (argc >= 5) {
        replay_log_path = argv[4];
    }

    auto replay_log = Optional<HeapArray<int32_t>>::none();
    uint32_t cur_replay_step = 0;
    uint32_t num_replay_steps = 0;
    if (replay_log_path != nullptr) {
        replay_log = readReplayLog(replay_log_path);
        num_replay_steps = replay_log->size() / (num_worlds * consts::maxAnts * 4);
    }

    bool enable_batch_renderer =
#ifdef MADRONA_MACOS
        false;
#else
        true;
#endif

    WindowManager wm {};
    WindowHandle window = wm.makeWindow("Hive", 2730, 1536);
    render::GPUHandle render_gpu = wm.initGPU(0, { window.get() });

    // Create the simulation manager
    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .randSeed = 5,
        // .autoReset = replay_log.has_value(),
        .autoReset = true,

        .minAntsRand = num_ants,
        .maxAntsRand = num_ants,
        .minMovableObjectsRand = 1,
        .maxMovableObjectsRand = 5,
        .minWallsRand = 1,
        .maxWallsRand = 5,

        .enableBatchRenderer = enable_batch_renderer,
        .extRenderAPI = wm.gpuAPIManager().backend(),
        .extRenderDev = render_gpu.device(),
    });

    float camera_move_speed = 30.f;
    float camera_height = (consts::worldWidth + consts::worldLength) / 2.0f;
    math::Vector3 initial_camera_position = { 0, 0, camera_height};

    // Top-down view for the ant colony
    math::Quat initial_camera_rotation =
        math::Quat::angleAxis(-math::pi / 2.f, math::right).normalize();


    // Create the viewer viewer
    viz::Viewer viewer(mgr.getRenderManager(), window.get(), {
        .numWorlds = num_worlds,
        .simTickRate = 20,
        .cameraMoveSpeed = camera_move_speed,
        .cameraPosition = initial_camera_position,
        .cameraRotation = initial_camera_rotation,
    });

    // Replay step for ant colony
    auto replayStep = [&]() {
        if (cur_replay_step == num_replay_steps - 1) {
            return true;
        }

        printf("Step: %u\n", cur_replay_step);
        
        for (uint32_t i = 0; i < num_worlds; i++) {
            
            assert(num_ants > 0);
            
            for (uint32_t j = 0; j < (uint32_t)num_ants; j++) {
                uint32_t base_idx = 0;
                base_idx = 4 * (cur_replay_step * consts::maxAnts * num_worlds +
                    i * consts::maxAnts + j);

                int32_t move_amount = (*replay_log)[base_idx];
                int32_t move_angle = (*replay_log)[base_idx + 1];
                int32_t turn = (*replay_log)[base_idx + 2];
                int32_t g = (*replay_log)[base_idx + 3];

                printf("World %d, Ant %d: move=%d angle=%d turn=%d grab=%d\n",
                       i, j, move_amount, move_angle, turn, g);
                mgr.setAction(i, j, move_amount, move_angle, turn, g);
            }
        }

        cur_replay_step++;

        return false;
    };

    // Printers for ant colony simulation
    auto ant_printer = mgr.observationTensor().makePrinter();
    auto ant_count_printer = mgr.numAntsTensor().makePrinter();
    auto lidar_printer = mgr.lidarTensor().makePrinter();
    auto steps_remaining_printer = mgr.stepsRemainingTensor().makePrinter();
    auto reward_printer = mgr.rewardTensor().makePrinter();

    auto printObs = [&]() {
        printf("Ant Observations\n");
        ant_printer.print();

        printf("Ant Count\n");
        ant_count_printer.print();

        printf("Lidar\n");
        lidar_printer.print();

        printf("Steps Remaining\n");
        steps_remaining_printer.print();

        printf("Hive Reward\n");
        reward_printer.print();

        printf("\n");
    };
    (void)printObs;

    // Main loop for the viewer viewer
    viewer.loop(
    [&mgr](CountT world_idx, const Viewer::UserInput &input)
    {
        using Key = Viewer::KeyboardKey;
        if (input.keyHit(Key::R)) {
            // Reset the current world
            mgr.triggerReset(world_idx);
        }
    },
    [&mgr](CountT world_idx, CountT ant_idx,
           const Viewer::UserInput &input)
    {
        using Key = Viewer::KeyboardKey;

        int32_t x = 0;
        int32_t y = 0;
        int32_t r = 2;
        int32_t g = 0;

        bool shift_pressed = input.keyPressed(Key::Shift);

        if (input.keyPressed(Key::W)) {
            y += 1;
        }
        if (input.keyPressed(Key::S)) {
            y -= 1;
        }

        if (input.keyPressed(Key::D)) {
            x += 1;
        }
        if (input.keyPressed(Key::A)) {
            x -= 1;
        }

        if (input.keyPressed(Key::Q)) {
            r += shift_pressed ? 2 : 1;
        }
        if (input.keyPressed(Key::E)) {
            r -= shift_pressed ? 2 : 1;
        }

        if (input.keyHit(Key::G)) {
            g = 1;
        }

        int32_t move_amount;
        if (x == 0 && y == 0) {
            move_amount = 0;
        } else if (shift_pressed) {
            move_amount = consts::numMoveAmountBuckets - 1;
        } else {
            move_amount = 1;
        }

        int32_t move_angle;
        if (x == 0 && y == 1) {
            move_angle = 0;
        } else if (x == 1 && y == 1) {
            move_angle = 1;
        } else if (x == 1 && y == 0) {
            move_angle = 2;
        } else if (x == 1 && y == -1) {
            move_angle = 3;
        } else if (x == 0 && y == -1) {
            move_angle = 4;
        } else if (x == -1 && y == -1) {
            move_angle = 5;
        } else if (x == -1 && y == 0) {
            move_angle = 6;
        } else if (x == -1 && y == 1) {
            move_angle = 7;
        } else {
            move_angle = 0;
        }

        // Set action for this individual ant
        mgr.setAction(world_idx, ant_idx, move_amount, move_angle, r, g);
    }, [&]() {
        if (replay_log.has_value()) {
            bool replay_finished = replayStep();

            if (replay_finished) {
                viewer.stopLoop();
            }
        }

        // Step the ant colony simulation
        mgr.step();
        
        // Uncomment to see ant observations and rewards during simulation
        printObs();
    }, []() {});
}
