#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>
#include <madrona/window.hpp>

#include "sim.hpp"
#include "mgr.hpp"
#include "types.hpp"

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

    // Dynamic number of ants, but start with a reasonable number for viewer
    constexpr int64_t default_num_ants = 8;

    // Read command line arguments
    uint32_t num_worlds = 1;
    if (argc >= 2) {
        num_worlds = (uint32_t)atoi(argv[1]);
    }

    ExecMode exec_mode = ExecMode::CPU;
    if (argc >= 3) {
        if (!strcmp("--cpu", argv[2])) {
            exec_mode = ExecMode::CPU;
        } else if (!strcmp("--cuda", argv[2])) {
            exec_mode = ExecMode::CUDA;
        }
    }

    // Setup replay log
    const char *replay_log_path = nullptr;
    if (argc >= 4) {
        replay_log_path = argv[3];
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
    WindowHandle window = wm.makeWindow("Ant Colony Simulation", 2730, 1536);
    render::GPUHandle render_gpu = wm.initGPU(0, { window.get() });

    // Create the simulation manager
    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .randSeed = 5,
        .autoReset = replay_log.has_value(),
        .enableBatchRenderer = enable_batch_renderer,
        .extRenderAPI = wm.gpuAPIManager().backend(),
        .extRenderDev = render_gpu.device(),
    });

    float camera_move_speed = 10.f;

    math::Vector3 initial_camera_position = { 0, 0, 40 };

    // Top-down view for the ant colony
    math::Quat initial_camera_rotation =
        math::Quat::angleAxis(-math::pi / 2.f, math::right).normalize();


    // Create the viewer viewer
    viz::Viewer viewer(mgr.getRenderManager(), window.get(), {
        .numWorlds = num_worlds,
        .simTickRate = 30,
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
            // Get the actual number of ants in this world from the ant count tensor
            int32_t* ant_counts = (int32_t*)mgr.antCountTensor().raw();
            int32_t num_ants = ant_counts[i];
            if (num_ants <= 0) num_ants = default_num_ants; // Fallback if tensor not initialized
            
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
    auto ant_printer = mgr.antObservationTensor().makePrinter();
    auto ant_count_printer = mgr.antCountTensor().makePrinter();
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
        //printObs();

        //printObs();
    }, []() {});
}
