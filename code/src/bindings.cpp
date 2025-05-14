#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

namespace nb = nanobind;

namespace madEscape {

// This file creates the python bindings used by the learning code.
// Refer to the nanobind documentation for more details on these functions.
NB_MODULE(madrona_escape_room, m) {
    // Each simulator has a madrona submodule that includes base types
    // like madrona::py::Tensor and madrona::py::PyExecMode.
    madrona::py::setupMadronaSubmodule(m);

    nb::class_<Manager> (m, "SimManager")
    // should match Manager::Config
        .def("__init__", [](Manager *self,
                            madrona::py::PyExecMode exec_mode,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            int64_t min_ants_rand,
                            int64_t max_ants_rand,
                            int64_t min_movable_objects_rand,
                            int64_t max_movable_objects_rand,
                            int64_t min_walls_rand,
                            int64_t max_walls_rand,
                            int64_t rand_seed,
                            bool auto_reset,
                            bool enable_batch_renderer) {
            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .randSeed = (uint32_t)rand_seed,
                .autoReset = auto_reset,
                .minAntsRand = (uint32_t)min_ants_rand,
                .maxAntsRand = (uint32_t)max_ants_rand,
                .minMovableObjectsRand = (uint32_t)min_movable_objects_rand,
                .maxMovableObjectsRand = (uint32_t)max_movable_objects_rand,
                .minWallsRand = (uint32_t)min_walls_rand,
                .maxWallsRand = (uint32_t)max_walls_rand,
                .enableBatchRenderer = enable_batch_renderer,
            });
        }, nb::arg("exec_mode"),
           nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("min_ants_rand"),
           nb::arg("max_ants_rand"),
           nb::arg("min_movable_objects_rand"),
           nb::arg("max_movable_objects_rand"),
           nb::arg("min_walls_rand"),
           nb::arg("max_walls_rand"),
           nb::arg("rand_seed"),
           nb::arg("auto_reset"),
           nb::arg("enable_batch_renderer") = false)
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("observation_tensor", &Manager::observationTensor)
        .def("lidar_tensor", &Manager::lidarTensor)
        .def("ant_count_tensor", &Manager::antCountTensor)
        .def("steps_remaining_tensor", &Manager::stepsRemainingTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
        .def("depth_tensor", &Manager::depthTensor)
    ;
}

}
