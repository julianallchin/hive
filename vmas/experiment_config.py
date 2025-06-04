from benchmarl.experiment import ExperimentConfig
import torch

import copy
from typing import Callable, Optional
from benchmarl.environments import VmasTask
from benchmarl.utils import DEVICE_TYPING
from torchrl.envs import EnvBase, VmasEnv
from scenerio import Scenario


def get_env_fun(
    self,
    num_envs: int,
    continuous_actions: bool,
    seed: Optional[int],
    device: DEVICE_TYPING,
) -> Callable[[], EnvBase]:
    # config = copy.deepcopy(self.config)
    # if (hasattr(self, "name") and self.name is "NAVIGATION") or (
    #     self is VmasTask.NAVIGATION
    # ):  # This is the only modification we make ....
    #     scenario = Scenario()  # .... ends here
    # else:
    #     scenario = self.name.lower()
    config = copy.deepcopy(self.config)
    scenario = Scenario()
    return lambda: VmasEnv(
        scenario=scenario,
        num_envs=num_envs,
        continuous_actions=continuous_actions,
        seed=seed,
        device=device,
        categorical_actions=True,
        clamp_actions=True,
        **config,
    )

try:
    from benchmarl.environments import VmasClass
    VmasClass.get_env_fun = get_env_fun
except ImportError:
    VmasTask.get_env_fun = get_env_fun

# Loads from "benchmarl/conf/task/vmas/navigation.yaml"
task = VmasTask.NAVIGATION.get_from_yaml()

# We override the NAVIGATION config with ours
task.config = {
        "max_steps": 100,
        "n_agents_holonomic": 2,
        "n_agents_diff_drive": 1,
        "n_agents_car": 1,
        "lidar_range": 0.35,
        "comms_rendering_range": 0,
        "shared_rew": False,
}

# Loads from "benchmarl/conf/experiment/base_experiment.yaml"
experiment_config = ExperimentConfig.get_from_yaml() # We start by loading the defaults

# Override devices
experiment_config.sampling_device = "mps"
experiment_config.train_device = "mps"

experiment_config.max_n_frames = 10_000_000 # Number of frames before training ends
experiment_config.gamma = 0.99
experiment_config.on_policy_collected_frames_per_batch = 60_000 # Number of frames collected each iteration
experiment_config.on_policy_n_envs_per_worker = 600 # Number of vmas vectorized enviornemnts (each will collect 100 steps, see max_steps in task_config -> 600 * 100 = 60_000 the number above)
experiment_config.on_policy_n_minibatch_iters = 45
experiment_config.on_policy_minibatch_size = 4096
experiment_config.evaluation = True
experiment_config.render = True
experiment_config.share_policy_params = True # Policy parameter sharing on
experiment_config.evaluation_interval = 120_000 # Interval in terms of frames, will evaluate every 120_000 / 60_000 = 2 iterations
experiment_config.evaluation_episodes = 200 # Number of vmas vectorized enviornemnts used in evaluation
experiment_config.loggers = ["csv"] # Log to csv, usually you should use wandb



from benchmarl.algorithms import MappoConfig

# We can load from "benchmarl/conf/algorithm/mappo.yaml"
algorithm_config = MappoConfig.get_from_yaml()

# Or create it from scratch
algorithm_config = MappoConfig(
    share_param_critic=True, # Critic param sharing on
    clip_epsilon=0.2,
    entropy_coef=0.001, # We modify this, default is 0
    critic_coef=1,
    loss_critic_type="l2",
    lmbda=0.9,
    scale_mapping="biased_softplus_1.0", # Mapping for standard deviation
    use_tanh_normal=True,
    minibatch_advantage=False,
)

from benchmarl.models.mlp import MlpConfig

model_config = MlpConfig(
    num_cells=[256, 256], # Two layers with 256 neurons each
    layer_class=torch.nn.Linear,
    activation_class=torch.nn.Tanh,
)

# Loads from "benchmarl/conf/model/layers/mlp.yaml" (in this case we use the defaults so it is the same)
model_config = MlpConfig.get_from_yaml()
critic_model_config = MlpConfig.get_from_yaml()