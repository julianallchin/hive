from benchmarl.experiment import Experiment
from experiment_config import experiment_config, algorithm_config, model_config, critic_model_config,task

experiment_config.max_n_frames = 1000_000 # Runs one iteration, change to 50_000_000 for full training
# experiment_config.on_policy_n_envs_per_worker = 60 # Remove this line for full training
# experiment_config.on_policy_n_minibatch_iters = 1 # Remove this line for full training

experiment = Experiment(
    task=task,
    algorithm_config=algorithm_config,
    model_config=model_config,
    critic_model_config=critic_model_config,
    seed=0,
    config=experiment_config,
)
experiment.run()