# train_custom_scenario.py

import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from tqdm import tqdm
from matplotlib import pyplot as plt

# Import your custom scenario
from scenerio import Scenario as MyCustomScenario # Make sure this path is correct

# --- Hyperparameters ---
# Devices
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
vmas_device = device

# Sampling
frames_per_batch = 6_000
n_iters = 200  # Increase for longer training
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 10  # Number of optimization steps per training iteration
minibatch_size = 600  # Size of the mini-batches in each optimization step
lr = 3e-4
max_grad_norm = 1.0

# PPO
clip_epsilon = 0.2
gamma = 0.99
lmbda = 0.9
entropy_eps = 1e-4 # Can be tuned, 0.01 is also common

# --- Environment Specific Parameters for your Custom Scenario ---
# These MUST match the kwargs your scenario's make_world expects,
# and how you want to configure the environment for training.
max_steps = 200  # Episode steps before done (VMAS max_steps)
num_vmas_envs = frames_per_batch // max_steps

# Your custom scenario's parameters:
scenario_n_agents = 2 # As used in your __main__ example
scenario_n_packages = 1
scenario_n_obstacles = 3
scenario_lidar_range = 0.5
# Add other parameters your scenario might need, e.g.,
# package_width, package_length, package_mass, obstacle_radius,
# n_lidar_rays_agents, n_lidar_rays_packages, n_lidar_rays_obstacles

torch.manual_seed(0)

# --- Environment Setup ---
env = VmasEnv(
    scenario=MyCustomScenario(), # Pass an instance of your scenario class
    num_envs=num_vmas_envs,
    continuous_actions=True,
    max_steps=max_steps,
    device=vmas_device,
    # Scenario kwargs for your custom scenario
    n_agents=scenario_n_agents,
    n_packages=scenario_n_packages,
    n_obstacles=scenario_n_obstacles,
    # Add other kwargs as needed by your Scenario's make_world
    # e.g., package_width=0.15, n_lidar_rays_agents=16, etc.
)

env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
)

print("Checking environment specs...")
check_env_specs(env)
print("Specs check passed.")

print("action_spec:", env.full_action_spec)
print("reward_spec:", env.full_reward_spec)
print("done_spec:", env.full_done_spec)
print("observation_spec:", env.observation_spec)
print("action_keys:", env.action_keys)
print("reward_keys:", env.reward_keys)

exit()

# --- Policy Network ---
share_parameters_policy = True # Or False

# The input for each agent is its own observation
# The output for each agent is 2 * number_of_action_dims (for loc and scale of TanhNormal)
policy_net = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=2 * env.action_spec["agents", "action"].shape[-1],
        n_agents=env.n_agents,
        centralised=False,
        share_params=share_parameters_policy,
        device=device,
        depth=2,
        num_cells=256, # Hidden layer size
        activation_class=torch.nn.Tanh,
    ),
    NormalParamExtractor(),
)

policy_module = TensorDictModule(
    policy_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "loc"), ("agents", "scale")],
)

policy = ProbabilisticActor(
    module=policy_module,
    spec=env.unbatched_action_spec, # Use unbatched spec for ProbabilisticActor
    in_keys=[("agents", "loc"), ("agents", "scale")],
    out_keys=[env.action_key], # This is typically ("agents", "action")
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.unbatched_action_spec[env.action_key].space.low,
        "high": env.unbatched_action_spec[env.action_key].space.high,
    },
    return_log_prob=True,
    log_prob_key=("agents", "sample_log_prob"),
)

# --- Critic Network ---
share_parameters_critic = True # Or False
mappo = True  # If True, critic is centralized (MAPPO-style). If False, decentralized (IPPO-style).

critic_net = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
    n_agent_outputs=1,  # 1 value per agent
    n_agents=env.n_agents,
    centralised=mappo,
    share_params=share_parameters_critic,
    device=device,
    depth=2,
    num_cells=256, # Hidden layer size
    activation_class=torch.nn.Tanh,
)

critic = TensorDictModule(
    module=critic_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "state_value")],
)

print("Policy and Critic instantiated.")
# print("Running policy on dummy reset data:", policy(env.reset()))
# print("Running critic on dummy reset data:", critic(env.reset()))


# --- Data Collector ---
collector = SyncDataCollector(
    env,
    policy,
    device=vmas_device, # Device for env and policy during collection
    storing_device=device, # Device where data is stored
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)
print("Collector instantiated.")

# --- Replay Buffer ---
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch, device=device),
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,
)
print("Replay buffer instantiated.")

# --- Loss Function ---
loss_module = ClipPPOLoss(
    actor_network=policy,
    critic_network=critic,
    clip_epsilon=clip_epsilon,
    entropy_coef=entropy_eps,
    normalize_advantage=False,
)

# Crucially, set the keys for the loss module based on your environment's structure
# VmasEnv by default sets env.reward_key to ("agents", "reward") and
# env.action_key to ("agents", "action").
# The sample_log_prob and state_value keys are defined by our policy and critic.
loss_module.set_keys(
    reward=env.reward_key,
    action=env.action_key,
    sample_log_prob=("agents", "sample_log_prob"),
    value=("agents", "state_value"),
    # done and terminated keys will be expanded to match reward shape later
    done=("agents", "done"), # This will be created before GAE
    terminated=("agents", "terminated"), # This will be created before GAE
)

loss_module.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
)
GAE = loss_module.value_estimator
print("Loss module and GAE instantiated.")

optim = torch.optim.Adam(loss_module.parameters(), lr)
print("Optimizer instantiated.")

# --- Training Loop ---
pbar = tqdm(total=total_frames)
episode_reward_mean_list = []
eval_interval = frames_per_batch * 5 # Evaluate every 5 collection iterations
frames_since_last_eval = 0

for i, tensordict_data in enumerate(collector):
    pbar.update(tensordict_data.numel())
    frames_since_last_eval += tensordict_data.numel()

    # Ensure done and terminated are expanded to per-agent shape for GAE
    # VmasEnv done/terminated are global, GAE expects per-agent for value target computation
    for key in ["done", "terminated"]:
        tensordict_data.set(
            ("next", "agents", key),
            tensordict_data.get(("next", key))
            .unsqueeze(-1) # Add agent dimension
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
            # Expand to match num_agents in reward_key shape
            inplace=False, # Added inplace=False for clarity, set returns new tensordict
        )


    with torch.no_grad():
        GAE(
            tensordict_data,
            params=loss_module.critic_network_params, # Pass the parameters
            target_params=loss_module.target_critic_network_params # Pass the target parameters
        )

    data_view = tensordict_data.reshape(-1) # Flatten num_envs and time steps
    replay_buffer.extend(data_view)

    for _ in range(num_epochs):
        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)

            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    collector.update_policy_weights_() # Update policy weights in the collector

    # Logging - Calculate mean episode reward from collected batch
    # episode_reward is a sum over the episode (due to RewardSum transform)
    # We look at 'done' in the 'next' state to identify episode ends
    done_in_batch = tensordict_data.get(("next", "done"))
    if done_in_batch.any():
        # Assuming episode_reward key is ("agents", "episode_reward")
        # Take the mean across agents first, then across finished episodes
        episode_rewards_at_done = tensordict_data.get(
            ("next", "agents", "episode_reward")
        )[done_in_batch.squeeze(-1)].mean() # mean over agents and then over envs
        episode_reward_mean_list.append(episode_rewards_at_done.item())
        pbar.set_description(f"Iter {i+1} | Episode Reward Mean: {episode_rewards_at_done.item():.2f}")
    else:
        pbar.set_description(f"Iter {i+1} | No episodes finished in this batch")


    # Simple evaluation (optional, more sophisticated evaluation can be added)
    if frames_since_last_eval >= eval_interval:
        frames_since_last_eval = 0
        print(f"\n--- Evaluating at iteration {i+1} ---")
        with torch.no_grad():
            eval_rewards = []
            # Use a temporary evaluation environment or reset the main one carefully
            # For simplicity, we'll just do a short rollout with the current policy
            eval_rollout = env.rollout(
                max_steps=max_steps, # Same as training episode length
                policy=policy,
                auto_cast_to_device=True, # Ensure policy runs on its device
                break_when_any_done=False # Collect full episodes
            )
            eval_done = eval_rollout.get(("next", "done"))
            if eval_done.any():
                eval_ep_rewards = eval_rollout.get(("next", "agents", "episode_reward"))[eval_done.squeeze(-1)].mean()
                print(f"Evaluation Episode Reward Mean: {eval_ep_rewards.item():.2f}")
            else:
                print("No full episodes completed during evaluation rollout.")
        print("--- End Evaluation ---")


pbar.close()
print("Training finished.")

# --- Plotting Results ---
if episode_reward_mean_list:
    plt.figure()
    plt.plot(episode_reward_mean_list)
    plt.xlabel("Training Iterations (where episodes ended)")
    plt.ylabel("Mean Episode Reward")
    plt.title("Training Progress for Custom VMAS Scenario")
    plt.savefig("custom_scenario_training_rewards.png")
    plt.show()
    print("Plot saved to custom_scenario_training_rewards.png")
else:
    print("No episodes finished during training, so no reward plot.")


# --- Optional: Render a few episodes with the trained policy ---
# Make sure you have rendering dependencies installed (pyglet, xvfb on headless, etc.)
# This part might require a display or virtual display setup (e.g., in Colab)

# print("\nRendering a few episodes with the trained policy...")
# try:
#     import pyvirtualdisplay
#     _display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
#     _display.start()
#     print("Virtual display started.")
# except ImportError:
#     print("pyvirtualdisplay not found. Rendering might not work on headless server without it.")
# except Exception as e:
#     print(f"Could not start virtual display: {e}")


# frames = []
# def render_callback(env, td):
#     frame = env.render(mode="rgb_array") # Ensure your VMAS env can render
#     frames.append(frame)

# with torch.no_grad():
#     # Reset env for rendering
#     env.reset()
#     env.rollout(
#         max_steps=max_steps * 3, # Render 3 episodes
#         policy=policy,
#         callback=render_callback,
#         auto_cast_to_device=True,
#         break_when_any_done=False, # Let it run for max_steps * 3
#     )

# if frames:
#     from moviepy.editor import ImageSequenceClip
#     clip = ImageSequenceClip(frames, fps=30) # Adjust fps as needed
#     clip.write_gif("trained_custom_scenario.gif", fps=30)
#     print("Rendered GIF saved to trained_custom_scenario.gif")
# else:
#     print("No frames collected for rendering.")

# if '_display' in locals():
#     _display.stop()