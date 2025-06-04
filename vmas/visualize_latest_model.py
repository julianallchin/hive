# view_trained_model.py

import torch
import time
import pyglet # Important for keeping the window open and responsive

from tensordict.nn import TensorDictModule # For policy
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Import your custom scenario
from scenerio import Scenario as MyCustomScenario # Ensure this path is correct

# --- Configuration ---
import os
from glob import glob
from datetime import datetime

# Model loading configuration
SAVE_DIR = "saved_models"  # Should match the directory used in training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VMAS_DEVICE = DEVICE  # Run VMAS on the same device

def find_latest_model():
    """Find the most recently saved policy model."""
    policy_files = glob(os.path.join(SAVE_DIR, "policy_iter_*_*.pt"))
    if not policy_files:
        raise FileNotFoundError(f"No model files found in {SAVE_DIR}. Please train a model first.")
    
    # Sort by modification time (newest first)
    latest_file = max(policy_files, key=os.path.getmtime)
    # Get the corresponding critic file
    base_name = "_" + "_".join(os.path.basename(latest_file).split("_")[2:])
    critic_file = os.path.join(SAVE_DIR, f"critic{base_name}")
    
    if not os.path.exists(critic_file):
        print(f"Warning: Could not find corresponding critic file for {latest_file}")
    
    print(f"Found latest model: {os.path.basename(latest_file)}")
    print(f"Corresponding critic: {os.path.basename(critic_file) if os.path.exists(critic_file) else 'Not found'}")
    return latest_file

# Environment parameters (should match training for model compatibility)
MAX_STEPS_PER_EPISODE = 200 # Or whatever you used during training
SCENARIO_N_AGENTS = 2
SCENARIO_N_PACKAGES = 1
SCENARIO_N_OBSTACLES = 3

# --- Helper function to re-create the policy architecture ---
# This needs to exactly match the architecture used during training
def create_policy(env_for_spec):
    policy_net_view = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env_for_spec.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env_for_spec.action_spec["agents", "action"].shape[-1],
            n_agents=env_for_spec.n_agents,
            centralised=False,
            share_params=True, # IMPORTANT: Match your training setting
            device=DEVICE,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )
    policy_module_view = TensorDictModule(
        policy_net_view,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    policy_view = ProbabilisticActor(
        module=policy_module_view,
        spec=env_for_spec.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env_for_spec.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env_for_spec.unbatched_action_spec[env_for_spec.action_key].space.low,
            "high": env_for_spec.unbatched_action_spec[env_for_spec.action_key].space.high,
        },
        return_log_prob=False, # Not needed for inference
    )
    return policy_view

# --- Main Viewing Function ---
def view_model():
    # 1. Create the VMAS environment (single instance)
    # We use num_envs=1 for real-time viewing.
    env = VmasEnv(
        scenario=MyCustomScenario(),
        num_envs=1, # Single environment for viewing
        continuous_actions=True,
        max_steps=MAX_STEPS_PER_EPISODE,
        device=VMAS_DEVICE,
        # Scenario kwargs
        n_agents=SCENARIO_N_AGENTS,
        n_packages=SCENARIO_N_PACKAGES,
        n_obstacles=SCENARIO_N_OBSTACLES,
    )
    print("Environment created.")

    # 2. Find and load the latest trained policy
    try:
        latest_policy_path = find_latest_model()
        policy = create_policy(env).to(DEVICE)
        policy.load_state_dict(torch.load(latest_policy_path, map_location=DEVICE))
        print(f"Successfully loaded policy from {os.path.basename(latest_policy_path)}")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("This might be due to a mismatch in model architecture or if you saved the entire loss_module.")
        print("If you saved loss_module, you might need to load it and then access loss_module.actor_network.load_state_dict(...)")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    policy.eval() # Set to evaluation mode
    print(f"Policy loaded from {SAVE_DIR} and set to eval mode.")

    # 3. Initialize viewer and run the loop
    td = env.reset() # Get initial observation tensordict
    td = td.to(DEVICE)

    # Initial render to create the window
    # VMAS's VmasEnv implicitly creates/uses a global viewer when render is called.
    frame = env.render(mode="human") # "human" mode shows the window

    print("\nStarting real-time model viewer. Press Ctrl+C in terminal to stop.")
    print("Close the Pyglet window to stop.")

    try:
        while not env.unwrapped.viewer.window.has_exit: # Loop until window is closed
            with torch.no_grad():
                # Get action from policy
                # The policy expects a batch dimension, even if it's 1
                td_action = policy(td)

            # Step the environment
            # VmasEnv.step takes a tensordict of actions
            td = env.step(td_action)
            td = td.to(DEVICE) # Move next state to device for next policy input

            # Render
            env.render(mode="human")

            # Check for done state to reset
            # VmasEnv done is global: td["done"] has shape [1, 1]
            if td["done"].any() or td.get("terminated", td["done"]).any(): # also check for terminated if present
                print("Episode finished. Resetting environment...")
                td = env.reset()
                td = td.to(DEVICE)

            # Pyglet event handling to keep window responsive and allow closing
            pyglet.app.platform_event_loop.dispatch_posted_events()
            env.unwrapped.viewer.window.dispatch_events()

            time.sleep(0.03) # Adjust for desired frame rate (e.g., ~30 FPS)

    except KeyboardInterrupt:
        print("\nViewer stopped by user (Ctrl+C).")
    finally:
        if hasattr(env, 'close'): # VmasEnv should have a close method
            env.close()
        print("Environment closed.")
        # Explicitly exit pyglet app if it's still running
        if pyglet.app.event_loop is not None and pyglet.app.event_loop.is_running:
             pyglet.app.exit()


if __name__ == "__main__":
    # Ensure your custom scenario is importable
    # e.g., place my_custom_vmas_scenario.py in the same directory
    # or install your project if it's a package.

    # Optional: For headless servers or Colab, setup virtual display
    try:
        import pyvirtualdisplay
        _display = pyvirtualdisplay.Display(visible=False, size=(1000, 800))
        _display.start()
        print("Virtual display started (for headless compatibility).")
    except ImportError:
        print("pyvirtualdisplay not found. If on a headless server, rendering might fail or not be visible.")
    except Exception as e:
        print(f"Could not start virtual display: {e}")

    view_model()

    if '_display' in locals() and _display.is_alive():
        _display.stop()
        print("Virtual display stopped.")