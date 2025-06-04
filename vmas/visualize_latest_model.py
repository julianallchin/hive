# record_trained_model_video.py

import torch
import time
import pyglet # Still needed for headless rendering context
import argparse # For command-line arguments
import random # For random seed
import os
import glob
from datetime import datetime

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from vmas.simulator.utils import save_video # VMAS utility for saving videos

# Import your custom scenario
from scenerio import Scenario as MyCustomScenario # Ensure this path is correct

# --- Configuration ---
# MODEL_PATH will be a command-line argument
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VMAS_DEVICE = DEVICE

# Environment parameters (should match training for model compatibility)
# These can also be made command-line args if they vary often
MAX_STEPS_PER_EPISODE = 200
SCENARIO_N_AGENTS = 2
SCENARIO_N_PACKAGES = 1
SCENARIO_N_OBSTACLES = 3
# ... add other necessary scenario kwargs

# --- Helper functions ---
def find_latest_model(model_type="policy", save_dir="saved_models"):
    """Find the latest saved model of the specified type (policy or critic).
    
    Args:
        model_type: Either 'policy' or 'critic'
        save_dir: Directory where models are saved
        
    Returns:
        Path to the latest model file or None if no models found
    """
    if not os.path.exists(save_dir):
        print(f"Error: Save directory '{save_dir}' does not exist.")
        return None
        
    # Find all model files of the specified type
    pattern = os.path.join(save_dir, f"{model_type}_iter_*.pt")
    model_files = glob.glob(pattern)
    
    if not model_files:
        print(f"No {model_type} models found in {save_dir}")
        return None
        
    # Sort by iteration and timestamp (both embedded in filename)
    # Format is: policy_iter_X_YYYYMMDD_HHMMSS.pt
    def extract_info(filename):
        base = os.path.basename(filename)
        parts = base.split('_')
        try:
            # Extract iteration number
            iter_num = int(parts[2])
            # Extract timestamp (if available)
            if len(parts) >= 4:
                timestamp = '_'.join(parts[3:])[:-3]  # Remove .pt
                return (iter_num, timestamp)
            return (iter_num, '')
        except (IndexError, ValueError):
            return (0, '')
    
    # Sort by iteration first, then by timestamp
    latest_model = sorted(model_files, key=extract_info, reverse=True)[0]
    
    print(f"Found latest {model_type} model: {os.path.basename(latest_model)}")
    return latest_model

# --- Helper function to re-create the policy architecture ---
def create_policy(env_for_spec, share_policy_params=True): # Added share_policy_params
    policy_net_view = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env_for_spec.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env_for_spec.action_spec["agents", "action"].shape[-1],
            n_agents=env_for_spec.n_agents,
            centralised=False,
            share_params=share_policy_params, # Use passed argument
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
        return_log_prob=False,
    )
    return policy_view

# --- Main Recording Function ---
def record_video(model_path: str, total_render_steps: int, output_filename: str, seed: int = None, share_policy_params: bool = True):
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        print(f"Using random seed: {seed}")
    else:
        print("Using a random seed for environment initialization.")


    # 1. Setup virtual display for headless rendering (important!)
    _display = None
    try:
        import pyvirtualdisplay
        _display = pyvirtualdisplay.Display(visible=False, size=(1000, 800)) # Adjust size as needed
        _display.start()
        print("Virtual display started for headless rendering.")
    except ImportError:
        print("pyvirtualdisplay not found. Headless rendering might fail or produce empty videos.")
        print("Please install it: pip install pyvirtualdisplay")
    except Exception as e:
        print(f"Could not start virtual display: {e}")
        # Proceeding without it, might work if X server is available but not ideal for headless

    # 2. Create the VMAS environment (single instance)
    env = VmasEnv(
        scenario=MyCustomScenario(),
        num_envs=1,
        continuous_actions=True,
        max_steps=MAX_STEPS_PER_EPISODE, # Episode will reset if it hits this
        device=VMAS_DEVICE,
        seed=seed, # Pass seed to VMAS environment
        # Scenario kwargs
        n_agents=SCENARIO_N_AGENTS,
        n_packages=SCENARIO_N_PACKAGES,
        n_obstacles=SCENARIO_N_OBSTACLES,
    )
    print("Environment created.")

    # 3. Load the trained policy
    policy = create_policy(env, share_policy_params=share_policy_params).to(DEVICE) # Pass share_policy_params
    try:
        print(f"Loading policy from: {model_path}")
        policy.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("This could be due to a mismatch between the saved model architecture and the current one.")
        print("Check if share_policy_params matches the training configuration.")
        if _display: _display.stop()
        return
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        if _display: _display.stop()
        return


    policy.eval()
    print(f"Policy loaded from {model_path} and set to eval mode.")

    # 4. Initialize and run the loop for recording
    td = env.reset()
    td = td.to(DEVICE)

    frames = []
    print(f"Recording {total_render_steps} steps to {output_filename}...")

    # Initial render to ensure viewer is created before the loop (good practice)
    # Even for rgb_array, some internal viewer setup might happen.
    try:
        env.render(mode="rgb_array", visualize_when_rgb=False)
    except Exception as e:
        print(f"Initial render call failed: {e}. This might indicate issues with graphics setup.")


    start_time = time.time()
    for step in range(total_render_steps):
        with torch.no_grad():
            td_action = policy(td)

        td = env.step(td_action)
        td = td.to(DEVICE)

        try:
            frame = env.render(mode="rgb_array", visualize_when_rgb=False) # Get frame as numpy array
            if frame is not None:
                frames.append(frame)
            else:
                print(f"Warning: Received None frame at step {step+1}")
        except Exception as e:
            print(f"Error during rendering at step {step+1}: {e}")
            # Optionally break or continue, depending on desired robustness
            break


        if (step + 1) % 100 == 0:
            print(f"Rendered step {step + 1}/{total_render_steps}")

        if td["done"].any() or td.get("terminated", td["done"]).any():
            print(f"Episode finished at step {step + 1}. Resetting environment...")
            td = env.reset()
            td = td.to(DEVICE)
            # If an episode finishes early, we still continue rendering until total_render_steps

    end_time = time.time()
    print(f"Finished rendering {len(frames)} frames in {end_time - start_time:.2f} seconds.")

    # 5. Save the video
    if frames:
        try:
            # VMAS save_video expects a list of numpy arrays (frames)
            # and the delta_t (dt) of the simulation for correct FPS.
            # env.unwrapped.world.dt should give the simulation timestep.
            simulation_dt = env.unwrapped.world.dt if hasattr(env.unwrapped, 'world') else 0.1 # Default if not found
            save_video(
                video_name=output_filename,
                frames=frames,
                fps=1.0 / simulation_dt
            )
            print(f"Video saved as {output_filename}")
        except Exception as e:
            print(f"Error saving video: {e}")
    else:
        print("No frames were collected, video not saved.")

    # 6. Clean up
    if hasattr(env, 'close'):
        env.close()
    if _display and _display.is_alive():
        _display.stop()
        print("Virtual display stopped.")
    print("Recording process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a video of a trained VMAS model.")
    parser.add_argument("--model_path", type=str, default=None, 
                        help="Path to the trained policy .pth file. If not provided, the latest model will be used.")
    parser.add_argument("--render_steps", type=int, default=500,
                        help="Total number of simulation steps to render.")
    parser.add_argument("--output_filename", type=str, default="trained_model_render.mp4", 
                        help="Output video filename (e.g., my_video.mp4).")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed for environment initialization.")
    parser.add_argument("--save_dir", type=str, default="saved_models",
                        help="Directory where models are saved.")
    parser.add_argument(
        "--no_share_policy_params",
        action="store_false", # Default is True (shared)
        dest="share_policy_params",
        help="Set if the loaded policy was trained WITHOUT parameter sharing.",
    )
    args = parser.parse_args()
    
    # If no model path is provided, find the latest model
    if args.model_path is None:
        model_path = find_latest_model(model_type="policy", save_dir=args.save_dir)
        if model_path is None:
            print("Error: Could not find any saved models. Please specify a model path.")
            exit(1)
    else:
        model_path = args.model_path
        
    # Generate a timestamp-based output filename if using the default
    if args.output_filename == "trained_model_render.mp4":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_basename = os.path.basename(model_path).replace(".pt", "")
        args.output_filename = f"render_{model_basename}_{timestamp}.mp4"

    record_video(model_path, args.render_steps, args.output_filename, args.seed, args.share_policy_params)