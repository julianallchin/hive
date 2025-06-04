import torch
import time
import os
import datetime

# TorchRL imports
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Import your custom scenario
from scenerio import Scenario as MyCustomScenario # Ensure 'scenerio.py' is the correct filename

# For MP4 generation - much faster than GIF
import cv2
import numpy as np

# Optional: For headless rendering
try:
    import pyvirtualdisplay
    _display = None
except ImportError:
    _display = None
    print("pyvirtualdisplay not found. If on a headless server, Xvfb might be needed directly or rendering might fail.")


# --- Configuration ---
POLICY_MODEL_PATH = None # Will be set by get_latest_model_path
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
VMAS_DEVICE = DEVICE # Use same device for VMAS

# MP4 Generation Parameters
MAX_STEPS_FOR_VIDEO = 600 # Number of environment steps to record for the video
VIDEO_FPS = 40 # Frames per second for the output video
OUTPUT_DIR = "videos"  # Directory to save videos
# Filename will be generated with current date

# Environment and Model parameters (MUST MATCH TRAINING CONFIGURATION)
MAX_STEPS_PER_EPISODE = 200 # Should match training, used for env setup
SCENARIO_N_AGENTS = 2
SCENARIO_N_PACKAGES = 1
SCENARIO_N_OBSTACLES = 3
SCENARIO_LIDAR_RANGE = 0.5
# ... add any other parameters your scenario's make_world expects

# Policy architecture parameters (MUST MATCH TRAINING)
SHARE_PARAMETERS_POLICY = True
POLICY_DEPTH = 2
POLICY_NUM_CELLS = 256
POLICY_ACTIVATION_CLASS = torch.nn.Tanh

# --- Helper Function to get latest model ---
def get_latest_model_path(save_dir="saved_models", model_prefix="policy_iter_"):
    if not os.path.isdir(save_dir):
        print(f"Error: Save directory '{save_dir}' not found.")
        return None
    models = [f for f in os.listdir(save_dir) if f.startswith(model_prefix) and f.endswith(".pt")]
    if not models:
        print(f"Error: No models found in '{save_dir}' with prefix '{model_prefix}'.")
        return None
    def sort_key(filename):
        parts = filename.replace(model_prefix, "").replace(".pt", "").split("_")
        try:
            iteration = int(parts[0])
            timestamp_str = "_".join(parts[1:])
            return (iteration, timestamp_str)
        except (ValueError, IndexError):
            print(f"Warning: Could not parse iteration/timestamp from {filename}, placing it last.")
            return (-1, filename)
    models.sort(key=sort_key, reverse=True)
    if not models or sort_key(models[0])[0] == -1:
        print(f"Error: No validly named models found in '{save_dir}' with prefix '{model_prefix}'.")
        return None
    latest_model = models[0]
    print(f"Found latest model: {latest_model}")
    return os.path.join(save_dir, latest_model)

if POLICY_MODEL_PATH is None:
    POLICY_MODEL_PATH = get_latest_model_path()

if not POLICY_MODEL_PATH or not os.path.exists(POLICY_MODEL_PATH):
    print(f"Error: Policy model path not set or file not found: {POLICY_MODEL_PATH}")
    exit()

# --- Optional: Start virtual display for headless environments ---
if os.environ.get("DISPLAY") is None and _display is None and 'pyvirtualdisplay' in globals():
    try:
        print("Attempting to start virtual display for headless video generation...")
        _display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
        _display.start()
        print("Virtual display started.")
    except Exception as e:
        print(f"Could not start virtual display: {e}. Video generation might fail if no display is available.")
        _display = None


# --- Environment Setup ---
# For video generation, a single environment instance is usually best.
env = VmasEnv(
    scenario=MyCustomScenario(),
    num_envs=1, # Single environment for video
    continuous_actions=True,
    max_steps=MAX_STEPS_PER_EPISODE, # This still defines episode length
    device=VMAS_DEVICE,
    n_agents=SCENARIO_N_AGENTS,
    n_packages=SCENARIO_N_PACKAGES,
    n_obstacles=SCENARIO_N_OBSTACLES,
    lidar_range=SCENARIO_LIDAR_RANGE,
    # ... add other kwargs as needed
    seed=np.random.randint(0, 10000) # Use a random seed for different video runs, or fix for consistency
)
print("Environment instantiated for video generation.")

# --- Policy Network Instantiation ---
policy_net_view = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=2 * env.action_spec["agents", "action"].shape[-1],
        n_agents=env.n_agents,
        centralised=False,
        share_params=SHARE_PARAMETERS_POLICY,
        device=DEVICE,
        depth=POLICY_DEPTH,
        num_cells=POLICY_NUM_CELLS,
        activation_class=POLICY_ACTIVATION_CLASS,
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
    spec=env.unbatched_action_spec, # Should be correct for num_envs=1
    in_keys=[("agents", "loc"), ("agents", "scale")],
    out_keys=[env.action_key],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.unbatched_action_spec[env.action_key].space.low,
        "high": env.unbatched_action_spec[env.action_key].space.high,
    },
    return_log_prob=False,
).to(DEVICE)
print("Policy architecture created.")

# --- Load Saved Model Weights ---
try:
    policy_view.load_state_dict(torch.load(POLICY_MODEL_PATH, map_location=DEVICE))
    policy_view.eval() # Set the model to evaluation mode
    print(f"Successfully loaded policy weights from: {POLICY_MODEL_PATH}")
except Exception as e:
    print(f"Error loading policy weights: {e}")
    if _display: _display.stop()
    exit()

# --- MP4 Generation Setup ---
print(f"\nGenerating MP4 video for {MAX_STEPS_FOR_VIDEO} steps...")

# Initialize video writer (will be set up after first frame)
video_writer = None
frame_width = None
frame_height = None
output_video_path = None

def setup_video_writer(frame):
    """Setup video writer with frame dimensions"""
    global video_writer, frame_width, frame_height, output_video_path
    frame_height, frame_width = frame.shape[:2]
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    # Generate filename with current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    output_filename = f"trained_policy_{current_date}.mp4"
    output_video_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Use MP4V codec for fast encoding
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_video_path, 
        fourcc, 
        VIDEO_FPS, 
        (frame_width, frame_height)
    )
    print(f"Video writer initialized: {frame_width}x{frame_height} at {VIDEO_FPS} FPS")
    print(f"Output will be saved to: {output_video_path}")

def rendering_rollout_callback(env_instance, current_td):
    """Callback to render a frame and write it directly to video file."""
    global video_writer
    
    # env_instance is the VmasEnv here
    # current_td is the TensorDict at the current step of the rollout
    frame = env_instance.render(mode="rgb_array")  # Render as numpy array
    
    # Setup video writer on first frame
    if video_writer is None:
        setup_video_writer(frame)
    
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Write frame directly to video file
    video_writer.write(frame_bgr)

try:
    start_time = time.time()
    
    with torch.no_grad(): # Ensure no gradients are computed
        # env.rollout will handle resetting the environment if episodes end within MAX_STEPS_FOR_VIDEO
        env.rollout(
            max_steps=MAX_STEPS_FOR_VIDEO,
            policy=policy_view,
            callback=rendering_rollout_callback,
            auto_cast_to_device=True, # Moves data to policy's device
            break_when_any_done=False # Continue rollout for MAX_STEPS_FOR_VIDEO regardless of early episode ends
        )

    # Finalize video
    if video_writer is not None:
        video_writer.release()
        end_time = time.time()
        print(f"MP4 video saved as {output_video_path}")
        print(f"Video generation completed in {end_time - start_time:.2f} seconds")
    else:
        print("No frames were collected. Video not created.")

except Exception as e:
    import traceback
    print("An error occurred during video generation:")
    traceback.print_exc()
finally:
    # Clean up
    if video_writer is not None:
        video_writer.release()
    env.close()
    if _display:
        _display.stop()
        print("Virtual display stopped.")
    print("Environment closed.")