# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import os
import sys
import random
import time
# import pdb
from dataclasses import dataclass

# Enable breakpoint() function
os.environ['PYTHONBREAKPOINT'] = 'pdb.set_trace'

# Add build directory to path for madrona_escape_room
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build')))

# import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import madrona_escape_room

# Local imports
from .ppo_models import RecurrentPPOAgent, MLP
from .ppo_utils import setup_obs, get_dones, get_rewards, get_obs_dim
from .action_utils import get_action_dims, flat_to_multi_discrete, multi_discrete_to_flat


from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


@dataclass
class Args:
    # Experiment
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    gpu_sim: bool = False
    """whether to use GPU for simulation"""
    gpu_id: int = 0
    """GPU ID to use"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    num_worlds: int = 4
    """number of parallel worlds for Madrona environment"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(512, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x / 255.0)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state
        
    def get_initial_state(self, batch_size, device):
        """Initialize the LSTM state with zeros."""
        return (
            torch.zeros(1, batch_size, self.lstm.hidden_size).to(device),
            torch.zeros(1, batch_size, self.lstm.hidden_size).to(device)
        )


def main():
    try:
        print("Starting PPO training...")
        args = tyro.cli(Args)
        print(f"Using args: {args}")
        
        # Calculate derived arguments
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
        
        print(f"Batch size: {args.batch_size}, Minibatch size: {args.minibatch_size}, Iterations: {args.num_iterations}")
        
        # Add a breakpoint here to inspect the environment setup
        breakpoint()
        
        # Rest of your training code...
        
        return args  # Return args for use in __main__
        
    except Exception as e:
        import traceback
        print(f"Error in PPO training: {str(e)}")
        traceback.print_exc()
        breakpoint()  # Drop into debugger on error
        return None  # Return None if there was an error

if __name__ == "__main__":
    args = main()
    if args is not None:  # Only proceed if main() returned args
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        
        # Initialize wandb if tracking is enabled
        if hasattr(args, 'track') and args.track and hasattr(args, 'wandb_project_name'):
            try:
                import wandb
                wandb.init(
                    project=args.wandb_project_name,
                    entity=getattr(args, 'wandb_entity', None),
                    name=run_name,
                    sync_tensorboard=True,
                    monitor_gym=True,
                    save_code=True,
                    config=vars(args)
                )
            except ImportError:
                print("Warning: wandb is not installed. Running without wandb tracking.")
                args.track = False
        
        # Initialize tensorboard writer
        writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # original gym environment
    sim = madrona_escape_room.SimManager(
        exec_mode = madrona_escape_room.madrona.ExecMode.CUDA if args.gpu_sim else madrona_escape_room.madrona.ExecMode.CPU,
        gpu_id = args.gpu_id,
        num_worlds = args.num_envs,
        rand_seed = args.seed,
        auto_reset = True,
    )
    
    # Initialize PPO agent
    obs_dim = get_obs_dim(sim)
    action_dims = get_action_dims()
    agent = RecurrentPPOAgent(obs_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Initialize storage
    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)  # (T, N)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)  # (T, N)
    
    # Initialize LSTM states storage
    lstm_states = (
        torch.zeros((args.num_steps, 1, args.num_envs, 128)).to(device),  # h
        torch.zeros((args.num_steps, 1, args.num_envs, 128)).to(device)   # c
    )

    # Initialize environment
    global_step = 0
    start_time = time.time()
    next_obs = setup_obs(sim, device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    # Initialize LSTM state
    agent = agent.to(device)
    with torch.no_grad():
        next_lstm_state = agent.get_initial_state(args.num_envs, device)
        
    print(f"Initialized LSTM state: {next_lstm_state[0].shape}, {next_lstm_state[1].shape}")
    print(f"Next obs shape: {next_obs.shape}")
    print(f"Next done shape: {next_done.shape}")

    for iteration in range(1, args.num_iterations + 1):
        # Store initial LSTM state for this iteration
        initial_lstm_state = (
            next_lstm_state[0].clone().detach(),
            next_lstm_state[1].clone().detach()
        )
        
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            
        print(f"\n--- Iteration {iteration}/{args.num_iterations} ---")

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            # Store LSTM states
            lstm_states[0][step] = next_lstm_state[0]
            lstm_states[1][step] = next_lstm_state[1]
            
            # Get action, value, and next LSTM state
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                    next_obs.unsqueeze(0), next_lstm_state, next_done.unsqueeze(0)
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            
            # Step the environment
            sim.step()
            
            # Get next observation and reward
            next_obs = setup_obs(sim, device)
            rewards[step] = get_rewards(sim, device)
            next_done = get_dones(sim, device)
            
            # Record rewards for plotting purposes
            if 'episode' in info:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # Compute advantages using GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape(-1, obs_dim)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # Optimize the policy and value network
        inds = np.arange(args.batch_size)

        clipfracs = []
        for epoch in range(args.update_epochs):
            # Randomly shuffle the indices for minibatch updates
            np.random.shuffle(inds)
            # Process minibatches
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = inds[start:end]
                
                # Get new logprob and entropy for the minibatch
                newlogprob, entropy = agent.get_log_prob_entropy(
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                    None,  # Don't need LSTM state for updates
                    None   # Don't need done for updates
                )
                
                # Get new value estimate
                with torch.no_grad():
                    newvalue = agent.get_value(b_obs[mb_inds], None)
                
                # Calculate policy loss
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                # Ensure ratio is finite
                ratio = torch.clamp(ratio, 0, 10)
                
                with torch.no_grad():
                    # Calculate approximate KL divergence
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                
                # Advantage normalization
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss with clipping
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            
            # Early stopping if KL divergence is too high
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log training metrics
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # Log to TensorBoard
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        
        # Print training stats
        if global_step % 1000 == 0:
            total_steps = args.num_steps * args.num_envs
            print(f"Global Step: {global_step}, "
                  f"FPS: {int(global_step / (time.time() - start_time))}, "
                  f"Value Loss: {v_loss.item():.3f}, "
                  f"Policy Loss: {pg_loss.item():.3f}")
        
        # Save model checkpoint
        if args.save_frequency and (global_step % args.save_frequency == 0 or global_step == args.total_timesteps):
            checkpoint_path = f"runs/{run_name}/checkpoints/checkpoint_{global_step}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'global_step': global_step,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args)
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Close the environment and writer
    writer.close()
    print("Training completed!")