import torch
import madrona_escape_room

from madrona_escape_room_learn import (
    train, profile, TrainConfig, PPOConfig, SimInterface,
)

from policy import make_policy, setup_obs

import argparse
import math
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)

class LearningCallback:
    def __init__(self, ckpt_dir, profile_report, run_name):
        self.mean_fps = 0
        self.ckpt_dir = ckpt_dir
        self.profile_report = profile_report
        
        # Initialize TensorBoard writer
        log_dir = os.path.join('runs', run_name)
        os.makedirs(log_dir, exist_ok=True)
        print(f"Saving TensorBoard logs to: {os.path.abspath(log_dir)}")
        self.writer = SummaryWriter(log_dir=log_dir)

    def __call__(self, update_idx, update_time, update_results, learning_state):
        update_id = update_idx + 1
        fps = args.num_worlds * args.steps_per_update / update_time
        self.mean_fps += (fps - self.mean_fps) / update_id
        
        if update_id != 1 and update_id % 10 != 0:
            return
            
        # Log to TensorBoard at the same frequency as printing
        self._log_to_tensorboard(update_idx, update_time, update_results, learning_state)

        ppo = update_results.ppo_stats

        with torch.no_grad():
            reward_mean = update_results.rewards.mean().cpu().item()
            reward_min = update_results.rewards.min().cpu().item()
            reward_max = update_results.rewards.max().cpu().item()

            value_mean = update_results.values.mean().cpu().item()
            value_min = update_results.values.min().cpu().item()
            value_max = update_results.values.max().cpu().item()

            advantage_mean = update_results.advantages.mean().cpu().item()
            advantage_min = update_results.advantages.min().cpu().item()
            advantage_max = update_results.advantages.max().cpu().item()

            bootstrap_value_mean = update_results.bootstrap_values.mean().cpu().item()
            bootstrap_value_min = update_results.bootstrap_values.min().cpu().item()
            bootstrap_value_max = update_results.bootstrap_values.max().cpu().item()

            vnorm_mu = learning_state.value_normalizer.mu.cpu().item()
            vnorm_sigma = learning_state.value_normalizer.sigma.cpu().item()

        print(f"\nUpdate: {update_id}")
        print(f"    Loss: {ppo.loss: .3e}, A: {ppo.action_loss: .3e}, V: {ppo.value_loss: .3e}, E: {ppo.entropy_loss: .3e}")
        print()
        print(f"    Rewards          => Avg: {reward_mean: .3e}, Min: {reward_min: .3e}, Max: {reward_max: .3e}")
        print(f"    Values           => Avg: {value_mean: .3e}, Min: {value_min: .3e}, Max: {value_max: .3e}")
        print(f"    Advantages       => Avg: {advantage_mean: .3e}, Min: {advantage_min: .3e}, Max: {advantage_max: .3e}")
        print(f"    Bootstrap Values => Avg: {bootstrap_value_mean: .3e}, Min: {bootstrap_value_min: .3e}, Max: {bootstrap_value_max: .3e}")
        print(f"    Returns          => Avg: {ppo.returns_mean}, σ: {ppo.returns_stddev}")
        print(f"    Value Normalizer => Mean: {vnorm_mu: .3e}, σ: {vnorm_sigma :.3e}")

        if self.profile_report:
            print()
            print(f"    FPS: {fps:.0f}, Update Time: {update_time:.2f}, Avg FPS: {self.mean_fps:.0f}")
            print(f"    PyTorch Memory Usage: {torch.cuda.memory_reserved() / 1024 / 1024 / 1024:.3f}GB (Reserved), {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.3f}GB (Current)")
            profile.report()

        if update_id % 100 == 0:
            learning_state.save(update_idx, self.ckpt_dir / f"{update_id}.pth")
    
    def _log_to_tensorboard(self, update_idx, update_time, update_results, learning_state):
        """Log training metrics to TensorBoard"""
        ppo = update_results.ppo_stats
        
        # Log PPO losses
        self.writer.add_scalar('Loss/total', ppo.loss, update_idx)
        self.writer.add_scalar('Loss/action', ppo.action_loss, update_idx)
        self.writer.add_scalar('Loss/value', ppo.value_loss, update_idx)
        self.writer.add_scalar('Loss/entropy', ppo.entropy_loss, update_idx)
        
        # Log reward statistics
        self.writer.add_scalar('Reward/mean', ppo.returns_mean, update_idx)
        self.writer.add_scalar('Reward/stddev', ppo.returns_stddev, update_idx)
        
        # Log detailed statistics
        with torch.no_grad():
            self.writer.add_scalar('Values/mean', update_results.values.mean().cpu().item(), update_idx)
            self.writer.add_scalar('Values/min', update_results.values.min().cpu().item(), update_idx)
            self.writer.add_scalar('Values/max', update_results.values.max().cpu().item(), update_idx)
            
            self.writer.add_scalar('Advantages/mean', update_results.advantages.mean().cpu().item(), update_idx)
            self.writer.add_scalar('Bootstrap/mean', update_results.bootstrap_values.mean().cpu().item(), update_idx)
            
            self.writer.add_scalar('ValueNorm/mean', learning_state.value_normalizer.mu.cpu().item(), update_idx)
            self.writer.add_scalar('ValueNorm/std', learning_state.value_normalizer.sigma.cpu().item(), update_idx)
        
        # Log performance metrics
        self.writer.add_scalar('Performance/fps', args.num_worlds * args.steps_per_update / update_time, update_idx)
        self.writer.add_scalar('Performance/update_time', update_time, update_idx)
        self.writer.add_scalar('Performance/mean_fps', self.mean_fps, update_idx)
        
        # Log learning rate if scheduler is being used
        if learning_state.scheduler is not None:
            self.writer.add_scalar('LearningRate', learning_state.scheduler.get_last_lr()[0], update_idx)
        
    def __del__(self):
        """Make sure to close the writer when done"""
        self.writer.close()


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--run-name', type=str, required=True)
arg_parser.add_argument('--restore', type=int)

arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-updates', type=int, required=True)
arg_parser.add_argument('--steps-per-update', type=int, default=40)
arg_parser.add_argument('--num-bptt-chunks', type=int, default=8)

arg_parser.add_argument('--lr', type=float, default=1e-4)
arg_parser.add_argument('--gamma', type=float, default=0.998)
arg_parser.add_argument('--entropy-loss-coef', type=float, default=0.01)
arg_parser.add_argument('--value-loss-coef', type=float, default=0.5)
arg_parser.add_argument('--clip-value-loss', action='store_true')

arg_parser.add_argument('--num-channels', type=int, default=64)
arg_parser.add_argument('--separate-value', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')

arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--profile-report', action='store_true')

args = arg_parser.parse_args()

sim = madrona_escape_room.SimManager(
    exec_mode = madrona_escape_room.madrona.ExecMode.CUDA if args.gpu_sim else madrona_escape_room.madrona.ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    rand_seed = 5,
    auto_reset = True,
)

ckpt_dir = Path('ckpts') / args.run_name

learning_cb = LearningCallback(ckpt_dir, args.profile_report, args.run_name)

if torch.cuda.is_available():
    dev = torch.device(f'cuda:{args.gpu_id}')
else:
    dev = torch.device('cpu')

ckpt_dir.mkdir(exist_ok=True, parents=True)

obs, num_obs_features_per_agent, num_agents_per_model = setup_obs(sim)
policy = make_policy(num_obs_features_per_agent, num_agents_per_model, args.num_channels, args.separate_value)

actions = sim.action_tensor().to_torch()
dones = sim.done_tensor().to_torch()
rewards = sim.reward_tensor().to_torch()

# Flatten N, M, ... tensors to N * M, -1
actions = actions.view(-1, math.prod(actions.shape[2:]))
dones  = dones.view(-1, math.prod(dones.shape[2:]))
rewards = rewards.view(-1, math.prod(rewards.shape[2:]))

if args.restore:
    restore_ckpt = ckpt_dir / f"{args.restore}.pth"
else:
    restore_ckpt = None

train(
    dev,
    SimInterface(
            step = lambda: sim.step(),
            obs = obs,
            actions = actions,
            dones = dones,
            rewards = rewards,
    ),
    TrainConfig(
        num_updates = args.num_updates,
        steps_per_update = args.steps_per_update,
        num_bptt_chunks = args.num_bptt_chunks,
        lr = args.lr,
        gamma = args.gamma,
        gae_lambda = 0.95,
        ppo = PPOConfig(
            num_mini_batches=1,
            clip_coef=0.2,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_loss_coef,
            max_grad_norm=0.5,
            num_epochs=2,
            clip_value_loss=args.clip_value_loss,
        ),
        value_normalizer_decay = 0.999,
        mixed_precision = args.fp16,
    ),
    policy,
    learning_cb,
    ckpt_dir,
    restore_ckpt
)