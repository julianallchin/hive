#!/usr/bin/env python3
"""
Random Policy Experiment for Hive Simulation

This script runs multiple episodes of the hive simulation with randomized ant actions,
records success metrics, and generates statistics and visualizations.

Usage:
    python run_random_policy_experiment.py [--episodes NUM_EPISODES] [--worlds NUM_WORLDS]

Arguments:
    --episodes: Number of episodes to run (default: 1000)
    --worlds: Number of parallel worlds to simulate (default: 10)
"""

import os
import re
import sys
import time
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Regular expression to match the success output (steps_taken:reward)
SUCCESS_PATTERN = re.compile(r'SUCCESS:(\d+):([\d\.\-]+)')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run random policy experiment for hive simulation')
    parser.add_argument('--worlds', type=int, default=100, help='Number of parallel worlds to simulate (= number of episodes)')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps per episode')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--exec-mode', type=str, default='CPU', choices=['CPU', 'CUDA'], help='Execution mode (CPU or CUDA)')
    return parser.parse_args()

def run_simulation(num_worlds, num_steps, random_seed, exec_mode='CPU'):
    """Run the headless simulation and capture output."""
    # Determine the path to the headless executable
    build_dir = Path(__file__).parent.parent / 'build'
    headless_path = build_dir / 'headless'
    
    if not headless_path.exists():
        print(f"Error: Headless executable not found at {headless_path}")
        print("Make sure to build the project first with `cmake -B build && cmake --build build`")
        sys.exit(1)
    
    # Each world is an independent episode
    # Total episodes = num_worlds
    print(f"Running {num_worlds} episodes (parallel worlds)")
    print(f"Each episode runs for up to {num_steps} steps")
    print(f"Execution mode: {exec_mode}")
    
    # Set environment variable for random seed if provided
    env = os.environ.copy()
    if random_seed is not None:
        env['RANDOM_SEED'] = str(random_seed)
    
    # Start the simulation process
    command = [str(headless_path), exec_mode, str(num_worlds), str(num_steps)] # can add "--rand-actions" once implemented
    print(f"Command: {' '.join(command)}")
    
    # Create output directory for results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"random_policy_results_{timestamp}.txt"
    
    print(f"Saving raw output to: {output_file}")
    
    # Run the simulation and capture output
    start_time = time.time()
    with open(output_file, 'w') as f:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                  universal_newlines=True, cwd=build_dir, env=env)
        
        for line in process.stdout:
            f.write(line)
            f.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            
            # Print progress indicator for successful episodes
            if line.startswith('SUCCESS:'):
                print('\rSuccessful episode detected!', end='')
    
    process.wait()
    elapsed_time = time.time() - start_time
    print(f"\nSimulation completed in {elapsed_time:.2f} seconds")
    
    return output_file

def analyze_results(output_file, num_worlds):
    """Analyze the simulation results and generate statistics."""
    print(f"\nAnalyzing results from {output_file}")
    
    # Parse the output file to find successful episodes and their lengths
    successful_episodes = []
    episode_rewards = []
    
    with open(output_file, 'r') as f:
        for line in f:
            match = SUCCESS_PATTERN.search(line)
            if match:
                # In the new format, the first number is steps_taken (not steps_remaining)
                steps_taken = int(match.group(1))
                episode_reward = float(match.group(2))
                successful_episodes.append(steps_taken)
                episode_rewards.append(episode_reward)
    
    # Calculate statistics
    num_successful = len(successful_episodes)
    success_rate = num_successful / num_worlds * 100
    
    print(f"Total episodes: {num_worlds}")
    print(f"Successful episodes: {num_successful}")
    print(f"Success rate: {success_rate:.2f}%")
    
    if num_successful > 0:
        avg_length = np.mean(successful_episodes)
        min_length = np.min(successful_episodes)
        max_length = np.max(successful_episodes)
        median_length = np.median(successful_episodes)
        
        avg_reward = np.mean(episode_rewards)
        min_reward = np.min(episode_rewards)
        max_reward = np.max(episode_rewards)
        
        print(f"Average episode length for successful episodes: {avg_length:.2f} steps")
        print(f"Minimum episode length: {min_length} steps")
        print(f"Maximum episode length: {max_length} steps")
        print(f"Median episode length: {median_length} steps")
        
        print(f"Average reward for successful episodes: {avg_reward:.4f}")
        print(f"Minimum reward: {min_reward:.4f}")
        print(f"Maximum reward: {max_reward:.4f}")
    
    # Generate visualizations
    generate_figures(successful_episodes, episode_rewards, num_worlds, output_file)
    
    return {
        'total_episodes': num_worlds,
        'successful_episodes': num_successful,
        'success_rate': success_rate,
        'episode_lengths': successful_episodes,
        'episode_rewards': episode_rewards,
        'avg_episode_length': np.mean(successful_episodes) if num_successful > 0 else 0,
        'avg_episode_reward': np.mean(episode_rewards) if num_successful > 0 else 0
    }

def generate_figures(successful_episodes, episode_rewards, num_worlds, output_file):
    """Generate figures from the results."""
    if not successful_episodes:
        print("No successful episodes to visualize")
        return
    
    results_dir = Path(output_file).parent
    basename = Path(output_file).stem
    
    # Create a histogram of episode lengths
    plt.figure(figsize=(10, 6))
    plt.hist(successful_episodes, bins=min(20, len(successful_episodes)) if len(successful_episodes) > 0 else 1, 
             alpha=0.7, color='blue')
    plt.xlabel('Episode Length (steps)')
    plt.ylabel('Number of Episodes')
    plt.title(f'Distribution of Episode Lengths for Successful Episodes\n'
              f'Success Rate: {len(successful_episodes)/num_worlds*100:.2f}% '
              f'({len(successful_episodes)}/{num_worlds})')
    plt.grid(True, alpha=0.3)
    
    hist_file = results_dir / f"{basename}_length_histogram.png"
    plt.savefig(hist_file)
    print(f"Saved episode length histogram to {hist_file}")
    
    # Create a histogram of episode rewards
    plt.figure(figsize=(10, 6))
    plt.hist(episode_rewards, bins=min(20, len(episode_rewards)) if len(episode_rewards) > 0 else 1, 
             alpha=0.7, color='green')
    plt.xlabel('Episode Reward')
    plt.ylabel('Number of Episodes')
    plt.title(f'Distribution of Rewards for Successful Episodes')
    plt.grid(True, alpha=0.3)
    
    reward_hist_file = results_dir / f"{basename}_reward_histogram.png"
    plt.savefig(reward_hist_file)
    print(f"Saved reward histogram to {reward_hist_file}")
    
    # Create a summary figure
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.pie([len(successful_episodes), num_worlds - len(successful_episodes)], 
            labels=['Success', 'Failure'], autopct='%1.1f%%', colors=['green', 'red'])
    plt.title('Success vs. Failure Rate')
    
    plt.subplot(2, 2, 2)
    plt.boxplot(successful_episodes)
    plt.ylabel('Episode Length (steps)')
    plt.title('Episode Length Statistics')
    
    plt.subplot(2, 2, 3)
    plt.scatter(successful_episodes, episode_rewards, alpha=0.7)
    plt.xlabel('Episode Length (steps)')
    plt.ylabel('Episode Reward')
    plt.title('Episode Length vs Reward')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.boxplot(episode_rewards)
    plt.ylabel('Episode Reward')
    plt.title('Episode Reward Statistics')
    
    plt.tight_layout()
    summary_file = results_dir / f"{basename}_summary.png"
    plt.savefig(summary_file)
    print(f"Saved summary figure to {summary_file}")
    
    # Save raw data for further analysis
    data_file = results_dir / f"{basename}_data.npz"
    np.savez(data_file, 
             episode_lengths=np.array(successful_episodes),
             episode_rewards=np.array(episode_rewards))
    print(f"Saved raw data to {data_file}")

def main():
    """Main function to run the experiment."""
    args = parse_args()
    
    # Run the simulation
    output_file = run_simulation(args.worlds, args.steps, args.random_seed, args.exec_mode)
    
    # Analyze the results
    results = analyze_results(output_file, args.worlds)
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()
