import subprocess
import sys
import argparse
import os
import re
from collections import defaultdict
from statistics import mean
from pathlib import Path

# Parse arguments
def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--gpu-id', type=int, default=0)
    arg_parser.add_argument('--ckpt-path', type=str, required=True)
    arg_parser.add_argument('--dump-suffix', type=str)

    arg_parser.add_argument('--num-worlds', type=int, required=True)
    arg_parser.add_argument('--num-steps', type=int, required=True)

    arg_parser.add_argument('--num-channels', type=int, default=64)
    arg_parser.add_argument('--separate-value', action='store_true')
    arg_parser.add_argument('--fp16', action='store_true')

    arg_parser.add_argument('--gpu-sim', action='store_true')
    
    return arg_parser.parse_args()

def analyze_results(output, num_worlds):
    # Parse the output lines that match the pattern: <world_id> <completion_metric>
    results = {}
    pattern = re.compile(r'(\d+)\s+([0-9.]+)')
    
    for line in output.split('\n'):
        match = pattern.match(line.strip())
        if match:
            world_id = int(match.group(1))
            completion = float(match.group(2))
            # Only take the first occurrence of each world ID
            if world_id not in results:
                results[world_id] = completion
    
    # Calculate statistics
    worlds_completed = len(results)
    success_rate = worlds_completed / num_worlds if num_worlds > 0 else 0
    avg_time_left = mean(results.values()) if results else 0
    
    # Store results directly (no need for grouping since we only take the first occurrence)
    world_stats = {world_id: [completion] for world_id, completion in results.items()}
    
    # Since we only have one value per world, the averages are the same as the results
    world_averages = results
    
    return {
        "success_rate": success_rate,
        "avg_time_left": avg_time_left,
        "world_stats": world_stats,
        "world_averages": world_averages,
        "total_worlds_completed": worlds_completed
    }

def main():
    args = parse_args()
    
    # Build the command to run infer.py with the same arguments
    script_path = Path(__file__).parent / "infer.py"
    command = [sys.executable, str(script_path)]
    
    # Add all the arguments from the parser
    for arg_name, arg_value in vars(args).items():
        # Convert argument name format
        arg_name = f"--{arg_name.replace('_', '-')}"
        
        # Handle boolean flags
        if isinstance(arg_value, bool):
            if arg_value:
                command.append(arg_name)
        # Handle all other arguments
        elif arg_value is not None:
            command.append(arg_name)
            command.append(str(arg_value))
    
    # Run the command and capture output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    # Analyze the results
    results = analyze_results(stdout, args.num_worlds)
    
    # Print the analysis
    print("\nAnalysis:")
    print(f"Total worlds: {args.num_worlds}")
    print(f"Worlds completed: {results['total_worlds_completed']}")
    print(f"Success rate: {results['success_rate']:.2f} ({results['total_worlds_completed']}/{args.num_worlds})")
    print(f"Average time left: {results['avg_time_left']:.4f}")
    
if __name__ == "__main__":
    main()