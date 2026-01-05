import torch
import numpy as np
import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.robot_gym import FourWheelRobotEnv
from src.vis_mcap import McapVisualizer

# Import ActorCritic from train script 
from .train_custom import ActorCritic

def play_policy(model_path, output_file="../outputs/recordings/policy_run_custom.mcap", num_episodes=3, force_cpu=False):
    """
    Run trained policy and record to MCAP for visualization.
    
    Args:
        model_path: Path to saved model weights (.pth file) - REQUIRED
        output_file: Output MCAP filename
        num_episodes: Number of episodes to record
        force_cpu: If True, force usage of CPU
    """
    # Device Check
    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize environment and policy
    env = FourWheelRobotEnv()
    policy = ActorCritic().to(device)
    
    # Load trained weights
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please provide a valid model path with --model")
    
    # Load with map_location
    policy.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded policy from {model_path}")
    
    policy.eval()  # Set to evaluation mode
    
    # Initialize visualization
    viz = McapVisualizer(output_file)
    
    print(f"\nRecording {num_episodes} episodes to {output_file}...")
    
    global_time = 0.0 # Track continuous time
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        episode_reward = 0
        
        print(f"\nEpisode {ep+1}/{num_episodes}")
        print(f"  Target: ({env.target[0]:.2f}, {env.target[1]:.2f})")
        
        # Log initial state BEFORE any actions
        state, target = env.get_robot_state()
        wheel_angles, _ = env.robot.get_wheel_states()
        t_nanos = int(global_time * 1e9)
        viz.log_frame(t_nanos, state, target, wheel_angles)
        global_time += env.dt
        
        while not done and step_count < 200:
            # Get action
            obs_tensor = torch.from_numpy(obs).float().to(device)
            with torch.no_grad():
                mean, std, _ = policy(obs_tensor)
                # Debug: print raw output before clamping
                if step_count == 0:
                    mean_np = mean.cpu().numpy()
                    print(f"    Raw network output: κ={mean_np[0]:.3f}, v={mean_np[1]:.3f}")
                
                # Use mean directly for evaluation
                action = torch.clamp(mean, 
                                    torch.tensor([-1.5, 0.0]).to(device), 
                                    torch.tensor([1.5, 2.0]).to(device))
            
            # Debug: Print action every 50 steps
            if step_count % 50 == 0:
                action_np = action.cpu().numpy()
                print(f"    Step {step_count}: Action=[κ={action_np[0]:.3f}, v={action_np[1]:.3f}], Pos=({env.state[0]:.2f}, {env.state[1]:.2f})")
            
            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated
            
            # Log to MCAP
            state, target = env.get_robot_state()
            wheel_angles, _ = env.robot.get_wheel_states()
            
            # Continuous time
            t_nanos = int(global_time * 1e9)
            viz.log_frame(t_nanos, state, target, wheel_angles)
            
            obs = next_obs
            episode_reward += reward
            step_count += 1
            global_time += env.dt # Increment global time
        
        # Calculate final distance to target
        dx = env.target[0] - env.state[0]
        dy = env.target[1] - env.state[1]
        final_dist = np.sqrt(dx**2 + dy**2)
        
        success = final_dist < 0.3
        print(f"  Steps: {step_count}, Reward: {episode_reward:.1f}")
        print(f"  Final distance: {final_dist:.3f}m - {'SUCCESS' if success else ' FAILED'}")
    
    viz.close()
    print(f"\n Recording saved to: {output_file}")
    print(f"View with: ros2 bag play {output_file} -s mcap")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run trained PPO policy and record MCAP')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights (.pth file)')
    parser.add_argument('--output', type=str, 
                       default='outputs/recordings/policy_run_custom.mcap' if os.path.exists('outputs') else '../outputs/recordings/policy_run_custom.mcap',
                       help='Output MCAP filename')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to record')
    parser.add_argument('--cpu', action='store_true', help='Force use of CPU')
    
    args = parser.parse_args()
    
    play_policy(model_path=args.model, 
               output_file=args.output, 
               num_episodes=args.episodes,
               force_cpu=args.cpu)
