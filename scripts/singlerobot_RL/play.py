import sys
import os
from stable_baselines3 import PPO

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.robot_gym import FourWheelRobotEnv
from src.vis_mcap import McapVisualizer


def play_policy(model_path, output_mcap, num_episodes=3):
    """
    Run trained Stable-Baselines3 PPO policy and record to MCAP
    """
    # Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    print(f"Device: {model.device}")
    
    # Create environment
    env = FourWheelRobotEnv()
    
    # Setup visualization
    viz = McapVisualizer(output_mcap)
    print(f"Recording to: {output_mcap}")
    
    global_time = 0.0
    
    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
            
            while not done:
                # Get action from policy - deterministic for evaluation
                action, _ = model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step += 1
                
                # Log visualization
                state, target = env.get_robot_state()
                wheel_angles, _ = env.robot.get_wheel_states()
                t_nanos = int(global_time * 1e9)
                viz.log_frame(t_nanos, state, target, wheel_angles)
                global_time += env.dt
            
            # Episode summary
            state, target = env.get_robot_state()
            distance = ((state[0] - target[0])**2 + (state[1] - target[1])**2)**0.5
            print(f"Steps: {step}")
            print(f"Total Reward: {episode_reward:.2f}")
            print(f"Final Distance to Target: {distance:.3f}m")
            print(f"Target: ({target[0]:.2f}, {target[1]:.2f})")
            print(f"Final Position: ({state[0]:.2f}, {state[1]:.2f})")
    
    finally:
        viz.close()
        print(f"\nVisualization saved to: {output_mcap}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Play trained SB3 PPO policy')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.zip)')
    parser.add_argument('--output', type=str, 
                       default='outputs/recordings/policy_run_sb3.mcap',
                       help='Output MCAP file path')
    parser.add_argument('--episodes', type=int, default=3, 
                       help='Number of episodes to run')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    model_path = os.path.abspath(args.model)
    output_path = os.path.abspath(args.output)
    
    play_policy(model_path=model_path, 
               output_mcap=output_path,
               num_episodes=args.episodes)
