import sys
import os
from stable_baselines3 import PPO

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.multirobot_gym import MultiRobotEnv
from src.vis_mcap import McapVisualizer


def play_multi_robot_policy(model_path, output_mcap, num_episodes=3, use_cpu=False):
    """
    Run trained multi-robot collision avoidance policy
    """
    # Load model
    print(f"Loading model from: {model_path}")
    device = 'cpu' if use_cpu else 'auto'
    model = PPO.load(model_path, device=device)
    print(f"Device: {model.device}")
    
    # Create environment
    env = MultiRobotEnv()
    
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
            collision_occurred = False
            
            print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
            
            while not done:
                # Get action from policy
                action, _ = model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step += 1
                
                # Check collision
                if env._check_collision():
                    collision_occurred = True
                
                # Log all 3 robots with different colors
                (state1, target1), (state2, target2), (state3, target3) = env.get_all_robot_states()
                wheel_angles1, _ = env.robot1.get_wheel_states()
                wheel_angles2, _ = env.robot2.get_wheel_states()
                wheel_angles3, _ = env.robot3.get_wheel_states()
                
                t_nanos = int(global_time * 1e9)
                
                # Log all 3 robots: Red (agent), Green (obstacles)
                viz.log_frame(t_nanos, state1[:3], target1, wheel_angles1, robot_id=1)
                viz.log_frame(t_nanos, state2[:3], target2, wheel_angles2, robot_id=2)
                viz.log_frame(t_nanos, state3[:3], target3, wheel_angles3, robot_id=3)
                
                global_time += env.dt
            
            # Episode summary
            state1, target1 = env.get_all_robot_states()[0]
            distance = ((state1[0] - target1[0])**2 + (state1[1] - target1[1])**2)**0.5
            
            print(f"Steps: {step}")
            print(f"Total Reward: {episode_reward:.2f}")
            print(f"Collision: {'YES' if collision_occurred else 'NO'}")
            print(f"Final Distance to Target: {distance:.3f}m")
            print(f"Target: ({target1[0]:.2f}, {target1[1]:.2f})")
            print(f"Final Position: ({state1[0]:.2f}, {state1[1]:.2f})")
    
    finally:
        viz.close()
        print(f"\nVisualization saved to: {output_mcap}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Play trained multi-robot PPO policy')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model (.zip)')
    parser.add_argument('--output', type=str, 
                       default='outputs/recordings/policy_run_multi_sb3.mcap',
                       help='Output MCAP file path')
    parser.add_argument('--episodes', type=int, default=3, 
                       help='Number of episodes to run')
    parser.add_argument('--cpu', action='store_true', help='Force use of CPU')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    model_path = os.path.abspath(args.model)
    output_path = os.path.abspath(args.output)
    
    play_multi_robot_policy(model_path=model_path, 
                           output_mcap=output_path,
                           num_episodes=args.episodes,
                           use_cpu=args.cpu)
