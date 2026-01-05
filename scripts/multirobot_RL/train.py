import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.multirobot_gym import MultiRobotEnv
from src.vis_mcap import McapVisualizer


class MultiRobotVisualizationCallback(BaseCallback):
    """Callback to record all 3 robots to MCAP"""
    def __init__(self, viz, env, record_freq=200, verbose=0):
        super().__init__(verbose)
        self.viz = viz
        self.env = env
        self.record_freq = record_freq
        self.episode_count = 0
        self.global_time = 0.0
        self.should_record = False
        
    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            self.episode_count += 1
            self.should_record = (self.episode_count % self.record_freq == 0)
            if self.should_record:
                print(f"Recording episode {self.episode_count} to MCAP...")
        
        if self.should_record:
            (state1, target1), (state2, target2), (state3, target3) = self.env.get_all_robot_states()
            
            # Get wheel angles for all robots
            wheel_angles1, _ = self.env.robot1.get_wheel_states()
            wheel_angles2, _ = self.env.robot2.get_wheel_states()
            wheel_angles3, _ = self.env.robot3.get_wheel_states()
            
            t_nanos = int(self.global_time * 1e9)
            
            # Log all 3 robots with different colors
            self.viz.log_frame(t_nanos, state1[:3], target1, wheel_angles1, robot_id=1)  # Red (agent)
            self.viz.log_frame(t_nanos, state2[:3], target2, wheel_angles2, robot_id=2)  # Green (obstacle)
            self.viz.log_frame(t_nanos, state3[:3], target3, wheel_angles3, robot_id=3)  # Green (obstacle)
            
            self.global_time += self.env.dt
            
        return True


class EpisodeRewardCallback(BaseCallback):
    """Print episode rewards and collision stats"""
    def __init__(self, print_freq=100, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.episode_count = 0
        self.episode_reward = 0.0
        
    def _on_step(self) -> bool:
        self.episode_reward += self.locals.get('rewards')[0]
        
        if self.locals.get('dones')[0]:
            self.episode_count += 1
            if self.episode_count % self.print_freq == 0:
                print(f"Episode {self.episode_count}: Reward = {self.episode_reward:.1f}")
            self.episode_reward = 0.0
            
        return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Multi-Robot PPO with Collision Avoidance')
    parser.add_argument('--cpu', action='store_true', help='Force use of CPU')
    parser.add_argument('--timesteps', type=int, default=500000, help='Total timesteps to train')
    args = parser.parse_args()

    # Create environment
    env = MultiRobotEnv()
    
    # Setup device
    device = 'cpu' if args.cpu else 'auto'
    print(f"Using device: {device}")
    
    # Configure TensorBoard logging
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../outputs/logs/ppo_multi_sb3"))
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    
    # Create PPO model
    print("Creating PPO model for multi-robot collision avoidance...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,  # Slightly lower for more complex task
        n_steps=1024,          # More samples per update
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
        tensorboard_log=log_dir
    )
    
    model.set_logger(new_logger)
    
    print("Training started with Stable-Baselines3 PPO (Multi-Robot Extension)...")
    print(f"Run: 'tensorboard --logdir={log_dir}' to view charts")
    
    # Setup visualization
    mcap_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                             "../../outputs/recordings/training_run_multi_sb3.mcap"))
    viz = McapVisualizer(mcap_path)
    print(f"Recording training visualization to: {mcap_path}")
    
    # Create callbacks
    viz_callback = MultiRobotVisualizationCallback(viz, env, record_freq=200)
    reward_callback = EpisodeRewardCallback(print_freq=100)
    
    try:
        # Train the model
        model.learn(
            total_timesteps=args.timesteps,
            callback=[viz_callback, reward_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        viz.close()
        
        # Save the model
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                                  "../../outputs/models/trained_policy_multi_sb3.zip"))
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        print("Training Complete (or Interrupted).")
