import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.robot_gym import FourWheelRobotEnv
from src.vis_mcap import McapVisualizer


class VisualizationCallback(BaseCallback):
    """
    Callback to record mcap visualizations every N episodes
    """
    def __init__(self, viz, env, record_freq=200, verbose=0):
        super().__init__(verbose)
        self.viz = viz
        self.env = env
        self.record_freq = record_freq
        self.episode_count = 0
        self.global_time = 0.0
        self.should_record = False
        
    def _on_step(self) -> bool:
        # Check if episode just started
        if self.locals.get('dones')[0]:
            self.episode_count += 1
            self.should_record = (self.episode_count % self.record_freq == 0)
            if self.should_record:
                print(f"Recording episode {self.episode_count} to MCAP...")
        
        # Record if we should recorrd
        if self.should_record:
            state, target = self.env.get_robot_state()
            wheel_angles, _ = self.env.robot.get_wheel_states()
            t_nanos = int(self.global_time * 1e9)
            self.viz.log_frame(t_nanos, state, target, wheel_angles)
            self.global_time += self.env.dt
            
        return True


class EpisodeRewardCallback(BaseCallback):
    """
    Callback to print episode rewards every N episodes
    """
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
    parser = argparse.ArgumentParser(description='Train PPO Agent using Stable-Baselines3')
    parser.add_argument('--cpu', action='store_true', help='Force use of CPU')
    parser.add_argument('--timesteps', type=int, default=500000, help='Total timesteps to train')
    args = parser.parse_args()

    # Create environment
    env = FourWheelRobotEnv()
    
    # Setup device
    device = 'cpu' if args.cpu else 'auto'
    print(f"Using device: {device}")
    
    # Configure TensorBoard logging
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs/logs/ppo_sb3"))
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    
    # Create PPO model with custom hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0005,
        n_steps=512,           # Feeding batch of 512 for back propagation
        batch_size=64,         # Mini-batch size
        n_epochs=3,            # K_epochs
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,        # eps_clip
        ent_coef=0.01,         # Entropy coefficient
        vf_coef=0.5,           # Value function coefficient
        max_grad_norm=0.5,
        verbose=1,
        device=device,
        tensorboard_log=log_dir
    )
    
    # Set logger
    model.set_logger(new_logger)
    
    print("Training started with Stable-Baselines3 PPO...")
    print(f"Run: 'tensorboard --logdir={log_dir}' to view charts")
    
    # Setup visualization
    mcap_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../outputs/recordings/training_run_sb3.mcap"))
    viz = McapVisualizer(mcap_path)
    print(f"Recording training visualization to: {mcap_path}")
    
    # Create callbacks
    viz_callback = VisualizationCallback(viz, env, record_freq=200)
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
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs/models/trained_policy_sb3.zip"))
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        print("Training Complete (or Interrupted).")
