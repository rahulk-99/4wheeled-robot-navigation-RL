import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.robot_gym import FourWheelRobotEnv
from src.vis_mcap import McapVisualizer

# --- Neural Network (Fixed for proper action bounds) ---
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        # Actor outputs mean for 2D continuous action
        self.actor_mean = nn.Linear(128, 2)
        # Learnable log std for exploration
        self.actor_logstd = nn.Parameter(torch.zeros(2))
        self.critic = nn.Linear(128, 1)
        
        # Action bounds
        self.action_low = torch.tensor([-1.5, 0.0])
        self.action_high = torch.tensor([1.5, 2.0])
        
        # Initialize velocity output bias to positive (so robot starts moving)
        with torch.no_grad():
            self.actor_mean.bias[1] = 1.0  # Initialize velocity bias to 1.0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        raw_mean = self.actor_mean(x)
        
        # Use tanh to bound output to [-1, 1], then scale to action range
        # This allows gradients to flow properly
        tanh_mean = torch.tanh(raw_mean)
        
        # Scale from [-1, 1] to action bounds
        action_low = self.action_low.to(x.device)
        action_high = self.action_high.to(x.device)
        mean = action_low + (tanh_mean + 1.0) * 0.5 * (action_high - action_low)
        
        std = torch.exp(self.actor_logstd).clamp(min=0.01)
        value = self.critic(x)
        return mean, std, value

# --- PPO Agent with Logging ---
class PPOAgent:
    def __init__(self, force_cpu=False):
        # Device Check
        if force_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.policy = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.0005)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100,  # Decay every 100 training steps
            gamma=0.95      # Multiply LR by 0.95
        )
        self.gamma = 0.98
        self.eps_clip = 0.2
        self.K_epochs = 3
        self.data = []
        
        # TensorBoard Writer
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../outputs/logs/ppo_custom"))
        self.writer = SummaryWriter(log_dir)
        self.training_step = 0

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        s, a, r, s_prime, log_prob_old, done_mask = zip(*self.data)
        
        # Convert to tensor and move to device
        s = torch.tensor(np.array(s), dtype=torch.float).to(self.device)
        a = torch.tensor(np.array(a), dtype=torch.float).to(self.device)
        r = torch.tensor(r, dtype=torch.float).to(self.device)
        s_prime = torch.tensor(np.array(s_prime), dtype=torch.float).to(self.device)
        done_mask = torch.tensor(done_mask, dtype=torch.float).to(self.device)
        log_prob_old = torch.tensor(log_prob_old, dtype=torch.float).to(self.device)

        with torch.no_grad():
            _, _, next_val = self.policy(s_prime)
            td_target = r.unsqueeze(1) + self.gamma * next_val * done_mask.unsqueeze(1)
            
        total_loss = 0
        total_entropy = 0
        
        for _ in range(self.K_epochs):
            mean, std, v_s = self.policy(s)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(a).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            advantage = (td_target - v_s).detach().squeeze()
            ratio = torch.exp(log_prob - log_prob_old)
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            
            # Components of loss
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.smooth_l1_loss(v_s.squeeze(), td_target.squeeze())
            
            loss = actor_loss + critic_loss - 0.01 * entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_entropy += entropy.mean().item()

        # LOGGING TO TENSORBOARD
        self.writer.add_scalar("Loss/total", total_loss / self.K_epochs, self.training_step)
        self.writer.add_scalar("PPO/entropy", total_entropy / self.K_epochs, self.training_step)
        self.training_step += 1
        
        # Update learning rate scheduler
        self.scheduler.step()
        
        self.data = []

# Main Loop 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train PPO Agent')
    parser.add_argument('--cpu', action='store_true', help='Force use of CPU')
    args = parser.parse_args()

    env = FourWheelRobotEnv() # Initialize the Gym Env
    agent = PPOAgent(force_cpu=args.cpu)
    
    print("Training started with TensorBoard logging...")
    print("Run: 'tensorboard --logdir=runs' to view charts")

    mcap_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../outputs/recordings/training_run_custom.mcap"))
    viz = McapVisualizer(mcap_path)
    print(f"Recording training visualization to: {mcap_path}")

    global_time = 0.0 # Track continuous time across episodes

    try:
        for episode in range(5000):
            obs, info = env.reset()
            episode_reward = 0
            done = False
            
            # Record every 200th episode
            should_record = (episode % 200 == 0)
            if should_record:
                print(f"Recording episode {episode} to MCAP...")
            
            while not done:
                # Send obs to device
                obs_tensor = torch.from_numpy(obs).float().to(agent.device) 
                mean, std, val = agent.policy(obs_tensor)
                dist = Normal(mean, std)
                action = dist.sample()
                
                # Clamp action to valid range [curvature, velocity]
                low_limit = torch.tensor([-1.5, 0.0]).to(agent.device)
                high_limit = torch.tensor([1.5, 2.0]).to(agent.device)
                
                action = torch.clamp(action, low_limit, high_limit)
                log_prob = dist.log_prob(action).sum().item()
                
                # Gym Step (needs CPU numpy)
                action_cpu = action.cpu().numpy()
                next_obs, reward, terminated, truncated, info = env.step(action_cpu)
                done = terminated or truncated
                
                # VISUALIZATION LOGGING
                if should_record:
                    state, target = env.get_robot_state()
                    wheel_angles, _ = env.robot.get_wheel_states()
                    # Continuous time logging
                    t_nanos = int(global_time * 1e9)
                    viz.log_frame(t_nanos, state, target, wheel_angles)
                
                if should_record:
                     global_time += env.dt
                
                done_mask = 0.0 if done else 1.0
                agent.put_data((obs, action.cpu().numpy(), reward, next_obs, log_prob, done_mask))
                
                obs = next_obs
                episode_reward += reward
                
                if len(agent.data) >= 512 or done:
                    agent.train_net()
            
            # Log Reward per Episode
            agent.writer.add_scalar("Reward/episode", episode_reward, episode)
            
            if episode % 100 == 0:
                print(f"Episode {episode}: Reward = {episode_reward:.1f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    finally:
        agent.writer.close()
        viz.close()
        
        # Save trained model
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../outputs/models/trained_policy_custom.pth"))
        torch.save(agent.policy.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
        print("Training Complete (or Interrupted).")