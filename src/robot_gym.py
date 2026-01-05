import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

# Import physics formulation
from .kinematics import FourWheelSteeringModel

class FourWheelRobotEnv(gym.Env):
    """
    Custom Gymnasium Environment for 4 Wheeled Robot.
    """
    def __init__(self):
        super(FourWheelRobotEnv, self).__init__()
        
        # Action Space - Continuous [curvature, velocity]
        self.action_space = spaces.Box(
            low=np.array([-1.5, 0.0], dtype=np.float32),
            high=np.array([1.5, 2.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation Space - We return [Distance, Heading Error, v, kappa]
        # Normalized to [-1, 1] for the neural net
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.pi, 0.0, -1.5], dtype=np.float32),
            high=np.array([20.0, np.pi, 2.0, 1.5], dtype=np.float32),
            dtype=np.float32
        )
        
        self.robot = FourWheelSteeringModel()
        
        # Track current dynamic state
        self.current_v = 0.0
        self.current_kappa = 0.0
        
        self.target = np.array([5.0, 2.0])
        self.dt = 0.1
        self.max_steps = 200
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.reset()
        self.current_step = 0
        self.current_v = 0.0
        self.current_kappa = 0.0
        
        # Randomize Target for generalization : x in [2, 8], y in [-3, 3]
        tx = np.random.uniform(2.0, 8.0)
        ty = np.random.uniform(-3.0, 3.0)
        self.target = np.array([tx, ty])
        
        # Initial State. x, y, theta
        self.state = np.array([0.0, 0.0, 0.0]) 
        
        # Track distance and heading for reward shaping
        dx = self.target[0] - self.state[0]
        dy = self.target[1] - self.state[1]
        self.prev_dist = math.sqrt(dx**2 + dy**2)
        
        # Calculate initial heading error
        angle_to_target = math.atan2(dy, dx)
        current_heading = self.state[2]
        heading_err = angle_to_target - current_heading
        # Normalize to [-pi, pi]
        heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi
        self.prev_heading_err = abs(heading_err)
        
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        """Execute one timestep of the environment"""
        self.current_step += 1
        
        # Process action
        k_cmd, v_cmd = self._process_action(action)
        
        # Update robot state
        self._update_state(v_cmd, k_cmd)
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward
        reward = self._compute_reward(obs)
        
        # Check termination
        terminated = self._check_terminated(obs)
        truncated = self._check_truncated()
        
        return obs, reward, terminated, truncated, {}
    
    def _process_action(self, action):
        """Convert action array to robot commands"""
        k_cmd = float(action[0])  # curvature
        v_cmd = float(action[1])  # velocity
        return k_cmd, v_cmd
    
    def _update_state(self, v_cmd, k_cmd):
        """Update robot state using physics kinematicss model"""
        # Physics update
        real_v, real_k = self.robot.update(v_cmd, k_cmd, self.dt)
        self.current_v = real_v
        self.current_kappa = real_k
        
        # Integrate motion
        theta = self.state[2]
        new_theta = theta + real_v * real_k * self.dt
        new_x = self.state[0] + real_v * math.cos(theta) * self.dt
        new_y = self.state[1] + real_v * math.sin(theta) * self.dt
        self.state = np.array([new_x, new_y, new_theta])
    
    def _compute_reward(self, obs):
        """
        Calculate reward based on current observation.
        
        - Progress: reaching near to goal
        - Heading: heading towards target
        - Velocity: Smooth transition form start to high speed to back to stop
        - Curvature: Penalty to avoid spiraling/orbitting
        - Time: Distance-proportional penalty
        """
        # obs: [dist/10, heading/pi, v/2.0, k/1.5]. Un-normalize for readable logic
        dist = obs[0] * 10.0
        heading_err = obs[1] * math.pi  # Range [-pi, pi]
        abs_heading_err = abs(heading_err)
        
        # REWARD COMPONENTS
        
        # 1. Progress Reward - Positive if getting closer, negative if moving away
        dist_delta = self.prev_dist - dist
        reward_progress = dist_delta * 10.0
        self.prev_dist = dist
        
        # 2. Heading Reward - [0, 0.5] when facing target, [-0.5, 0] when facing away
        # This provides positive feedback for correct orientation
        heading_alignment = 1.0 - (abs_heading_err / math.pi)  # [0, 1]
        reward_heading = (heading_alignment - 0.5) * 1.0  # It will be in range [-0.5, 0.5]
        
        # 3. Time Penalty based om Distance-Proportionality. Less penalty when close to target
        reward_time = -0.05 * (1.0 + dist / 10.0)  # [-0.1, -0.15]
        
        # Penalize standing still when far from target
        if dist > 0.5 and self.current_v < 0.1:
            reward_time -= 2.0  
        
        # 4. Velocity Reward .Focus on stopping only when very close
        
        if dist < 0.3:
            # AT TARGET - Must stop NOW
            # Very strong penalty for moving when at target
            reward_vel = -self.current_v * 10.0
            
            # Large bonus for being stopped at target
            if self.current_v < 0.2:
                reward_vel += 30.0
                
        elif dist < 1.0:
            # CLOSE - Start slowing down
            # Target velocity proportional to distance
            v_target = dist * 0.5
            reward_vel = -abs(self.current_v - v_target) * 2.0
            
        else:
            # FAR - Encourage movement toward target
            heading_factor = max(0.0, heading_alignment)
            reward_vel = self.current_v * 0.3 * heading_factor
        
        # 5. Curvature Penalty to avoid spiraling and orbitting
        reward_curvature = -abs(self.current_kappa) * 0.1
        
        # 6. Success/Failure - Termination Reward
        reward_terminal = 0.0
        
        # Success Condition. Considred arrived at target successfully if distance is less than
        # 0.3m and velocity is less than 0.2m/s
        is_at_target = dist < 0.3
        is_stopped = self.current_v < 0.2
        
        if is_at_target and is_stopped:
            reward_terminal = 500.0  # Large reward for success
        # Failure Condition. Considred failed if robot goes out of bounds
        elif abs(self.state[0]) > 10 or abs(self.state[1]) > 10:
            reward_terminal = -10.0 
        # Failure Condition. Considred failed if robot takes too many steps
        elif self.current_step >= self.max_steps:
            reward_terminal = -5.0 

        # Total Reward
        reward = (reward_progress + reward_heading + reward_time + 
                 reward_vel + reward_curvature + reward_terminal)
        
        return reward
    
    def _check_terminated(self, obs):
        """Check if episode should terminate i.e success or failur"""
        dist = obs[0] * 10.0
        
        # Success (Parked)
        if dist < 0.3 and self.current_v < 0.2:
            return True
        
        # Failure (out of bounds)
        if abs(self.state[0]) > 10 or abs(self.state[1]) > 10:
            return True
            
        # Failure (Timeout). Episode terminates after 200 steps.
        if self.current_step >= self.max_steps:
            return True
        
        return False
    
    def _check_truncated(self):
        """Check if episode should truncate (timeout)"""
        # We handle timeout in terminated now
        return False

    def _get_obs(self):
        dx = self.target[0] - self.state[0]
        dy = self.target[1] - self.state[1]
        dist = math.sqrt(dx**2 + dy**2)
        
        target_angle = math.atan2(dy, dx)
        heading_err = target_angle - self.state[2]
        heading_err = (heading_err + math.pi) % (2*math.pi) - math.pi
        
        # Normalize for Neural Net stability
        # [dist/10, heading/pi, v/2.0, k/1.5]
        return np.array([
            dist / 10.0, 
            heading_err / math.pi,
            self.current_v / 2.0,
            self.current_kappa / 1.5
        ], dtype=np.float32)
        
    def get_robot_state(self):
        # Helper for visualization later
        return self.state, self.target