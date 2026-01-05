import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

# Import physics formulation
from .kinematics import FourWheelSteeringModel

class MultiRobotEnv(gym.Env):
    """
    Multi-robot environment with collision avoidance.
    - Robot 1 (Agent): Learns via PPO
    - Robots 2 & 3 (Obstacles): Follow simple policies toward their goals
    """
    def __init__(self):
        super(MultiRobotEnv, self).__init__()
        
        # Action Space - Only for Robot 1 (the learning agent)
        self.action_space = spaces.Box(
            low=np.array([-1.5, 0.0], dtype=np.float32),
            high=np.array([1.5, 2.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation Space - 12D
        # [agent_dist, agent_heading, agent_v, agent_k,
        #  robot2_rel_x, robot2_rel_y, robot2_vx, robot2_vy,
        #  robot3_rel_x, robot3_rel_y, robot3_vx, robot3_vy]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.pi, 0.0, -1.5,
                         -20.0, -20.0, -2.0, -2.0,
                         -20.0, -20.0, -2.0, -2.0], dtype=np.float32),
            high=np.array([20.0, np.pi, 2.0, 1.5,
                          20.0, 20.0, 2.0, 2.0,
                          20.0, 20.0, 2.0, 2.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Create 3 robots
        self.robot1 = FourWheelSteeringModel()  # Learning agent
        self.robot2 = FourWheelSteeringModel()  # Obstacle 1
        self.robot3 = FourWheelSteeringModel()  # Obstacle 2
        
        # States: [x, y, theta, v, kappa]
        self.state1 = np.zeros(5)
        self.state2 = np.zeros(5)
        self.state3 = np.zeros(5)
        
        # Targets
        self.target1 = np.array([5.0, 2.0])
        self.target2 = np.array([5.0, -2.0])
        self.target3 = np.array([8.0, 0.0])
        
        self.dt = 0.1
        self.max_steps = 300
        self.current_step = 0
        
        # Collision parameters
        self.robot_radius = 0.6  # approx bounding circle for 1.0x0.6m robot
        self.safety_distance = 2 * self.robot_radius
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset all robots
        self.robot1.reset()
        self.robot2.reset()
        self.robot3.reset()
        
        self.current_step = 0
        
        # Helper to generate valid random pose
        def get_random_pose(existing_poses=[], min_dist=2.0):
            while True:
                # Random x, y in range [-8, 8] x [-6, 6]
                x = np.random.uniform(-8.0, 8.0)
                y = np.random.uniform(-6.0, 6.0)
                theta = np.random.uniform(-np.pi, np.pi)
                
                valid = True
                for px, py in existing_poses:
                    if math.sqrt((x-px)**2 + (y-py)**2) < min_dist:
                        valid = False
                        break
                if valid:
                    return np.array([x, y, theta, 0.0, 0.0])

        # 1. Randomize Start Positions (ensure separation)
        start_poses = []
        
        # Agent
        self.state1 = get_random_pose(start_poses)
        start_poses.append((self.state1[0], self.state1[1]))
        
        # Obstacle 1
        self.state2 = get_random_pose(start_poses)
        start_poses.append((self.state2[0], self.state2[1]))
        
        # Obstacle 2
        self.state3 = get_random_pose(start_poses)
        start_poses.append((self.state3[0], self.state3[1]))

        # Helper to generate valid target
        def get_random_target(start_pos, min_dist=4.0):
            while True:
                tx = np.random.uniform(-9.0, 9.0)
                ty = np.random.uniform(-7.0, 7.0)
                # Ensure target is far enough from start
                if math.sqrt((tx-start_pos[0])**2 + (ty-start_pos[1])**2) > min_dist:
                    return np.array([tx, ty])

        # 2. Randomize Targets
        self.target1 = get_random_target(self.state1)
        self.target2 = get_random_target(self.state2)
        self.target3 = get_random_target(self.state3)
        
        # Track previous distance for agent
        dx = self.target1[0] - self.state1[0]
        dy = self.target1[1] - self.state1[1]
        self.prev_dist = math.sqrt(dx**2 + dy**2)
        
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        """Execute one timestep"""
        self.current_step += 1
        
        # 1. Agent (Robot 1) takes learned action
        k_cmd1, v_cmd1 = float(action[0]), float(action[1])
        self._update_robot(self.robot1, self.state1, v_cmd1, k_cmd1)
        
        # 2. Obstacles (Robots 2 & 3) follow simple policies
        # They now avoid EACH OTHER and the AGENT
        
        # Robot 2 avoids [Robot 1 (Agent), Robot 3]
        k_cmd2, v_cmd2 = self._simple_policy(self.state2, self.target2, [self.state1, self.state3])
        self._update_robot(self.robot2, self.state2, v_cmd2, k_cmd2)
        
        # Robot 3 avoids [Robot 1 (Agent), Robot 2]
        k_cmd3, v_cmd3 = self._simple_policy(self.state3, self.target3, [self.state1, self.state2])
        self._update_robot(self.robot3, self.state3, v_cmd3, k_cmd3)
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward
        reward = self._compute_reward(obs)
        
        # Check termination
        terminated = self._check_terminated(obs)
        truncated = False
        
        return obs, reward, terminated, truncated, {}
    
    def _simple_policy(self, state, target, other_states):
        """
        Simple proportional controller for obstacle robots.
        Now stops at goal and slows down/stops near ANY other robot.
        """
        dx = target[0] - state[0]
        dy = target[1] - state[1]
        dist_to_goal = math.sqrt(dx**2 + dy**2)
        
        # STRICT STOPPING LOGIC: Force stop at goal
        if dist_to_goal < 0.5:
            return 0.0, 0.0
        
        target_heading = math.atan2(dy, dx)
        heading_error = target_heading - state[2]
        # Normalize to [-pi, pi]
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        
        # Simple P-controller for steering
        k_cmd = np.clip(1.2 * heading_error, -1.5, 1.5)
        
        # Velocity control
        v_cmd = 1.0
        
        # Slow down when approaching goal
        if dist_to_goal < 2.0:
            v_cmd = max(0.2, dist_to_goal * 0.4)
        
        # COLLISION AVOIDANCE LOGIC
        # Check against ALL other robots (Agent + Other Obstacle)
        closest_dist = float('inf')
        
        for other_state in other_states:
            dist_to_other = math.sqrt((state[0] - other_state[0])**2 + 
                                     (state[1] - other_state[1])**2)
            if dist_to_other < closest_dist:
                closest_dist = dist_to_other
        
        # Logic:
        # < 1.0m: EMERGENCY STOP (Don't hit them!)
        # < 2.0m: Slow down drastically
        
        if closest_dist < 1.0:
            v_cmd = 0.0  # Stop to avoid collision
        elif closest_dist < 2.0:
            v_cmd = min(v_cmd, 0.2)  # Crawl
            
        return k_cmd, v_cmd
    
    def _update_robot(self, robot_model, state, v_cmd, k_cmd):
        """Update a single robot's state"""
        real_v, real_k = robot_model.update(v_cmd, k_cmd, self.dt)
        
        theta = state[2]
        new_theta = theta + real_v * real_k * self.dt
        new_x = state[0] + real_v * math.cos(theta) * self.dt
        new_y = state[1] + real_v * math.sin(theta) * self.dt
        
        state[0] = new_x
        state[1] = new_y
        state[2] = new_theta
        state[3] = real_v
        state[4] = real_k
    
    def _check_collision(self):
        """Check if agent collides with any obstacle robot"""
        # Agent vs Robot 2
        dist_12 = math.sqrt((self.state1[0] - self.state2[0])**2 + 
                           (self.state1[1] - self.state2[1])**2)
        if dist_12 < self.safety_distance:
            return True
        
        # Agent vs Robot 3
        dist_13 = math.sqrt((self.state1[0] - self.state3[0])**2 + 
                           (self.state1[1] - self.state3[1])**2)
        if dist_13 < self.safety_distance:
            return True
        
        return False
    
    def _get_obs(self):
        """Get observation for the learning agent"""
        # Agent's goal-related observations
        dx1 = self.target1[0] - self.state1[0]
        dy1 = self.target1[1] - self.state1[1]
        dist1 = math.sqrt(dx1**2 + dy1**2)
        
        target_angle = math.atan2(dy1, dx1)
        heading_err = target_angle - self.state1[2]
        heading_err = (heading_err + math.pi) % (2*math.pi) - math.pi
        
        # Relative positions and velocities of obstacle robots
        rel_x2 = self.state2[0] - self.state1[0]
        rel_y2 = self.state2[1] - self.state1[1]
        vx2 = self.state2[3] * math.cos(self.state2[2])
        vy2 = self.state2[3] * math.sin(self.state2[2])
        
        rel_x3 = self.state3[0] - self.state1[0]
        rel_y3 = self.state3[1] - self.state1[1]
        vx3 = self.state3[3] * math.cos(self.state3[2])
        vy3 = self.state3[3] * math.sin(self.state3[2])
        
        # Normalize
        return np.array([
            dist1 / 10.0,
            heading_err / math.pi,
            self.state1[3] / 2.0,
            self.state1[4] / 1.5,
            rel_x2 / 10.0,
            rel_y2 / 10.0,
            vx2 / 2.0,
            vy2 / 2.0,
            rel_x3 / 10.0,
            rel_y3 / 10.0,
            vx3 / 2.0,
            vy3 / 2.0
        ], dtype=np.float32)

    def _compute_reward(self, obs):
        """Compute reward for the agent"""
        # Un-normalize
        dist = obs[0] * 10.0
        heading_err = obs[1] * math.pi
        abs_heading_err = abs(heading_err)
        
        # 1. Progress reward
        dist_delta = self.prev_dist - dist
        reward_progress = dist_delta * 10.0
        self.prev_dist = dist
        
        # 2. Heading reward
        heading_alignment = 1.0 - (abs_heading_err / math.pi)
        reward_heading = (heading_alignment - 0.5) * 1.0
        
        # 3. Time penalty
        reward_time = -0.05 * (1.0 + dist / 10.0)
        
        # 4. Velocity reward
        if dist < 0.3:
            reward_vel = -self.state1[3] * 10.0
            if self.state1[3] < 0.2:
                reward_vel += 30.0
        elif dist < 1.0:
            v_target = dist * 0.5
            reward_vel = -abs(self.state1[3] - v_target) * 2.0
        else:
            heading_factor = max(0.0, heading_alignment)
            reward_vel = self.state1[3] * 0.3 * heading_factor
        
        # 5. Curvature penalty
        reward_curvature = -abs(self.state1[4]) * 0.1
        
        # 6. Safety / Collision
        reward_collision = 0.0
        
        # Hard collision
        if self._check_collision():
            reward_collision = -500.0  # Huge penalty
            
        # Proximity penalty (Soft constraint)
        # Penalize if closer than 1.5m to any obstacle
        min_dist_obs = float('inf')
        
        d2 = math.sqrt((self.state1[0]-self.state2[0])**2 + (self.state1[1]-self.state2[1])**2)
        d3 = math.sqrt((self.state1[0]-self.state3[0])**2 + (self.state1[1]-self.state3[1])**2)
        min_dist_obs = min(d2, d3)
        
        if min_dist_obs < 1.5:
             # Exponential penalty as it gets closer
             reward_collision -= 10.0 * (1.5 - min_dist_obs)
        
        # 7. Success/Failure
        reward_terminal = 0.0
        is_at_target = dist < 0.5  # Slightly relaxed goal radius
        is_stopped = self.state1[3] < 0.2
        
        if is_at_target and is_stopped:
            reward_terminal = 500.0
        elif abs(self.state1[0]) > 10 or abs(self.state1[1]) > 10:
            reward_terminal = -10.0
        elif self.current_step >= self.max_steps:
            reward_terminal = -5.0
        
        # Total reward
        reward = (reward_progress + reward_heading + reward_time + 
                 reward_vel + reward_curvature + reward_collision + reward_terminal)
        
        return reward
    
    def _check_terminated(self, obs):
        """Check termination conditions"""
        dist = obs[0] * 10.0
        
        # Success
        if dist < 0.3 and self.state1[3] < 0.2:
            return True
        
        # Collision
        if self._check_collision():
            return True
        
        # Out of bounds
        if abs(self.state1[0]) > 10 or abs(self.state1[1]) > 10:
            return True
        
        # Timeout
        if self.current_step >= self.max_steps:
            return True
        
        return False
    
    def get_all_robot_states(self):
        """Helper for visualization - returns all robot states and targets"""
        return (
            (self.state1, self.target1),
            (self.state2, self.target2),
            (self.state3, self.target3)
        )
