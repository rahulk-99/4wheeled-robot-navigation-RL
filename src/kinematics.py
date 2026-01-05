import math
import numpy as np

class FourWheelSteeringModel:
    def __init__(self, L=0.5, W=0.3, wheel_radius=0.1, 
                 max_steer_rate=2.0, max_wheel_accel=5.0):
        """
        Kinematic model of a 4-Wheel Steering robot with Double Ackermann geometry.
        
        Args:
            L (float): Half-length of the robot (distance from center to front/rear axle).
            W (float): Half-width of the robot (distance from center to left/right wheels).
            wheel_radius (float): Radius of the wheels in meters.
            max_steer_rate (float): Max steering speed in rad/s.
            max_wheel_accel (float): Max wheel angular acceleration in rad/s^2.
        """
        # Robot Geometry
        self.L = L
        self.W = W
        self.R_wheel = wheel_radius
        
        # Physical Limits
        self.max_steer_rate = max_steer_rate
        self.max_wheel_accel = max_wheel_accel
        
        # Internal State [FL, FR, RL, RR]
        # We store the actual physical state of the 4 wheels
        self.wheel_angles = np.zeros(4)      # delta (radians)
        self.wheel_ang_vels = np.zeros(4)    # phi_dot (rad/s). Spin speed
        
        # Wheel Positions relative to center (x, y)
        # Order: Front-Left, Front-Right, Rear-Left, Rear-Right
        self.wheel_positions = [
            (L, W),   # FL
            (L, -W),  # FR
            (-L, W),  # RL
            (-L, -W)  # RR
        ]

    def reset(self):
        """Resets the robot kinematics to zero."""
        self.wheel_angles = np.zeros(4)
        self.wheel_ang_vels = np.zeros(4)

    def update(self, cmd_v, cmd_kappa, dt):
        """
        Updates the wheel states based on command and time step.
        Returns the real body motion that resulted from the limited wheel movement.
        
        Args:
            cmd_v (float): Desired linear velocity (m/s).
            cmd_kappa (float): Desired curvature (1/m).
            dt (float): Time step in seconds.
            
        Returns:
            real_v (float): The actual linear velocity achieved (m/s).
            real_kappa (float): The actual curvature achieved (1/m).
        """
        
        # Inverse Kinematics: Calculate Target Wheel States
        target_angles = []
        target_ang_vels = []
        
        for i, (x, y) in enumerate(self.wheel_positions):
            # 1. Steering Angle (delta)
            # Formula: delta = arctan( kappa * x / (1 - kappa * y) )
            # Denom logic handles the "inner/outer" wheel math automatically
            denom = 1.0 - cmd_kappa * y
            # Avoid division by zero if kappa is huge. Safe check
            if abs(denom) < 1e-6: denom = 1e-6
                
            angle = math.atan2(cmd_kappa * x, denom)
            target_angles.append(angle)
            
            # 2. Wheel Spin Speed (phi_dot)
            # Formula: v_wheel = v_cmd * sqrt( (kappa*x)^2 + (1 - kappa*y)^2 )
            # This is the "Electronic Differential"
            drive_ratio = math.sqrt((cmd_kappa * x)**2 + denom**2)
            
            # If the denominator was negative (center of rotation between wheels),
            # we might need to flip velocity, but for valid bicycle model (ICR on Y axis outside width),
            # this simple magnitude check is usually sufficient. 
            v_wheel = cmd_v * drive_ratio
            
            # Convert linear m/s to angular rad/s
            w_wheel = v_wheel / self.R_wheel
            target_ang_vels.append(w_wheel)

        # 2. Dynamics: Apply Limits equivalent to the Physics Simulation
        # We cannot jump to targets instantly. We must move there over 'dt'.
        
        real_angles = []
        real_ang_vels = []
        
        for i in range(4):
            # Steer Limit
            current_angle = self.wheel_angles[i]
            tgt_angle = target_angles[i]
            
            # Find shortest path (e.g., going 350 -> 10 deg is +20, not -340)
            diff_angle = (tgt_angle - current_angle + math.pi) % (2*math.pi) - math.pi
            
            # Clip change to max_steer_rate * dt
            max_delta = self.max_steer_rate * dt
            step_angle = np.clip(diff_angle, -max_delta, max_delta)
            
            new_angle = current_angle + step_angle
            # Normalize to [-pi, pi]
            new_angle = (new_angle + math.pi) % (2*math.pi) - math.pi
            
            #Acceleration Limit
            current_w = self.wheel_ang_vels[i]
            tgt_w = target_ang_vels[i]
            
            diff_w = tgt_w - current_w
            max_dw = self.max_wheel_accel * dt
            step_w = np.clip(diff_w, -max_dw, max_dw)
            
            new_w = current_w + step_w
            
            # Store
            real_angles.append(new_angle)
            real_ang_vels.append(new_w)

        # Update internal state
        self.wheel_angles = np.array(real_angles)
        self.wheel_ang_vels = np.array(real_ang_vels)

        # 3. Forward Kinematics: Estimate Body Motion
        # Now that we know where the wheels actually are, how does the robot body move?
        # We can approximate this by reversing the equations for the Front-Left wheel
        
        # Reversing: v = (w_wheel * r) / drive_ratio
        # But drive_ratio depends on kappa. This is circular. 
 
        # The body velocity is roughly the average of wheel linear speeds.
        # The body curvature is determined by the steering angles.
        
        # Let's take the effective curvature from the Front Left wheel
        # tan(delta) = kappa * L / (1 - kappa * W)  -> solve for kappa
        # kappa = tan(delta) / (L + W * tan(delta))
        
        # Use Front Left (Index 0)
        fl_angle = self.wheel_angles[0]
        fl_w = self.wheel_ang_vels[0]
        fl_v = fl_w * self.R_wheel
        
        # Avoid div by zero
        tan_d = math.tan(fl_angle)
        denom_k = self.L + self.W * tan_d
        
        if abs(denom_k) < 1e-4:
            real_kappa = 0.0
        else:
            real_kappa = tan_d / denom_k
            
        # Real Velocity 
        # Based on how fast wheels are spinning vs the curvature ratio
        # v_body = v_wheel / sqrt(...)
        # Recalculate geometric ratio with the REAL kappa
        ratio = math.sqrt((real_kappa * self.L)**2 + (1 - real_kappa * self.W)**2)
        if ratio < 1e-4:
            real_v = fl_v 
        else:
            real_v = fl_v / ratio
            
        return real_v, real_kappa
        
    def get_wheel_states(self):
        """Returns current (angles, velocities) for visualization."""
        return self.wheel_angles, self.wheel_ang_vels

# QUICK TEST 
if __name__ == "__main__":
    # If run `python kinematics.py`, this runs to verify logic.
    model = FourWheelSteeringModel()
    
    print("Testing Kinematics...")
    print(f"Initial State: {model.wheel_angles}")
    
    # Command: Go Forward (1 m/s), Turn Left (kappa=0.5 which is 2m radius)
    v_cmd = 1.0
    k_cmd = 0.5
    dt = 0.1
    
    print(f"\nSending Command: v={v_cmd}, kappa={k_cmd}")
    
    # Run for 1 second (10 steps)
    for i in range(10):
        real_v, real_k = model.update(v_cmd, k_cmd, dt)
        fl_angle = model.wheel_angles[0]
        print(f"Step {i+1}: FL_Angle={fl_angle:.3f} rad, Real_v={real_v:.3f}, Real_k={real_k:.3f}")
        
    print("\nNotice how the angle ramps up gradually due to 'max_steer_rate'.")