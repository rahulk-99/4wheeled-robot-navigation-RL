import math
import time
# mcap library
from mcap.writer import Writer
# standard ROS 2 serialization
from rclpy.serialization import serialize_message
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np

class McapVisualizer:
    def __init__(self, filename="simulation.mcap", L=0.5, W=0.3, wheel_radius=0.1):
        self.filename = filename
        self.L = L
        self.W = W
        self.wheel_radius = wheel_radius
        
        # 1. Open File
        self.file = open(self.filename, "wb")
        self.writer = Writer(self.file)
        self.writer.start()
        
        # 2. Register MCAP MarkerArrays
        self.schema_id = self.writer.register_schema(
            name="visualization_msgs/msg/MarkerArray",
            encoding="ros2msg",
            data=b"" 
        )
        
        # 3. Register Channel - Topic name
        self.channel_id = self.writer.register_channel(
            schema_id=self.schema_id,
            topic="/visualization",
            message_encoding="cdr"
        )
        
        print(f"Visualization started. Output: {self.filename}")

    def close(self):
        self.writer.finish()
        self.file.close()
        print(f"Visualization saved to {self.filename}")

    def log_frame(self, t_nanos, robot_state, target_pos, wheel_angles, robot_id=1):
        """
        Log a robot frame to MCAP.
        robot_id: 1 (green), 2 (red), 3 (blue) for multi-robot scenarios
        """
        rx, ry, rtheta = robot_state
        
        # Color mapping for different robots
        colors = {
            1: (1.0, 0.0, 0.0),  # Red (learning agent)
            2: (0.0, 1.0, 0.0),  # Green (obstacle 1 - top lane)
            3: (0.0, 1.0, 0.0)   # Green (obstacle 2 - bottom lane)
        }
        body_color = colors.get(robot_id, (0.5, 0.5, 0.5))
        
        marker_array = MarkerArray()
        
        # ROBOT BODY
        body = Marker()
        body.header.frame_id = "map"
        body.header.stamp.sec = int(t_nanos / 1e9)
        body.header.stamp.nanosec = int(t_nanos % 1e9)
        body.ns = f"robot{robot_id}_body"  # Unique namespace per robot
        body.id = robot_id
        body.type = Marker.CUBE
        body.action = Marker.ADD
        
        body.pose.position.x = rx
        body.pose.position.y = ry
        body.pose.position.z = self.wheel_radius
        
        q = self.euler_to_quat(0, 0, rtheta)
        body.pose.orientation.x = q[0]
        body.pose.orientation.y = q[1]
        body.pose.orientation.z = q[2]
        body.pose.orientation.w = q[3]
        
        body.scale.x = self.L * 2.0
        body.scale.y = self.W * 2.0
        body.scale.z = 0.2
        body.color.r = body_color[0]
        body.color.g = body_color[1]
        body.color.b = body_color[2]
        body.color.a = 0.7
        marker_array.markers.append(body)

        # TARGET - Same color as robot
        target = Marker()
        target.header = body.header
        target.ns = f"robot{robot_id}_target"
        target.id = 100 + robot_id
        target.type = Marker.SPHERE
        target.action = Marker.ADD
        
        target.pose.position.x = target_pos[0]
        target.pose.position.y = target_pos[1]
        target.pose.position.z = 0.0
        
        target.scale.x = 0.4; target.scale.y = 0.4; target.scale.z = 0.4
        target.color.r = body_color[0]
        target.color.g = body_color[1]  
        target.color.b = body_color[2]
        target.color.a = 1.0
        marker_array.markers.append(target)

        # WHEELS & LINKS
        # Offset wheels outside the body (W + 0.15) so rotation is visible
        wheel_W = self.W + 0.15
        rel_pos = [(self.L, wheel_W), (self.L, -wheel_W), (-self.L, wheel_W), (-self.L, -wheel_W)]
        
        for i, (lx, ly) in enumerate(rel_pos):
            wx = rx + (lx * math.cos(rtheta) - ly * math.sin(rtheta))
            wy = ry + (lx * math.sin(rtheta) + ly * math.cos(rtheta))
            
            # Link (Axle) - draw from body center-ish to wheel
            # Body edge y is self.W * sign(ly)
            body_y_local = self.W if ly > 0 else -self.W
            bx = rx + (lx * math.cos(rtheta) - body_y_local * math.sin(rtheta))
            by = ry + (lx * math.sin(rtheta) + body_y_local * math.cos(rtheta))
            
            link = Marker()
            link.header = body.header
            link.ns = f"robot{robot_id}_links"
            link.id = robot_id * 100 + 10 + i
            link.type = Marker.LINE_LIST
            link.scale.x = 0.05
            link.color.r = body_color[0]; link.color.g = body_color[1]; link.color.b = body_color[2]; link.color.a = 1.0
            link.points = [Point(x=bx, y=by, z=self.wheel_radius), Point(x=wx, y=wy, z=self.wheel_radius)]
            marker_array.markers.append(link)
            
            # Wheel
            wheel = Marker()
            wheel.header = body.header
            wheel.ns = f"robot{robot_id}_wheels"
            wheel.id = robot_id * 100 + 20 + i
            wheel.type = Marker.CYLINDER
            wheel.pose.position.x = wx
            wheel.pose.position.y = wy
            wheel.pose.position.z = self.wheel_radius
            
            # Rotate 90 deg around X to make cylinder horizontal
            wq = self.euler_to_quat(math.pi/2, 0, rtheta + wheel_angles[i])
            wheel.pose.orientation.x = wq[0]; wheel.pose.orientation.y = wq[1]
            wheel.pose.orientation.z = wq[2]; wheel.pose.orientation.w = wq[3]
            
            # Make bigger: 2.5x radius
            wheel.scale.x = self.wheel_radius * 2.5
            wheel.scale.y = self.wheel_radius * 2.5
            wheel.scale.z = 0.15 # Thicker width
            
            # Same color as body
            wheel.color.r = body_color[0]; wheel.color.g = body_color[1]; wheel.color.b = body_color[2]; wheel.color.a = 1.0
            marker_array.markers.append(wheel)


        # 1. Serialize message to bytes using standard ROS 2 serializer
        serialized_data = serialize_message(marker_array)
        
        # 2. Write bytes to MCAP
        self.writer.add_message(
            channel_id=self.channel_id,
            log_time=t_nanos,
            data=serialized_data,
            publish_time=t_nanos
        )

    def euler_to_quat(self, roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]

# Testing Kinematics modeling using mcap markers
if __name__ == "__main__":
    from kinematics import FourWheelSteeringModel
    
    print("Testing Visualization Generation...")
    viz = McapVisualizer("test_vis.mcap")
    robot = FourWheelSteeringModel()
    
    x, y, theta = 0.0, 0.0, 0.0
    dt = 0.1
    current_time = 0
    target = [5.0, 2.0]
    
    for i in range(50):
        v, k = robot.update(1.0, 0.5, dt)
        theta += v * k * dt
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        
        angles, _ = robot.get_wheel_states()
        t_nanos = int(current_time * 1e9)
        
        viz.log_frame(t_nanos, [x, y, theta], target, angles)
        current_time += dt
        
    viz.close()
    print("mcap recorded! Open 'test_vis.mcap' in rviz2/Foxglove to validate.")
