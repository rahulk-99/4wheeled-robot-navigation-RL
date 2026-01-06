import math
import time
import os
# mcap library
from mcap.writer import Writer
# standard ROS 2 serialization
from rclpy.serialization import serialize_message
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
import numpy as np

def get_ros_msg_definition(msg_type):
    """
    Building a full 'ros2msg' definition.
    Format:
    <MainMsgDef>
    MSG: <Dep1Type>
    <Dep1Def>
    ...
    """
    
    # Cache to avoid re-reading files
    seen_deps = set()
    definitions = []
    
    # Queue of msg_name to process. msg_type example "visualization_msgs/msg/MarkerArray"
    parts = msg_type.split('/')
    if len(parts) == 3:
        pkg, _, name = parts
    elif len(parts) == 2: # sometimes referred as pkg/Msg
        pkg, name = parts
    else:
        return b""
        
    queue = [(pkg, name)]
    
    # Function to find .msg file
    def find_msg_file(package, msg_name):
        base_paths = ["/opt/ros/humble/share", "/opt/ros/foxy/share", "/usr/share"]
        if "AMENT_PREFIX_PATH" in os.environ:
             base_paths = os.environ["AMENT_PREFIX_PATH"].split(":") + base_paths
             
        for base in base_paths:
            path = os.path.join(base, package, "msg", f"{msg_name}.msg")
            if os.path.exists(path):
                return path
        return None

    main_def = ""
    
    while queue:
        curr_pkg, curr_name = queue.pop(0)
        
        # Unique key
        key = (curr_pkg, curr_name)
        if key in seen_deps and key != (pkg, name):
            continue
        seen_deps.add(key)
        
        path = find_msg_file(curr_pkg, curr_name)
        if not path:
            print(f"Warning: Could not find definition for {curr_pkg}/{curr_name}")
            continue
            
        with open(path, 'r') as f:
            content = f.read()
            
        # For the main message, appending content.
        # For dependencies, we prepend "MSG: pkg/Msg\n" (standard ros2msg format)
        if key == (pkg, name):
            main_def = content
        else:
            definitions.append(f"MSG: {curr_pkg}/{curr_name}\n{content}")
            
        # Parse for dependencies to add to queue
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Simple parser: type field
            parts = line.split()
            if not parts: continue
            typ = parts[0]
            
            # Remove array brackets []
            if '[' in typ:
                typ = typ.split('[')[0]
                
            # Check primitives in ROS2 (designator)
            primitives = ['bool', 'byte', 'char', 'float32', 'float64', 'int8', 'uint8', 
                          'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'string', 'wstring']
            
            if typ in primitives:
                continue
                
            # It's a complex type
            if '/' in typ:
                dep_pkg, dep_name = typ.split('/')
            else:
                dep_pkg = curr_pkg
                dep_name = typ
                # Handle special case: Header is usually std_msgs/Header
                if typ == 'Header':
                    dep_pkg = 'std_msgs'
                    dep_name = 'Header'
            
            # Enqueue
            if (dep_pkg, dep_name) not in seen_deps:
                queue.append((dep_pkg, dep_name))

    # Combine
    full_text = main_def.strip()
    separator = "=" * 80
    
    for dep_def in definitions:
        full_text += f"\n{separator}\n{dep_def.strip()}"
        
    return full_text.encode('utf-8')


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
        
        # 2. Register Schemas with ros2msg definitions
        self.marker_schema_id = self.writer.register_schema(
            name="visualization_msgs/msg/MarkerArray", 
            encoding="ros2msg", 
            data=get_ros_msg_definition("visualization_msgs/msg/MarkerArray")
        )
        self.odom_schema_id = self.writer.register_schema(
            name="nav_msgs/msg/Odometry", 
            encoding="ros2msg", 
            data=get_ros_msg_definition("nav_msgs/msg/Odometry")
        )
        self.tf_schema_id = self.writer.register_schema(
            name="tf2_msgs/msg/TFMessage", 
            encoding="ros2msg", 
            data=get_ros_msg_definition("tf2_msgs/msg/TFMessage")
        )
        
        # 3. Register Channels
        # Marker channel
        self.marker_channel_id = self.writer.register_channel(
            schema_id=self.marker_schema_id,
            topic="/visualization",
            message_encoding="cdr"
        )
        
        # TF channel
        self.tf_channel_id = self.writer.register_channel(
            schema_id=self.tf_schema_id,
            topic="/tf",
            message_encoding="cdr"
        )
        
        # Dictionary to store odom channel IDs per robot
        self.odom_channels = {}
        
        print(f"Visualization started. Output: {self.filename}")

    def close(self):
        self.writer.finish()
        self.file.close()
        print(f"Visualization saved to {self.filename}")

    def get_odom_channel(self, robot_id):
        if robot_id not in self.odom_channels:
            topic = f"/robot{robot_id}/odom"
            self.odom_channels[robot_id] = self.writer.register_channel(
                schema_id=self.odom_schema_id,
                topic=topic,
                message_encoding="cdr"
            )
        return self.odom_channels[robot_id]

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
        
        # Marker Array (Visualization)
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
            channel_id=self.marker_channel_id,
            log_time=t_nanos,
            data=serialized_data,
            publish_time=t_nanos
        )

        # 2. TF & Odometry 
        # Frame names
        map_frame = "map"
        base_frame = f"robot{robot_id}_base_link"
        
        # Create TF Map -> Base Link
        tf_msg = TFMessage()
        ts = TransformStamped()
        ts.header.frame_id = map_frame
        ts.header.stamp = body.header.stamp
        ts.child_frame_id = base_frame
        ts.transform.translation.x = rx
        ts.transform.translation.y = ry
        ts.transform.translation.z = 0.0
        ts.transform.rotation = body.pose.orientation # Reuse quaternion from marker
        tf_msg.transforms.append(ts)
        
        self.writer.add_message(
            channel_id=self.tf_channel_id,
            log_time=t_nanos,
            data=serialize_message(tf_msg),
            publish_time=t_nanos
        )
        
        # Create Odometry
        odom = Odometry()
        odom.header.frame_id = map_frame
        odom.header.stamp = body.header.stamp
        odom.child_frame_id = base_frame
        
        # Copy translation (Vector3) to position (Point)
        odom.pose.pose.position.x = ts.transform.translation.x
        odom.pose.pose.position.y = ts.transform.translation.y
        odom.pose.pose.position.z = ts.transform.translation.z
        
        odom.pose.pose.orientation = ts.transform.rotation
        
        odom_channel_id = self.get_odom_channel(robot_id)
        self.writer.add_message(
            channel_id=odom_channel_id,
            log_time=t_nanos,
            data=serialize_message(odom),
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
