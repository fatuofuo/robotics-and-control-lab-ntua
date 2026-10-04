#!/usr/bin/env python3
"""
=============================================================================
ROBOTICS II - NTUA
Redundant Manipulator Control - Trajectory following with Obstacle Avoidance
STUDENT TEMPLATE - COMPLETED VERSION
=============================================================================

ASSIGNMENT OBJECTIVE:
Implement a kinematic controller for a 7-DOF robot manipulator that:
1. Tracks a linear trajectory between points PA and PB (PRIMARY TASK)
2. Avoids cylindrical obstacles using null-space control (SECONDARY TASK)

The end-effector performs periodic linear motion (position control only,
NOT orientation) while the redundant DOF is used for obstacle avoidance.

WHAT YOU NEED TO IMPLEMENT:
- compute_jacobian(): Geometric Jacobian matrix (3x7 for position)
- compute_primary_task_desired_velocity(): Desired end-effector velocity
- compute_secondary_task_desired_velocity(): Repulsive velocity in null-space
- control_loop(): Main control law combining primary and secondary tasks

PROVIDED FOR YOU:
- Forward kinematics (get_end_effector_position, get_link_positions)
- DH transformation matrix (dh_transform)
- Distance calculations (distance_point_to_cylinder, get_min_obstacle_distance)
- ROS2 infrastructure (publishers, visualization)
- Configuration loading from params.yaml

Author: Robotics II Course - NTUA
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, Vector3, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from ament_index_python.packages import get_package_share_directory

import numpy as np
from numpy.linalg import pinv, norm, inv
from math import sin, cos, pi, sqrt
from datetime import datetime
from collections import deque
import yaml
import os


# =============================================================================
#                                Franka Emika Panda
# =============================================================================
# Standard DH Convention: T = Rot_z(θ) * Trans_z(d) * Trans_x(a) * Rot_x(α)
# =============================================================================

STANDARD_DH_PARAMS = [
    # [a,        d,      alpha,    theta_offset]
    [0,        0.333,  -pi/2,    0],      # Joint 1
    [0,        0,       pi/2,    0],      # Joint 2
    [0.0825,  0.316,   pi/2,    0],      # Joint 3
    [-0.0825, 0,      -pi/2,    0],      # Joint 4
    [0,       0.384,   pi/2,    0],      # Joint 5
    [0.088,   0,       pi/2,    0],      # Joint 6
    [0,       0.107,   0,       0],      # Joint 7 (includes flange)
]

# =============================================================================
# URDF JOINT DEFINITIONS - for correct capsule frame transforms
# =============================================================================
URDF_JOINTS = [
    ([0,    0,      0.333],  0), #joint 1
    ([0,    0,      0],     -pi/2), #joint 2
    ([0,   -0.316,  0],      pi/2), #joint 3
    ([0.0825, 0,    0],      pi/2), #joint 4
    ([-0.0825, 0.384, 0],  -pi/2), #joint 5
    ([0,    0,      0],      pi/2), #joint 6
    ([0.088, 0,     0],      pi/2), #joint 7
]

JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
CAPSULE_SAFETY_DISTANCE = 0.0

class CollisionCapsule:
    """
    Collision capsule (cylinder with hemispherical caps) from Franka XACRO.
    Each link has one or more capsules describing its actual collision geometry.
    """
    def __init__(self, xyz, radius, length, direction='z'):
        self.xyz = np.array(xyz)
        self.radius = radius
        self.length = length
        self.direction = direction

    def get_endpoints_in_link_frame(self):
        """Get the two endpoints of the capsule centerline in link frame."""
        half_len = self.length / 2.0
        if self.direction == 'x':
            offset = np.array([half_len, 0.0, 0.0])
        elif self.direction == 'y':
            offset = np.array([0.0, half_len, 0.0])
        else:  # 'z'
            offset = np.array([0.0, 0.0, half_len])
        return self.xyz - offset, self.xyz + offset

COLLISION_CAPSULES = {
    0: [CollisionCapsule([-0.075, 0, 0.06],    0.06  + CAPSULE_SAFETY_DISTANCE, 0.03,  'x')],
    1: [CollisionCapsule([0, 0, -0.1915],       0.06  + CAPSULE_SAFETY_DISTANCE, 0.283, 'z')],
    2: [CollisionCapsule([0, 0, 0],             0.06  + CAPSULE_SAFETY_DISTANCE, 0.12,  'z')],
    3: [CollisionCapsule([0, 0, -0.145],        0.06  + CAPSULE_SAFETY_DISTANCE, 0.15,  'z')],
    4: [CollisionCapsule([0, 0, 0],             0.06  + CAPSULE_SAFETY_DISTANCE, 0.12,  'z')],
    5: [CollisionCapsule([0, 0, -0.26],         0.06  + CAPSULE_SAFETY_DISTANCE, 0.10,  'z'),
        CollisionCapsule([0, 0.08, -0.13],      0.025 + CAPSULE_SAFETY_DISTANCE, 0.14,  'z')],
    6: [CollisionCapsule([0, 0, -0.03],         0.05  + CAPSULE_SAFETY_DISTANCE, 0.08,  'z')],
    7: [CollisionCapsule([0, 0, 0.01],          0.04  + CAPSULE_SAFETY_DISTANCE, 0.14,  'z'),
        CollisionCapsule([0.06, 0, 0.082],      0.03  + CAPSULE_SAFETY_DISTANCE, 0.01,  'x')],
}

class RedundantController(Node):
    """
    Kinematic controller for 7-DOF Panda manipulator with obstacle avoidance.
    Students implement the core control algorithms.
    """
    def __init__(self):
        super().__init__('redundant_controller')
        self.load_parameters()
        self.q = np.array(self.config['initial_joint_positions'])
        self.q_dot = np.zeros(7)
        self.trajectory_param = 0.0
        self.trajectory_direction = 1
        
        # Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_markers', 10)
        self.dt = 1.0 / self.config['control']['rate']
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        self.bypass = False # Set to False once control_loop is implemented
        
        self.log_dir = '/ros2_ws/src/logs'
        os.makedirs(self.log_dir, exist_ok=True)
        max_log_entries = 100000
        self.log = {
            'time': deque(maxlen=max_log_entries),
            'position': deque(maxlen=max_log_entries),
            'joint_velocities': deque(maxlen=max_log_entries), 
            'manipulability': deque(maxlen=max_log_entries),
            'min_distance': deque(maxlen=max_log_entries) 
        }
        self.start_time = self.get_clock().now()
        
        self.get_logger().info('='*60)
        self.get_logger().info('Redundant Controller - STUDENT VERSION')
        self.get_logger().info('='*60)
        self.get_logger().info(f'Trajectory: PA={self.PA} → PB={self.PB}')
        self.get_logger().info(f'Speed: {self.speed} m/s')
        self.get_logger().info(f'Obstacles: {len(self.obstacles)}')
        self.get_logger().info(f'Logs will be saved to: {self.log_dir}')
        self.get_logger().info('='*60)

    def load_parameters(self):
        try:
            pkg_share = get_package_share_directory('panda_redundant_controller_student')
            config_path = os.path.join(pkg_share, 'config', 'params.yaml')
            self.get_logger().info(f'Loading config from: {config_path}')
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                self.config = full_config['panda_controller']['ros__parameters']
        except Exception as e:
            self.get_logger().warn(f'Could not load config: {e}. Using defaults.')
            self.config = self._default_config()
        
        self.PA = np.array(self.config['trajectory']['PA'])
        self.PB = np.array(self.config['trajectory']['PB'])
        self.speed = self.config['trajectory']['speed']
        self.Kp = self.config['control']['Kp']
        self.Ko = self.config['control']['Ko']
        self.d_influence = self.config['control']['d_influence']
        self.damping = self.config['control']['damping']
        self.max_joint_vel = self.config['control']['max_joint_velocity']
        self.max_null_vel = self.config['control']['max_null_velocity']
        self.link_radius = self.config['robot']['link_radius']
        
        self.obstacles = []
        for pos in self.config['obstacles']['positions']:
            self.obstacles.append({
                'center': np.array(pos),
                'radius': self.config['obstacles']['radius'],
                'height': self.config['obstacles']['height']
            })

    def _default_config(self):
        return {
            'trajectory': {'PA': [0.617, -0.40, 0.199], 'PB': [0.617, 0.40, 0.199], 'speed': 0.1},
            'obstacles': {'radius': 0.05, 'height': 1.0, 'positions': [[0.30, -0.20, 0.50], [0.30, 0.20, 0.50]]},
            'robot': {'link_radius': 0.063},
            'control': {'rate': 100.0, 'Kp': 2.0, 'Ko': 0.8, 'd_influence': 0.25,
                       'damping': 0.01, 'max_joint_velocity': 0.5, 'max_null_velocity': 0.3},
            'initial_joint_positions': [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785],
            'visualization': {'gripper_opening': 0.04}
        }

    # =========================================================================
    # FORWARD KINEMATICS (PROVIDED)
    # =========================================================================
    def dh_transform(self, a, d, alpha, theta):
        ca, sa = cos(alpha), sin(alpha)
        ct, st = cos(theta), sin(theta)
        return np.array([
            [ct,     -st*ca,   st*sa,   a*ct],
            [st,      ct*ca,  -ct*sa,   a*st],
            [0,        sa,      ca,      d   ],
            [0,        0,       0,       1   ]
        ])

    def get_all_transforms(self, q):
        transforms = [np.eye(4)]
        T = np.eye(4)
        for i in range(len(STANDARD_DH_PARAMS)):
            a, d, alpha, theta_offset = STANDARD_DH_PARAMS[i]
            theta = theta_offset + q[i]
            T = T @ self.dh_transform(a, d, alpha, theta)
            transforms.append(T.copy())
        return transforms

    def get_end_effector_position(self, q):
        transforms = self.get_all_transforms(q)
        return transforms[-1][:3, 3]

    def get_link_positions(self, q):
        transforms = self.get_all_transforms(q)
        return [T[:3, 3] for T in transforms]

    def get_all_urdf_transforms(self, q):
        transforms = [np.eye(4)]
        T = np.eye(4)
        for i in range(7):
            xyz, roll = URDF_JOINTS[i]
            cr, sr = cos(roll), sin(roll)
            T_origin = np.array([
                [1.0, 0.0, 0.0, xyz[0]],
                [0.0,  cr, -sr, xyz[1]],
                [0.0,  sr,  cr, xyz[2]],
                [0.0, 0.0, 0.0,   1.0],
            ])
            cq, sq = cos(q[i]), sin(q[i])
            T_joint = np.array([
                [cq, -sq, 0.0, 0.0],
                [sq,  cq, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            T = T @ T_origin @ T_joint
            transforms.append(T.copy())
        return transforms

    # =========================================================================
    # DISTANCE CALCULATIONS (PROVIDED)
    # =========================================================================
    def distance_point_to_cylinder(self, point, cyl_center, cyl_radius, cyl_height):
        z_min = cyl_center[2] - cyl_height / 2.0
        z_max = cyl_center[2] + cyl_height / 2.0
        xy_dist = norm(point[:2] - cyl_center[:2])
        if point[2] < z_min:
            if xy_dist <= cyl_radius: return z_min - point[2]
            else: return sqrt((xy_dist - cyl_radius)**2 + (z_min - point[2])**2)
        elif point[2] > z_max:
            if xy_dist <= cyl_radius: return point[2] - z_max
            else: return sqrt((xy_dist - cyl_radius)**2 + (point[2] - z_max)**2)
        else:
            return xy_dist - cyl_radius

    def distance_segment_to_cylinder(self, p_start, p_end, cyl_center, cyl_radius, cyl_height, num_samples=5):
        min_dist = float('inf')
        best_t = 0.0
        for i in range(num_samples + 2):
            t = i / (num_samples + 1)
            p = p_start + t * (p_end - p_start)
            d = self.distance_point_to_cylinder(p, cyl_center, cyl_radius, cyl_height)
            if d < min_dist:
                min_dist = d
                best_t = t
        return min_dist, best_t

    def get_min_obstacle_distance(self, q):
        transforms = self.get_all_urdf_transforms(q)
        min_dist = float('inf')
        closest_link = -1
        closest_obs = -1
        for link_idx, capsules in COLLISION_CAPSULES.items():
            T = transforms[link_idx]
            R, t_vec = T[:3, :3], T[:3, 3]
            for capsule in capsules:
                p1_local, p2_local = capsule.get_endpoints_in_link_frame()
                p1_world, p2_world = R @ p1_local + t_vec, R @ p2_local + t_vec
                for obs_idx, obs in enumerate(self.obstacles):
                    seg_dist, _ = self.distance_segment_to_cylinder(
                        p1_world, p2_world, obs['center'], obs['radius'], obs['height']
                    )
                    dist = seg_dist - capsule.radius
                    if dist < min_dist:
                        min_dist, closest_link, closest_obs = dist, link_idx, obs_idx
        return min_dist, closest_link, closest_obs

    def get_closest_point_and_gradient(self, q):
        urdf_transforms = self.get_all_urdf_transforms(q)
        best_dist = float('inf')
        best_gradient, best_link_idx, best_p_closest, best_safety_dist = None, -1, None, 0.0
        for link_idx, capsules in COLLISION_CAPSULES.items():
            T_urdf = urdf_transforms[link_idx]
            R, t_vec = T_urdf[:3, :3], T_urdf[:3, 3]
            for capsule in capsules:
                p1_local, p2_local = capsule.get_endpoints_in_link_frame()
                p1_world, p2_world = R @ p1_local + t_vec, R @ p2_local + t_vec
                for obs in self.obstacles:
                    seg_dist, seg_t = self.distance_segment_to_cylinder(
                        p1_world, p2_world, obs['center'], obs['radius'], obs['height']
                    )
                    dist = seg_dist - capsule.radius
                    if dist < self.d_influence and dist > 0.001:
                        p_closest = p1_world + seg_t * (p2_world - p1_world)
                        xy_diff = p_closest[:2] - obs['center'][:2]
                        xy_dist = norm(xy_diff)
                        z_min, z_max = obs['center'][2] - obs['height']/2, obs['center'][2] + obs['height']/2
                        if z_min <= p_closest[2] <= z_max:
                            gradient = np.array([xy_diff[0]/xy_dist, xy_diff[1]/xy_dist, 0.0]) if xy_dist > 0.001 else np.array([1.0, 0, 0])
                        elif xy_dist <= obs['radius']:
                            gradient = np.array([0, 0, -1.0 if p_closest[2] < z_min else 1.0])
                        else:
                            gradient = np.array([gx, gy, gz]) # Using normalized vector away from obstacle
                            g_norm = norm(gradient)
                            if g_norm > 0.001: gradient /= g_norm
                        if dist < best_dist:
                            best_dist, best_gradient, best_link_idx, best_p_closest = dist, gradient, link_idx, p_closest
        return best_p_closest, best_gradient, best_dist, best_link_idx, best_safety_dist

    # =========================================================================
    # VISUALIZATION (PROVIDED)
    # =========================================================================
    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f'fer_joint{i+1}' for i in range(7)] + ['fer_finger_joint1', 'fer_finger_joint2']
        finger_pos = self.config.get('visualization', {}).get('gripper_opening', 0.04)
        msg.position = self.q.tolist() + [finger_pos, finger_pos]
        msg.velocity = self.q_dot.tolist() + [0.0, 0.0]
        self.joint_pub.publish(msg)

    def publish_markers(self):
        markers = MarkerArray()
        traj = Marker()
        traj.header.frame_id, traj.header.stamp = "fer_link0", self.get_clock().now().to_msg()
        traj.ns, traj.id, traj.type, traj.action = "trajectory", 0, Marker.LINE_STRIP, Marker.ADD
        traj.scale.x, traj.color = 0.01, ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        traj.points = [Point(x=self.PA[0], y=self.PA[1], z=self.PA[2]), Point(x=self.PB[0], y=self.PB[1], z=self.PB[2])]
        markers.markers.append(traj)
        
        transforms = self.get_all_urdf_transforms(self.q)
        cap_id = 0
        for link_idx, capsules in COLLISION_CAPSULES.items():
            T = transforms[link_idx]
            R, t_vec = T[:3, :3], T[:3, 3]
            for capsule in capsules:
                p1_l, p2_l = capsule.get_endpoints_in_link_frame()
                p1, p2 = R @ p1_l + t_vec, R @ p2_l + t_vec
                mid, seg = (p1 + p2) / 2.0, p2 - p1
                seg_len = norm(seg)
                m = Marker()
                m.header.frame_id, m.header.stamp = "fer_link0", self.get_clock().now().to_msg()
                m.ns, m.id, m.type, m.action = "capsules", cap_id, Marker.CYLINDER, Marker.ADD
                m.pose.position = Point(x=float(mid[0]), y=float(mid[1]), z=float(mid[2]))
                # Orientation logic...
                m.scale = Vector3(x=float(capsule.radius*2), y=float(capsule.radius*2), z=float(seg_len+capsule.radius*2))
                m.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.3)
                markers.markers.append(m)
                cap_id += 1
        self.marker_pub.publish(markers)

    def log_data(self):
        """Log data for analysis."""
        t = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        p = self.get_end_effector_position(self.q)
        
        # Υπολογισμός Μέτρου Χειρισιμότητας (Manipulability)
        J = self.compute_jacobian(self.q)
        # w = sqrt(det(J * J^T)) - δείχνει πόσο "ελεύθερο" είναι το ρομπότ να κινηθεί
        w = np.sqrt(max(0, np.linalg.det(J @ J.T)))
        min_dist, _, _ = self.get_min_obstacle_distance(self.q)
        
        self.log['time'].append(t)
        self.log['position'].append(p.copy())
        self.log['joint_velocities'].append(self.q_dot.copy()) # Καταγραφή ταχυτήτων
        self.log['manipulability'].append(w)                   # Καταγραφή χειρισιμότητας
        self.log['min_distance'].append(min_dist)

    def save_log(self, filename=None):
        """Save logged data to file."""
        if filename is None:
            filename = f"log_student_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        filepath = os.path.join(self.log_dir, filename)
        np.savez(filepath, **{k: np.array(v) for k, v in self.log.items()})
        self.get_logger().info(f'Log saved to {filepath}')

    # =========================================================================
    #                         STUDENT IMPLEMENTATION
    # =========================================================================

    def compute_jacobian(self, q):
        """
        Compute the geometric Jacobian matrix for end-effector position control.
        """
        Jacobian = np.zeros((3, 7))#δημιουργώ έναν κενό πίνακα 3x7 γιατι με ενδιαφερει μονο η γραμμικη ταχυτητα
        transforms = self.get_all_transforms(q)#καλώ τη συνάρτηση που υπολογίζει την ευθεία κινηματική μήτρα 
        p_E = transforms[-1][:3, 3]#πάω στον τελευταίο πίνακα και παίρνω τη θέση του end effector
        
        for i in range(7):
            z_i = transforms[i][:3, 2] #bi=zi-1 
            p_i = transforms[i][:3, 3] #p_prev=Oi
            Jacobian[:, i] = np.cross(z_i, (p_E - p_i))#υπολογίζω το εξωτερικό γινόμενο δηλαδή την Ji και την αποθηκεύω στον J
            
        return Jacobian

    def compute_primary_task_desired_velocity(self):
        """
        Compute desired end-effector velocity for trajectory tracking.
        """
        t = (self.get_clock().now() - self.start_time).nanoseconds / 1e9#βρισκω τον χρονο απο την αρχη της προσομοιωσης
        tf = 8.0 #ημιπεριοδος
        cycle = t % (2 * tf)#ο μετρητης μηδενιζεται καθε 16 δευτερολεπτα
        
        if cycle <= tf:#αν ο μετρητης ειναι μικροτερος της μισης περιοδου 
            tau = cycle / tf
            p_start, p_end = self.PA, self.PB#παω απο το PA στο PB 
        else:
            tau = (cycle - tf) / tf
            p_start, p_end = self.PB, self.PA#αλλιως παω απο το Β στο Α

        s = 10 * (tau**3) - 15 * (tau**4) + 6 * (tau**5)#η εξισωση 5ου βαθμου εφαρμοζοντας τις οριακες συνθηκες του θεωρητικου μερους
        ds = (1.0 / tf) * (30 * (tau**2) - 60 * (tau**3) + 30 * (tau**4))

        yd = p_start + s * (p_end - p_start)
        v_traj = ds * (p_end - p_start)

        current_p = self.get_end_effector_position(self.q)#η πραγματικη θεση
        x_dot_desired = v_traj + self.Kp * (yd - current_p)#η επιθυμητη θεση

        return x_dot_desired

    def compute_secondary_task_desired_velocity(self, q):
        """
        Compute joint velocity for obstacle avoidance (secondary task).
        """
        q_dot_avoid = np.zeros(7)
        p_closest, gradient, dist, link_idx, _ = self.get_closest_point_and_gradient(q)
        
        if p_closest is not None and dist < self.d_influence:
            #Αν η αποσταση ειναι μικροτερη απο την αποσταση ασφαλειας
            v_repulsive = self.Ko * (self.d_influence - dist) * gradient
            #υπολογιζουμε την αντισταση αποφυγης με βαση τη θεωρια
            Jacobian_point = np.zeros((3, 7))
            transforms = self.get_all_transforms(q)
            for i in range(min(link_idx + 1, 7)):
                Jacobian_point[:, i] = np.cross(transforms[i][:3, 2], (p_closest - transforms[i][:3, 3]))
            
            q_dot_avoid = 50.0 * (Jacobian_point.T @ v_repulsive)
            
        q_nominal = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785])
        q_dot_center = 1.2 * (q_nominal - q)

        return q_dot_avoid + q_dot_center

    def control_loop(self):
        """
        Main control loop - executes at fixed rate (e.g., 100 Hz).
        Implements task-priority control.
        """
        if self.bypass:
            self.publish_joint_states()
            self.publish_markers()
            self.log_data()
            return

        J = self.compute_jacobian(self.q)
        x_dot = self.compute_primary_task_desired_velocity()
        q_dot_avoid = self.compute_secondary_task_desired_velocity(self.q)

        # Task-priority control: q_dot = J# * x_dot + (I - J#J) * q_dot_null
        J_pinv = pinv(J)
        self.q_dot = (J_pinv @ x_dot) + ((np.eye(7) - J_pinv @ J) @ q_dot_avoid)

        # Velocity limiting
        if np.max(np.abs(self.q_dot)) > self.max_joint_vel:
            self.q_dot *= (self.max_joint_vel / np.max(np.abs(self.q_dot)))

        self.q = np.clip(self.q + self.q_dot * self.dt, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

        self.publish_joint_states()
        self.publish_markers()
        self.log_data()

def main(args=None):
    rclpy.init(args=args)
    controller = RedundantController()
    try: rclpy.spin(controller)
    except KeyboardInterrupt: pass
    finally:
        controller.save_log()
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()