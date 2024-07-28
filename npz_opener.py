import numpy as np
np.set_printoptions(precision=6, suppress=True)


# Load from npz file
data = np.load("robot_poses.npz")
robot_pose_read_from_robot = data['robot_pose_read_from_robot']

print(robot_pose_read_from_robot)