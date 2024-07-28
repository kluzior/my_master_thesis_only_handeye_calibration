import numpy as np
import os
import cv2
from packages2.image_processor import ImageProcessor
from scipy.spatial.transform import Rotation as R
import packages2.common  as common

def prepare_one_mtx(rvec, tvec):
    mtx = np.zeros((4, 4))
    
    rmtx = cv2.Rodrigues(rvec)[0]
    mtx[:3, :3] = rmtx
    mtx[:3, 3] = tvec.ravel()
    mtx[3, 3] = 1
    return mtx 

def calculate_rvec_tvec_from_robot_pose(pos):
    print(f"pos: {pos}")

    roll = pos[3]
    pitch = pos[4]
    yaw = pos[5]

    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    rotation_matrix = r.as_matrix()


    rvec = cv2.Rodrigues(np.array(rotation_matrix))[0]
    tvec = np.zeros((3, 1))
    for i in range(3):
        tvec[i] = pos[i]

    return rvec, tvec

data = np.load(f"R_T_results.npz")
camera_tcp_mtx = data['camera_tcp_mtx']
image_procesor = ImageProcessor()
mtx, dist = image_procesor.load_camera_params("CameraParams_inna_kamera_17-07_16-34.npz")

print(f"\ncamera_tcp_mtx: {camera_tcp_mtx}")


wait_pose = [-0.191371,0.109018,0.67845,-2.86535,-1.27539,-0.0543387]

image_path = "WAITING_POSE.jpg"


image = cv2.imread(image_path)

cv2.imshow("test", image)


uimg = image_procesor.undistort_frame(image, mtx, dist)




cv2.imshow("test2", uimg)


cam_rvec, cam_tvec = image_procesor.calculate_rvec_tvec(uimg)
obj_to_cam_mtx = prepare_one_mtx(cam_rvec, cam_tvec)

print(f"\n obj_to_cam_mtx: {obj_to_cam_mtx}")



obj_to_tcp_mtx = obj_to_cam_mtx @ camera_tcp_mtx

print(f"\nobj_to_tcp_mtx: {obj_to_tcp_mtx}\n")

rob_rvec, rob_tvec = calculate_rvec_tvec_from_robot_pose(wait_pose)
rob_mtx = prepare_one_mtx(rob_rvec, rob_tvec)

print(f"\n rob_mtx: {rob_mtx}")


obj_to_base_mtx = obj_to_tcp_mtx @ rob_mtx
print(f"\nobj_to_base_mtx: {obj_to_base_mtx}")

cv2.waitKey(0)
cv2.destroyAllWindows()