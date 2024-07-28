import numpy as np
import os
import cv2
from packages2.image_processor import ImageProcessor
from scipy.spatial.transform import Rotation as R
import packages2.common  as common



def get_images( path=''):
    images = []
    # Sprawdź, czy ścieżka jest poprawna
    if not os.path.exists(path):
        print(f"The provided path does not exist: {path}")
        return

    # Przeglądaj pliki w folderze
    for filename in os.listdir(path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Failed to load image: {img_path}")
    print(f"Loaded {len(images)} images from {path}")
    return images    




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



def prepare_one_mtx(rvec, tvec):
    mtx = np.zeros((4, 4))
    
    rmtx = cv2.Rodrigues(rvec)[0]
    mtx[:3, :3] = rmtx
    mtx[:3, 3] = tvec.ravel()
    mtx[3, 3] = 1
    return mtx    

def prepare_one_mtx2(rmtx, tvec):
    mtx = np.zeros((4, 4))
    
    mtx[:3, :3] = rmtx
    mtx[:3, 3] = tvec.ravel()
    mtx[3, 3] = 1
    return mtx   

def calibrate_tsai(A_list, B_list):

    N = len(A_list)
    S = None
    RA_I = None
    T = None
    TA = None
    TB = None

    for i in range(N):
        An = A_list[i]
        Bn = B_list[i]
        RA = An[:3,:3]
        tA = An[:3,3].reshape(3,1)
        tA_ = common.skew2(tA)
        RB = Bn[:3,:3]
        tB = Bn[:3,3].reshape(3,1)
        
        uA, wA = common.R_2_angle_axis(RA)
        uB, wB = common.R_2_angle_axis(RB)
        _S = common.skew2(uA + uB)
        _T = uB - uA 
        _RA_I = RA - np.eye(3)
        _TA = tA
        _TB = tB
        
        S = np.append(S, _S, axis=0) if S is not None else _S
        T = np.append(T, _T, axis=0) if T is not None else _T
        
        RA_I = np.append(RA_I,_RA_I,axis=0) if RA_I is not None else _RA_I
        TA = np.append(TA,_TA,axis=0) if TA is not None else _TA
        TB = np.append(TB,_TB,axis=0) if TB is not None else _TB

    ux = common.solveLS(S,T)    
    uX = 2*ux/(np.sqrt(1+common.norm(ux)**2))
    Rx = (1-common.norm(uX)**2/2)*np.eye(3) + 0.5*(uX*uX.T + np.sqrt(4-common.norm(uX)**2)*common.skew2(uX))
    tX = common.get_Translation(Rx,RA_I,TA,TB)

    tsai_result = common.to_mtx(Rx, tX)
    
    return tsai_result


def calibrate_li(A_list, B_list):
    
    N = len(A_list)
    I = np.eye(3)
    I9 = np.eye(9)
    S = None
    T = None

    for i in range(N):
        An = A_list[i]
        Bn = B_list[i]
        RA = An[:3,:3]
        tA = An[:3,3].reshape(3,1)
        tA_ = common.skew2(tA)
        RB = Bn[:3,:3]
        tB = Bn[:3,3].reshape(3,1)
        S1 = np.append(I9 - np.kron(RB, RA), np.zeros((9,3)), axis=1)
        S2 = np.append(np.kron(tB.T, tA_), np.dot(tA_, (I - RA)), axis=1)

        _S = np.append(S1, S2, axis=0)
        _T = np.append(np.zeros((9,1)), tA.reshape(3,1), axis=0)
        
        S = np.append(S, _S, axis=0) if S is not None else _S
        T = np.append(T, _T, axis=0) if T is not None else _T

    Rx_tX = common.solveSVD(S)
    Rx, tX = common.get_RX_tX(Rx_tX)
    Rx = common.getRotation(Rx)

    li_result = common.to_mtx(Rx, tX)

    return li_result


np.set_printoptions(precision=6, suppress=True)

#DATA TAKEN FROM HERE
files_path = "images/for_hand_eye_calib/images_18-07_15-44"
# files_path = "images/for_hand_eye_calib/images_18-07_16-36"



# Load from npz file
data = np.load(f"{files_path}/robot_poses.npz")
robot_pose_read_from_robot = data['robot_pose_read_from_robot']

image_processor = ImageProcessor()
mtx, dist = image_processor.load_camera_params("CameraParams_inna_kamera_17-07_16-34.npz")
images = get_images(files_path)


print(len(robot_pose_read_from_robot))
cam_rvecs = []
cam_tvecs = []
cam_mtxs = []
rob_rvecs = []
rob_tvecs = []
rob_mtxs = []
for robot_pose, image in zip(robot_pose_read_from_robot, images):
    image_to_show = image.copy()
    print(f"robot pose: {robot_pose}")
    cv2.imshow("Image", image)

    u_image = image_processor.undistort_frame(image, mtx, dist)
    
    cv2.imshow("uImage", u_image)


    cam_rvec, cam_tvec = image_processor.calculate_rvec_tvec(u_image)
    cam_mtx = prepare_one_mtx(cam_rvec, cam_tvec)
    cam_rvecs.append(cam_rvec)
    cam_tvecs.append(cam_tvec)
    cam_mtxs.append(cam_mtx)

    print(f"Calculated object to camera rvec: {cam_rvec}")
    print(f"Calculated object to camera tvec: {cam_tvec}")
    print(f"Calculated object to camera mtx: {cam_mtx}")

    rob_rvec, rob_tvec = calculate_rvec_tvec_from_robot_pose(robot_pose)
    rob_mtx = prepare_one_mtx(rob_rvec, rob_tvec)
    rob_rvecs.append(rob_rvec)
    rob_tvecs.append(rob_tvec)
    rob_mtxs.append(rob_mtx)

    print(f"Calculated robot to base rvec: {rob_rvec}")
    print(f"Calculated robot to base tvec: {rob_tvec}")
    print(f"Calculated robot to base mtx: {rob_mtx}")

    gray = cv2.cvtColor(image_to_show, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (8,7), None)
    cv2.drawChessboardCorners(image_to_show, (8, 7), corners, True)

    cv2.imshow("Image with corners", image_to_show)

    cv2.waitKey(500)
    cv2.destroyAllWindows()


print("DONE")
print(f"len of: {len(cam_rvecs)}")

# proceed with OpenCv calibration
R1, T1 = cv2.calibrateHandEye(rob_rvecs, rob_tvecs, cam_rvecs, cam_tvecs, method=cv2.CALIB_HAND_EYE_TSAI)   
print("RESULTS OF CALIBRATION (cv2.CALIB_HAND_EYE_TSAI)")
print(f"R: {R1}")
print(f"T: {T1}")

R2, T2 = cv2.calibrateHandEye(rob_rvecs, rob_tvecs, cam_rvecs, cam_tvecs, method=cv2.CALIB_HAND_EYE_PARK )   
print("RESULTS OF CALIBRATION (cv2.CALIB_HAND_EYE_PARK )")
print(f"R: {R2}")
print(f"T: {T2}")

R3, T3 = cv2.calibrateHandEye(rob_rvecs, rob_tvecs, cam_rvecs, cam_tvecs, method=cv2.CALIB_HAND_EYE_HORAUD)
print("RESULTS OF CALIBRATION (cv2.CALIB_HAND_EYE_HORAUD)")
print(f"R: {R3}")
print(f"T: {T3}")

# R4, T4 = cv2.calibrateHandEye(rob_rvecs, rob_tvecs, cam_rvecs, cam_tvecs, method=cv2.CALIB_HAND_EYE_ANDREFF)
# print("RESULTS OF CALIBRATION (cv2.CALIB_HAND_EYE_ANDREFF)")
# print(f"R: {R4}")
# print(f"T: {T4}")

R5, T5 = cv2.calibrateHandEye(rob_rvecs, rob_tvecs, cam_rvecs, cam_tvecs, method=cv2.CALIB_HAND_EYE_DANIILIDIS)
print("RESULTS OF CALIBRATION (cv2.CALIB_HAND_EYE_DANIILIDIS)")
print(f"R: {R5}")
print(f"T: {T5}")



tsai_result = calibrate_tsai(cam_mtxs, rob_mtxs)
print("RESULTS OF CALIBRATION (TSAI)")
print(f"R: {tsai_result[:3,:3]}")
print(f"T: {tsai_result[:3,3]}")

li_result = calibrate_li(cam_mtxs, rob_mtxs)
print("RESULTS OF CALIBRATION (LI)")
print(f"R: {li_result[:3,:3]}")
print(f"T: {li_result[:3,3]}")


result_mtx = prepare_one_mtx2(R5, T5)

# Save to npz file
np.savez(f"R_T_results.npz", camera_tcp_mtx = result_mtx)