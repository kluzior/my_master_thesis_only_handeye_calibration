import time
from cmd_generator import CmdGenerator  # Ensure this module is available
from robot_positions import RobotPositions  # Ensure this module is available
from robot_functions import RobotFunctions  # Ensure this module is available
from packages2.image_processor import ImageProcessor  # Ensure this module is available
from packages2.handeye_eye_in_hand import HandEyeCalibration  # Ensure this module is available
import socket
import cv2
import threading
import numpy as np
import pickle
import packages2.common as common

def calibrate_tsai(A_list, B_list):
    # A_list = self.camera_mtx.copy()
    # B_list = self.robot_mtx.copy()

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

def show_camera(frame_event, frame_storage):
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("All Frames", frame)  # Display each frame
        if frame_event.is_set():
            frame_storage['frame'] = frame  # Store frame in dictionary on request
            frame_event.clear()  # Reset event after storing frame
        if cv2.waitKey(1) == ord('q'):  # Allow exiting the loop by pressing 'q'
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    frame_event = threading.Event()
    frame_storage = {}
    robot_poses = RobotPositions()
    image_processor = ImageProcessor()
    # Create a new thread for the camera function
    camera_thread = threading.Thread(target=show_camera, args=(frame_event, frame_storage))
    camera_thread.start()

    HOST = "192.168.0.1"
    PORT = 10000
    print("Start listening...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(5)
    c, addr = s.accept()
    robot_functions = RobotFunctions(c)
    handeye = HandEyeCalibration()

    try:
        msg = c.recv(1024)
        print(msg)
        if msg == b"Hello server, robot here":
            print("Robot requested for data!")
            time.sleep(0.5)

            robot_functions.moveJ(RobotPositions.look_at_chessboard)

            cam_rvecs = []
            cam_tvecs = []
            rob_rvecs = []
            rob_tvecs = []
            robot_pose_read_from_robot = []
            for pose in robot_poses.poses_2:
                robot_functions.moveJ(pose)

                frame_event.set()  # Signal camera thread to capture frame

                # Wait for the frame to be stored
                while frame_event.is_set():
                    time.sleep(0.1)  # Wait a bit before checking again

                # Now the frame should be available in frame_storage['frame']
                if 'frame' in frame_storage:
                    test_img = frame_storage['frame']

                    cam_rvec, cam_tvec = handeye.get_camera_rvec_tvec(test_img)
                    print(f"Calculated camera rvec: {cam_rvec}")
                    print(f"Calculated camera tvec: {cam_tvec}")
                    cam_rvecs.append(cam_rvec)
                    cam_tvecs.append(cam_tvec)
                    cv2.imshow("Requested Frame", test_img)
                    # Save test_img to file as img_1...n
                    img_name = f"img_{len(cam_rvecs) }.jpg"
                    cv2.imwrite(img_name, test_img)
                    print(f"Saved {img_name}")
                    cv2.waitKey(500)  # Wait for any key press
                    cv2.destroyWindow("Requested Frame")
                else:
                    print("Frame was not captured")
                robot_pose = robot_functions.give_pose()
                rob_rvec, rob_tvec, _ = handeye.calculate_rvec_tvec_from_robot_pose(robot_pose)
                print(f"TEST list(robot_pose): {list(robot_pose)}")
                robot_pose_read_from_robot.append(list(robot_pose))
                print(f"Calculated robot rvec: {rob_rvec}")
                print(f"Calculated robot tvec: {rob_tvec}")
                rob_rvecs.append(rob_rvec)
                rob_tvecs.append(rob_tvec)

                time.sleep(1)
            print(f"TESTTESTTEST   Robot poses: {robot_pose_read_from_robot}")
            np.savez("robot_poses.npz", robot_pose_read_from_robot=robot_pose_read_from_robot)
            robot_functions.moveJ(RobotPositions.look_at_chessboard)
            # Assuming robot_pose_read_from_robot is defined and populated with your data
            with open('robot_poses.pkl', 'wb') as f:
                pickle.dump(robot_pose_read_from_robot, f)
            # CALCULATE OPENCV HAND EYE CALIB
            cam_rvecs_np = np.array(cam_rvecs)
            cam_tvecs_np = np.array(cam_tvecs)
            rob_rvecs_np = np.array(rob_rvecs)
            rob_tvecs_np = np.array(rob_tvecs)

            # transformation matrix 
            R, T = cv2.calibrateHandEye(rob_rvecs_np, rob_tvecs_np, cam_rvecs_np, cam_tvecs_np)    
            print("RESULTS OF CALIBRATION")
            print(f"R: {R}")
            print(f"T: {T}")



            time.sleep(1)
            frame_event.set()  # Signal camera thread to capture frame
            while frame_event.is_set():
                time.sleep(0.1)  # Wait a bit before checking again
                
            if 'frame' in frame_storage:
                immmage = frame_storage['frame']
                test_rvec, test_tvec = image_processor.calculate_rvec_tvec(immmage)
                test_rvec2, test_tvec2 = image_processor.calculate_rvec_tvec(immmage,  point_shift=(1,1))
            np.set_printoptions(precision=6, suppress=True)


            test_rmtx = cv2.Rodrigues(test_rvec)[0]
            test_rmtx2 = cv2.Rodrigues(test_rvec2)[0]
            print(f"test rmtx (in camera coords): {test_rmtx}")
            print(f"test tvec (in camera coords): {test_tvec}")   
            print(f"test rmtx2 (in camera coords): {test_rmtx2}")
            print(f"test tvec2 (in camera coords): {test_tvec2}")

            T_point_to_cam = handeye.prepare_one_mtx(test_rmtx, test_tvec)
            # T_point_to_cam = np.array(T_point_to_cam)
            print(f"T_point_to_cam: \n{T_point_to_cam}")
            T_cam_to_tcp = handeye.prepare_one_mtx(R, T)
            # T_base_to_cam = np.array(T_base_to_cam)       
            print(f"T_cam_to_tcp: \n{T_cam_to_tcp}")

            T_point_to_tcp = np.dot(T_point_to_cam, T_cam_to_tcp)
            print("WYNIKOWA MACIERZ TRANSLACJI (4x4) Z PUNKTU DO TCP ROBOTA (OpenCV):")
            print(f"{T_point_to_tcp}")


            T_point_to_cam2 = handeye.prepare_one_mtx(test_rmtx2, test_tvec2)
            T_point_to_cam2 = np.array(T_point_to_cam2)

            T_cam_to_tcp2 = handeye.prepare_one_mtx(R, T)
            T_cam_to_tcp2 = np.array(T_cam_to_tcp2)

            T_point_to_tcp2 = np.dot(T_point_to_cam2, T_cam_to_tcp2)
            print("WYNIKOWA MACIERZ TRANSLACJI (4x4) Z PUNKTU 2 DO TCP ROBOTA (OpenCV):")
            print(f"{T_point_to_tcp2}")

            robot_pose_test = robot_functions.give_pose()
            rob_rvec_test, rob_tvec_test, _ = handeye.calculate_rvec_tvec_from_robot_pose(robot_pose_test)    
            T_tcp_to_base_test = handeye.prepare_one_mtx(rob_rvec_test, rob_tvec_test)

            T_point_to_base = np.dot(T_point_to_tcp, T_tcp_to_base_test)
            T_point_to_base_x = T_point_to_tcp @ T_tcp_to_base_test

            print("WYNIKOWA MACIERZ TRANSLACJI (4x4) Z PUNKTU DO BAZY ROBOTA (OpenCV):")
            print(f"{T_point_to_base}")
            print(f"\n{T_point_to_base_x}")

            rob_mtxs = []
            cam_mtxs = [] 
            rob_rmtxs = []
            cam_rmtxs = []
            for rob_rvec in rob_rvecs:
                rob_rmtxs.append(cv2.Rodrigues(rob_rvec)[0])
            print(f"len rob_rmtxs: {len(rob_rmtxs)}")

            for cam_rvec in cam_rvecs:
                cam_rmtxs.append(cv2.Rodrigues(cam_rvec)[0])
            print(f"len cam_rmtxs: {len(cam_rmtxs)}")


            for rr, tt in cam_rmtxs, cam_tvecs:
                print(f"rr: {rr}")
                print(f"tt: {tt}")
                cam_mtxs.append(handeye.prepare_one_mtx(rr, tt))
            for rrr, ttt in rob_rmtxs, rob_tvecs:
                rob_mtxs.append(handeye.prepare_one_mtx(rrr, ttt))


            R_tsai, T_tsai = calibrate_tsai(rob_mtxs, cam_mtxs)
            
            print("RESULTS OF CALIBRATION (TSAI)")
            print(f"R_tsai: {R_tsai}")
            print(f"T_tsai: {T_tsai}")

    except socket.error as socketerror:
        print("Socket error: ", socketerror)
    finally:
        c.close()
        s.close()
        print("Socket closed")
        cv2.destroyAllWindows()
