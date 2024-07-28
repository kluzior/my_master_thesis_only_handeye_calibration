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
import datetime
from pathlib import Path
import os


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



            robot_pose_read_from_robot = []
            folder_with_time = "images_" + datetime.datetime.now().strftime("%d-%m_%H-%M")
            waiting_pos_folder = folder_with_time + "/waiting_pos"
            directory_with_time = Path("images/for_hand_eye_calib/"+folder_with_time)
            directory_with_time.mkdir(parents=True, exist_ok=True)

            directory_waiting = Path("images/for_hand_eye_calib/"+waiting_pos_folder)
            directory_waiting.mkdir(parents=True, exist_ok=True)
            i = 0

            frame_event.set()  # Signal camera thread to capture frame
            
            # Wait for the frame to be stored
            while frame_event.is_set():
                time.sleep(0.1)  # Wait a bit before checking again

            # Now the frame should be available in frame_storage['frame']
            if 'frame' in frame_storage:
                _img = frame_storage['frame']
                img_name = f"{directory_waiting}/WAITING_POSE.jpg"
                cv2.imwrite(img_name, _img)

                robot_waiting_pose = robot_functions.give_pose()
                _, _, pose_waiting = handeye.calculate_rvec_tvec_from_robot_pose(robot_waiting_pose)

            for pose in robot_poses.poses_2:
                robot_functions.moveJ(pose)

                frame_event.set()  # Signal camera thread to capture frame
                
                # Wait for the frame to be stored
                while frame_event.is_set():
                    time.sleep(0.1)  # Wait a bit before checking again

                # Now the frame should be available in frame_storage['frame']
                if 'frame' in frame_storage:
                    i+=1
                    test_img = frame_storage['frame']
                        
                    gray = test_img.copy()
                    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(gray, (8,7), None)
                    if not ret:
                        print("****** PHOTO SKIPPED ******")
                        continue
                    cv2.imshow("Requested Frame", test_img)

                    img_name = f"{directory_with_time}/img_{i}.jpg"
                    cv2.imwrite(img_name, test_img)
                    print(f"Saved {img_name}")
                    cv2.waitKey(500)  # Wait for any key press
                    cv2.destroyWindow("Requested Frame")
                else:
                    print("Frame was not captured")
                robot_pose = robot_functions.give_pose()
                _, _, pose = handeye.calculate_rvec_tvec_from_robot_pose(robot_pose)
                robot_pose_read_from_robot.append(pose)
                print("ROBOT POSE APPENDED!")

                

                time.sleep(1)

            print("DONE")
            print(f"robot_poses: {robot_pose_read_from_robot}")
            print(f"i: {i}")
            print(f"lenPose: {len(robot_pose_read_from_robot)}")
            # Save to npz file
            np.savez(f"{directory_with_time}/robot_poses.npz", robot_pose_read_from_robot=robot_pose_read_from_robot, pose_waiting=pose_waiting)

            # Save to txt file
            with open(os.path.join(directory_with_time, 'robot_poses.txt'), 'w') as file:
                for pose in robot_pose_read_from_robot:
                    file.write(','.join(map(str, pose)) + '\n')
                file.write('\n' + '\n' + '\n')
                file.write(','.join(map(str, pose_waiting)) + '\n')

            with open(os.path.join(waiting_pos_folder, 'wait_pose.txt'), 'w') as file2:
                file2.write(','.join(map(str, pose_waiting)) + '\n')

    except socket.error as socketerror:
        print("Socket error: ", socketerror)
    finally:
        c.close()
        s.close()
        print("Socket closed")
        cv2.destroyAllWindows()
