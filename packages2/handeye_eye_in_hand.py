from packages2.robot import Robot
from packages2.camera import Camera
import cv2 
import numpy as np
import packages2.common as common
import logging
from packages2.image_processor import ImageProcessor
import pickle
from robot_positions import RobotPositions
import time
from robot_functions import RobotFunctions
from scipy.spatial.transform import Rotation as R
import math


class HandEyeCalibration:
    def __init__(self):
        self.image_processor = ImageProcessor()
        camera_calib_result_path = 'CameraParams_inna_kamera_17-07_16-34.npz'
        mtx, distortion = self.image_processor.load_camera_params(camera_calib_result_path)


        self.camera = Camera(mtx, distortion, 1)
        # self.robot = Robot(robot_client)
        self.robot_poses = RobotPositions()
        # self.robot_functions = RobotFunctions(robot_client)

        pass 


    def run(self):

        chess_image = None
        print(f'IF YOU ARE READY PRESS "i" TO SAVE THE CHECKERBOARD IMAGE FOR CALIBRATION')
        if self.camera.cap.isOpened():
            img = self.camera.get_frame_old()
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                print(f'User pressed Esc key - exit program')
            cv2.imshow('Camera', img)
        self.camera.cap.release()
        cv2.destroyAllWindows()


        for pose in self.robot_poses.poses:
            print(f"pose: {pose}")
            self.robot.moveJ(pose)
            time.sleep(1)
            chess_image = img
            mtx, distortion = self.image_processor.load_camera_params('CameraParams_16-07_18-05.npz')
            u_chess_image = self.image_processor.undistort_frame(chess_image, mtx, distortion)
            cam_rvec, cam_tvec = self.image_processor.calculate_rvec_tvec(u_chess_image)
            print(f"cam_rvec: {cam_rvec}")
            print(f"cam_tvec: {cam_tvec}")
            robot_position = self.robot.give_pose()
            rob_rvec, rob_tvec = self.calculate_rvec_tvec_from_robot_pose(robot_position)
            print(f"rob_rvec: {rob_rvec}")
            print(f"rob_tvec: {rob_tvec}")
            time.sleep(1)



    def get_camera_rvec_tvec(self, chess_image):
    
        mtx, distortion = self.image_processor.load_camera_params('CameraParams_inna_kamera_17-07_16-34.npz')
        u_chess_image = self.image_processor.undistort_frame(chess_image, mtx, distortion)
        cam_rvec, cam_tvec = self.image_processor.calculate_rvec_tvec(u_chess_image)

        return cam_rvec, cam_tvec



    # def calculate_rvec_tvec_from_robot_pose_with_save(self, pose):
    #     print(f"pose: {pose}")
    #     pos = list(pose)
    #     print(f"pos: {pos}")

    #     roll = pos[3]
    #     pitch = pos[4]
    #     yaw = pos[5]
    #     # rotation_matrix = common.RPY_to_rmtx(roll, pitch, yaw)

    #     r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    #     rotation_matrix = r.as_matrix()

    #     # R_matrix[:3, :3] = rotation_matrix
    #     # print(f"rotation_matrix: {rotation_matrix}")
    #     rvec = cv2.Rodrigues(np.array(rotation_matrix))[0]
    #     tvec = np.zeros((3, 1))
    #     for i in range(3):
    #         tvec[i] = pos[i]
    #     # print(f"Calculated (rvec, tvec) for pose_{index}: {(rvec, tvec)}")

    #     return rvec, tvec, pos




    def calculate_rvec_tvec_from_robot_pose(self, pose):
        print(f"pose: {pose}")
        pos = list(pose)
        print(f"pos: {pos}")

        roll = pos[3]
        pitch = pos[4]
        yaw = pos[5]
        # rotation_matrix = common.RPY_to_rmtx(roll, pitch, yaw)

        r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
        rotation_matrix = r.as_matrix()

        # R_matrix[:3, :3] = rotation_matrix
        # print(f"rotation_matrix: {rotation_matrix}")
        rvec = cv2.Rodrigues(np.array(rotation_matrix))[0]
        tvec = np.zeros((3, 1))
        for i in range(3):
            tvec[i] = pos[i]
        # print(f"Calculated (rvec, tvec) for pose_{index}: {(rvec, tvec)}")

        return rvec, tvec, pos



    def prepare_one_mtx(self, rvec, tvec):
        mtx = np.zeros((4, 4))
        
        rmtx = cv2.Rodrigues(rvec)[0]
        mtx[:3, :3] = rmtx
        mtx[:3, 3] = tvec.ravel()
        mtx[3, 3] = 1
        return mtx    







    def calibrate_tsai(self, A_list, B_list):
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


    def calibrate_li(self, A_list, B_list):
        # A_list = self.camera_mtx.copy()
        # B_list = self.robot_mtx.copy()
        
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



    # def __init__(self, robot_client):
    #     """
    #     Initializes the HandEyeCalibration object.

    #     Args:
    #         robot_client: The client object for communicating with robot.

    #     Attributes:
    #         camera: Camera object for capturing images
    #         robot: Robot object for controlling the robot
    #         image_processor: ImageProcessor object for framework to process images

    #         rvec_tvec_pairs: A list to store the rotation and translation vectors for objects seen by camera
    #         robot_positions: A list to store the robot's TCP positions

    #         camera_mtx: A list to store calculated object-camera matrices 
    #         robot_mtx: A list to store calculated TCP-robot base matrices 

    #         tsai_result: The result of Tsai's hand-eye calibration method
    #         li_result: The result of Li's hand-eye calibration method

    #         coordinates_of_chessboard: The defined chess corners for calibration which will be used to collect robot positions

    #         _logger: logger object for logging  messages
    #     """
    #     my_image_framework = ImageProcessor()
    #     camera_calib_result_path = 'CameraParams_16-07_18-05.npz'
    #     mtx, distortion = my_image_framework.load_camera_params(camera_calib_result_path)


    #     self.camera = Camera(mtx, distortion, 1)
    #     self.robot = Robot(robot_client)
    #     self.image_processor = ImageProcessor()

    #     self.robot_rvec_tvec_pairs = []
    #     self.rvec_tvec_pairs = []
    #     self.robot_positions = []

    #     self.camera_mtx = []  
    #     self.robot_mtx = []    

    #     self.tsai_result = None  
    #     self.li_result = None

    #     self.coordinates_of_chessboard=[(0,0),
    #                                     (0,1),
    #                                     (0,2),
    #                                     (0,3),
    #                                     (0,4),
    #                                     (0,5),
    #                                     (0,6),
    #                                     (1,0),
    #                                     (2,0),
    #                                     (3,0),
    #                                     (4,0),
    #                                     (5,0),
    #                                     (6,0),
    #                                     (7,0)   
    #                                     ]               # defined chess corners to calibration

    #     self._logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
    #     self._logger.debug(f'HandEyeCalibration({self}) was initialized.')

    # def run(self):
    #     """
    #     Main function to run hand-eye calibration process.

    #     This method captures images of a chessboard pattern using a camera and records the corresponding robot TCP positions.
    #     Then it performs calibration calculations to determine 4x4 transformation matrix between camera and robot base.

    #     Returns:
    #         None
    #     """
    #     chess_image = None
    #     self._logger.info(f'IF YOU ARE READY PRESS "i" TO SAVE THE CHECKERBOARD IMAGE FOR CALIBRATION')
    #     while self.camera.cap.isOpened():
    #         img = self.camera.get_frame_old()
    #         k = cv2.waitKey(5) & 0xFF
    #         if k == 27:
    #             self._logger.info(f'User pressed Esc key - exit program')
    #             break
    #         elif k == ord('i'):  # wait for 'i' key to capture chessboard image
    #             chess_image = img
    #             self._logger.info(f"Photo of chessboard for calibration saved")
    #             mtx, distortion = self.image_processor.load_camera_params('CameraParams_16-07_18-05.npz')
    #             u_chess_image = self.image_processor.undistort_frame(chess_image, mtx, distortion)
    #             for index, coords in enumerate(self.coordinates_of_chessboard, start=1):
    #                 rvec, tvec = self.image_processor.calculate_rvec_tvec(u_chess_image, point_shift=coords)
    #                 self.image_processor.show_chess_corner(u_chess_image, point=coords)
    #                 self.rvec_tvec_pairs.append((rvec, tvec))
    #                 self._logger.info(f'PRESS "s" TO CONFIRM THAT ROBOT IS ON PLACE FOR POINT {index}: {coords}')
    #                 while True:
    #                     k2 = cv2.waitKey(5)
    #                     if k2 == ord('s'):  # wait for 's' key to capture robot TCP position
    #                         robot_position = self.robot.give_pose()
    #                         self.robot_positions.append(robot_position)
    #                         self._logger.info(f'Received & saved pose for point {index}/{len(self.coordinates_of_chessboard)}')
    #                         break
    #             cv2.destroyWindow("POINT GUIDANCE")
    #             self._logger.info("RECEIVED ALL POSES - PRESS ESC")
    #         cv2.imshow('Camera', img)
    #     self.camera.cap.release()
    #     cv2.destroyAllWindows()

    #     for index, coords in enumerate(self.coordinates_of_chessboard, start=1):
    #         self._logger.debug(f'aggregated data for point no. {index} ({coords}):')
    #         self._logger.debug(f'\trobot position: {self.robot_positions[index-1]}')
    #         self._logger.debug(f'\tcamera rvec: {self.rvec_tvec_pairs[index-1][0]}')
    #         self._logger.debug(f'\tcamera tvec: {self.rvec_tvec_pairs[index-1][1]}')


    # def prepare_robot_cameta_mtx(self):

    #     self.camera_mtx = common.prepare_mtx_from_rvec_tvec(self.rvec_tvec_pairs)
    #     self.robot_mtx = common.prepare_mtx_from_rvec_tvec(self.robot_rvec_tvec_pairs)



    # def prepare_object_camera_mtx(self):
    #     """
    #     Prepare the rotation and translation matrix for object seen by camera.

    #     This method combine the rotation and translation 4x4 matrix for the object based on the provided rotation and
    #     translation vectors. It stores result matrix in the class `camera_mtx` list.

    #     Returns:
    #         None
    #     """
    #     for index, (rvec, tvec) in enumerate(self.rvec_tvec_pairs):
    #         self._logger.debug(f'Calculating rotation&translation matrix for camera no. {index}')
    #         mtx = np.zeros((4, 4))
            
    #         rmtx = cv2.Rodrigues(rvec)[0]
    #         mtx[:3, :3] = rmtx
    #         mtx[:3, 3] = tvec.ravel()
    #         mtx[3, 3] = 1
            
    #         # self._logger.info(f'Obtained camera matrix for point no. {index} {self.coordinates_of_chessboard[index]}: \n{mtx}')
    #         self._logger.info(f'Obtained camera matrix for point no. {index} {self.coordinates_of_chessboard[index]}: \n{np.array2string(mtx, formatter={"float_kind":lambda x: "%.6f" % x})}')            
    #         self.camera_mtx.append(mtx)

    # def calculate_rvec_tvec_from_robot_pose(self):
    #     for index, pos in enumerate(self.robot_positions):            
    #         pos = list(pos)
    #         self._logger.debug(f'Calculating rotation&translation for robot position no. {index}')
    #         # R_matrix = np.zeros((3, 3))

    #         roll = pos[3]
    #         pitch = pos[4]
    #         yaw = pos[5]
    #         rotation_matrix = common.RPY_to_rmtx(roll, pitch, yaw)
    #         # R_matrix[:3, :3] = rotation_matrix
    #         print(f"rotation_matrix: {rotation_matrix}")
    #         rvec = cv2.Rodrigues(np.array(rotation_matrix))[0]
    #         tvec = np.zeros((3, 1))
    #         for i in range(3):
    #             tvec[i] = pos[i]
    #         print(f"Calculated (rvec, tvec) for pose_{index}: {(rvec, tvec)}")
    #         self.robot_rvec_tvec_pairs.append((rvec, tvec))



    #     pass

    # def prepare_robot_mtx(self):
    #     """
    #     Prepares the rotation and translation 4x4 matrix from robot's TCP position.

    #     This method calculates the rotation and translation (TCP to robot's base) 4x4 matrix from 
    #     robot position (X, Y, Z, Roll, Pitch, Yaw). It stores result matrix in the class `robot_mtx` list.

    #     Returns:
    #         None
    #     """
    #     for index, pos in enumerate(self.robot_positions):            
    #         pos = list(pos)
    #         self._logger.debug(f'Calculating rotation&translation matrix for robot position no. {index}')
    #         matrix = np.zeros((4, 4))

    #         roll = pos[3]
    #         pitch = pos[4]
    #         yaw = pos[5]
    #         rotation_matrix = common.RPY_to_rmtx(roll, pitch, yaw)
    #         matrix[:3, :3] = rotation_matrix
    #         for i in range(3):
    #             matrix[i, 3] = pos[i]
    #         matrix[3, 3] = 1
            
    #         self._logger.info(f'Obtained robot matrix for point no. {index}: \n{matrix}')
    #         self.robot_mtx.append(matrix)


    def calibrate_tsai(self, A_list, B_list):
        """
        Performs Tsai's hand-eye calibration algorithm.

        This method calculates the the rotation and translation 4x4 matrix beetwen camera and robot base 
        using the Tsai hand-eye calibration algorithm with the camera and robot matrices
        stored in the `camera_mtx` and `robot_mtx` attributes of the class. Then saves the result to 
        a pickle file named 'tsai_results.pkl'.

        Returns:
            None
        """
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

        self.tsai_result = common.to_mtx(Rx, tX)
        print(f"Calculated Tsai result: \n{self.tsai_result}")
        
        with open('tsai_results.pkl', 'wb') as f:
            pickle.dump(self.tsai_result, f)

        print(f"Tsai result saved to tsai_results.pkl")
        return  self.tsai_result

    def calibrate_li(self, A_list, B_list):
        """
        Performs Li's hand-eye calibration algorithm.

        This method calculates the the rotation and translation 4x4 matrix beetwen camera and robot base 
        using the Li hand-eye calibration algorithm with the camera and robot matrices
        stored in the `camera_mtx` and `robot_mtx` attributes of the class. Then saves the result to 
        a pickle file named 'li_results.pkl'.

        Returns:
            None
        """

        # A_list = self.camera_mtx.copy()
        # B_list = self.robot_mtx.copy()
        
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

        self.li_result = common.to_mtx(Rx, tX)
        print(f"Calculated Li result: \n{self.li_result}")

        with open('li_results.pkl', 'wb') as f:
                pickle.dump(self.li_result, f)

        print(f"Li result saved to li_results.pkl")
        return  self.li_result


    # def calibrate_with_opencv(self):

    #     gripper_to_base_vecs = self.robot_rvec_tvec_pairs.copy()
    #     point_to_camera_vecs = self.rvec_tvec_pairs.copy()

    #     base_to_gripper_vecs = self.inverse_rvec_tvec_list(gripper_to_base_vecs)

    #     pass




    # def inverse_rvec_tvec_list(self, list):
    #     list_inv = []
    #     for index, (rvec, tvec) in enumerate(list):
            
    #         rvec_inv, tvec_inv = common.inverse_rmtx_tvec(rvec, tvec)
    #         list_inv.append(rvec_inv, tvec_inv)

    #     return list_inv