import logging  
import numpy as np
from packages2.image_processor import ImageProcessor
import cv2

class ObjectPoseEstimator:
    def __init__(self, table_roi_points=None):
        self._logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self._logger.debug(f'ObjectPoseEstimator({self}) was initialized.')

        self.image_processor = ImageProcessor()

        # import calibration results and prepare it to use
        self.calib_data = np.load("CameraParams_13-07_12-15.npz")
        self.camera_matrix = self.calib_data["cameraMatrix"]
        self.dist_coeffs = self.calib_data["dist"]
        self.rvec = self.calib_data["rvec_for_pose"]
        self.tvec = self.calib_data["tvec_for_pose"]
        self.first_corner_coords = self.calib_data["first_corner_coords"]

        self.table_roi_points = table_roi_points

    def prepare_frame(self, frame):
        img = frame.copy()

        uimg = self.image_processor.undistort_frame(img, self.camera_matrix, self.dist_coeffs)

        # get ROI if not defined
        if self.table_roi_points is None:
            self.table_roi_points = self.image_processor.get_roi_points(uimg)
        
        gimg = cv2.cvtColor(uimg, cv2.COLOR_BGR2GRAY)
        iimg, _ = self.image_processor.ignore_background(gimg, vertices=self.table_roi_points)
        bimg = self.image_processor.apply_binarization(iimg)
        cv2.imshow("Image used to contour detection", bimg)
        cv2.waitKey(0)
        return bimg, uimg

    def calculate_2d_to_3d_pose(self, image_point):
        R, _ = cv2.Rodrigues(self.rvec)
        T = self.tvec
        
        c_x = self.camera_matrix[0, 2]
        c_y = self.camera_matrix[1, 2]
        f_x = self.camera_matrix[0, 0]
        f_y = self.camera_matrix[1, 1]
        print(f"\tc_x: {c_x}, c_y: {c_y}, f_x: {f_x}, f_y: {f_y}")

        # calculate pixel on image plane to normalized plane (don't depended on camera matrix and distortion coefficients)
        normalized_points = cv2.undistortPoints(image_point, self.camera_matrix, self.dist_coeffs)
        u_prime, v_prime = normalized_points[0, 0]

        print(f"\tu_prime: {u_prime}, v_prime: {v_prime}")
        Z = 0    # the same surface as calibration chessboard

        X = (u_prime - c_x) / f_x 
        Y = (v_prime - c_y) / f_y 
        print(f"\tX: {X}, Y: {Y}")
        camera_coords = np.array([X, Y, Z], dtype=np.float32).reshape(3, 1)
        # print(f"Camera coords: {camera_coords}")
        world_coords = np.dot(R.T, camera_coords - T)
        # print(f"Test coords: {camera_coords - T}")
        return world_coords

    def calculate_2d_to_3d_pose_stackoverflow(self, image_point):
        R, _ = cv2.Rodrigues(self.rvec)
        T = self.tvec
        
        c_x = self.camera_matrix[0, 2]
        c_y = self.camera_matrix[1, 2]
        f_x = self.camera_matrix[0, 0]
        f_y = self.camera_matrix[1, 1]
        print(f"\tc_x: {c_x}, c_y: {c_y}, f_x: {f_x}, f_y: {f_y}")

        # # calculate pixel on image plane to normalized plane (don't depended on camera matrix and distortion coefficients)
        normalized_points = cv2.undistortPoints(image_point, self.camera_matrix, self.dist_coeffs)
        u_prime, v_prime = normalized_points[0, 0]

        # u_prime, v_prime = image_point



        # Step 1: Convert to homogeneous coordinate (assuming the point is on the image plane, z=1)
        point_camera = np.array([u_prime * f_x + c_x, v_prime * f_y + c_y, 0.0])

        # Step 2: Apply the inverse extrinsic parameters
        # Calculate the inverse translation vector
        inv_translation = -np.dot(R.T, T.reshape(-1))

        # Transform the point to world coordinates
        point_world = np.dot(R.T, point_camera) + inv_translation

        return point_world


    def calculate_2d_to_3d_pose2(self, image_point):
        R, _ = cv2.Rodrigues(self.rvec)
        t = self.tvec
        
        c_x = self.camera_matrix[0, 2]
        c_y = self.camera_matrix[1, 2]
        f_x = self.camera_matrix[0, 0]
        f_y = self.camera_matrix[1, 1]
        print(f"\tc_x: {c_x}, c_y: {c_y}, f_x: {f_x}, f_y: {f_y}")

        image_point_reshaped = np.array([image_point], dtype=np.float32).reshape(-1, 1, 2)
        idealPts = cv2.undistortPoints(image_point_reshaped, self.camera_matrix, self.dist_coeffs)
        u_prime, v_prime = idealPts[0, 0, :]
        print(f"\tu_prime: {u_prime}, v_prime: {v_prime}")

        # Set lambda
        lambda_val = 1.0

        # Create cameraPt matrix
        cameraPt = np.zeros((3, 1), dtype=np.float64)
        cameraPt[0, 0] = idealPts[0, 0, 0] * lambda_val
        cameraPt[1, 0] = idealPts[0, 0, 1] * lambda_val
        cameraPt[2, 0] = lambda_val

        # Assuming R and t are already defined as the rotation matrix and translation vector, respectively

        # Create an identity matrix equivalent to cv::Mat::eye(4, 4, CV_64FC1)
        camToWorld = np.eye(4, dtype=np.float64)

        # Fill camToWorld with [R^T|-R^T*t]
        camToWorld[:3, :3] = R.T  # Assuming R is a 3x3 rotation matrix
        camToWorld[:3, 3] = (-R.T @ t).flatten()  # Assuming t is a 3x1 translation vector

        # Assuming cameraPt is already defined
        # Append 1.0 to cameraPt to make it homogeneous
        cameraPt_homogeneous = np.vstack([cameraPt, np.array([[1.0]])])

        # Perform matrix multiplication to get worldPt
        worldPt = camToWorld @ cameraPt_homogeneous

        return worldPt



    def calculate_2d_to_3d_pose_with_correction(self, image_point):
        print(f"First corner coords: {self.first_corner_coords}")
        print(f"Image point: {image_point}")
        # Skorygowanie punktu obrazu o pozycję punktu (0,0) szachownicy
        corrected_image_point = image_point - self.first_corner_coords
        print(f"Corrected image point: {corrected_image_point}")

        corrected_image_point_homogeneous = np.array([corrected_image_point[0], corrected_image_point[1], 1]).reshape(-1, 1)

        print(f"Corrected image point homogeneous: {corrected_image_point_homogeneous}")

        R, _ = cv2.Rodrigues(self.rvec)
        tvecs_last = self.tvec
        inv_camera_matrix = np.linalg.inv(self.camera_matrix)
        
        # Uwzględnienie korekty punktu referencyjnego w przekształceniu
        point_3d = np.dot(np.linalg.inv(R), (np.dot(inv_camera_matrix, corrected_image_point_homogeneous) - tvecs_last))

        return point_3d.flatten()



    def run(self, frame, origin_frame):
        image = frame.copy()
        image_to_show = origin_frame.copy()
        # find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(len(contours))

        # for loop over contours  
        for idx, contour in enumerate(contours, start=1):
            # find object centroid      
            center = self.image_processor.find_centroid(contour)
            print(f"\tCalculated centroid {idx}: {center}")
            
            # draw contour & centroid on image_to_show
            cv2.polylines(image_to_show, contour, True, (0, 0, 255), 3)
            center = (int(center[0]), int(center[1]))
            cv2.circle(image_to_show, center, 5, (255, 0, 0), -1)

            #check if centroid if applicable for gripper

            # if not, program algoritm to look around centroid for good spot for gripper    

            # prepare centroid or other point for computation
            image_point = center

            # compute 2D to 3D pose
            point_3d = self.calculate_2d_to_3d_pose_stackoverflow(image_point)
            print(f"Calculated 3D pose {idx}: {point_3d.flatten()}")

            # put 3D pose on image and show
            text = f"{idx}: {point_3d.flatten().astype(int)}"
            cv2.putText(image_to_show, text, (center[0] - 70, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            # save this data in way to connect it later with object

        cv2.imshow("Contours & centroids", image_to_show)
        cv2.waitKey(0)
