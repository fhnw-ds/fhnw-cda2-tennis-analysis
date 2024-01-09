from datetime import datetime
import pyzed.sl as sl
import numpy as np
from tqdm import tqdm
import cv2
from numpy.lib import stride_tricks
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import cProfile

class BallDetectionPytorch:

    def __init__(self, svo_path):
        self.svo_path = svo_path
        self.median_background_l = torch.tensor([]).to(device)
        self.median_background_r = torch.tensor([]).to(device)
        self.ball_positions = []

    def get_ball_by_frame(self, frameFrom, frameTo, camera = 'left', return_video = False):

        """Returns the ball position in a frame.

        Args:
            frameFrom (int): Frame to start from
            frameTo (int): Frame to end at
            camera (str): Camera to use, standard: 'left'
            svo_path (str): Path to the svo file
            return_video (bool): Whether to return the video with the bounding box or just a list of ball positions, standard: False

        Returns:
            - Ball position: 2*1 array of the ball position of pixels in the frame (x, y)

        Examples:
            from BallDetection import get_ball_by_frame

            ball_pos = get_ball_by_frame(0, 100, 'path/to/svo/file.svo', median_background_l, depth_background_l)
        """



        camera_lens = sl.VIEW
        if camera == 'left':
            camera_lens = sl.VIEW.LEFT
            tensor_median_background = self.median_background_l
        elif camera == 'right':
            camera_lens = sl.VIEW.RIGHT
            tensor_median_background = self.median_background_r
        else:
            raise Exception('camera must be either left or right')

        zed = sl.Camera()

        # init parameters
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(self.svo_path)
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_maximum_distance = 40
        init_params.depth_minimum_distance = 1
        init_params.sdk_verbose = True

        zed.open(init_params)

        current_frame = sl.Mat()

        # init detection parameters
        detection_parameters = sl.ObjectDetectionParameters()

        detection_parameters.image_sync = True
        detection_parameters.enable_tracking = True
        detection_parameters.enable_segmentation = True

        detection_parameters.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM

        if detection_parameters.enable_tracking:
            zed.enable_positional_tracking()

        zed.enable_object_detection(detection_parameters)

        detection_confidence = 10
        detection_parameters_rt = sl.ObjectDetectionRuntimeParameters(detection_confidence)

        detection_parameters_rt.object_class_filter = [sl.OBJECT_CLASS.PERSON]

        detected_objects = sl.Objects()
        runtime_parameters = sl.RuntimeParameters()

        last_ball_position = torch.tensor([0, 0]).to(device)
        last_picture = torch.zeros((1080, 1920, 3)).to(device)

        list_of_ball_positions = []
        video = torch.zeros((1080, 1920, 3, frameTo - frameFrom))
        zed.set_svo_position(frameFrom)
        current_depth_frame = sl.Mat()
        pointcloud = sl.Mat()
        list_of_ball_depths = []

        for frame in tqdm(range(frameFrom, frameTo)):
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Get Images
                zed.retrieve_image(current_frame, camera_lens)
                zed.retrieve_objects(detected_objects, detection_parameters_rt)


                # Get Arrays
                current_frame_data = current_frame.get_data()[:, :, :3]
                tensor_current_frame_data = torch.from_numpy(current_frame_data).to(device)

                # Get Moving Pixels
                tensor_moving = (tensor_current_frame_data - tensor_median_background).to(device)


                # Get Tennis Ball Position
                tensor_tennis_ball_pos = self.detect_tennis_ball(tensor_moving, detected_objects, tensor_current_frame_data, last_ball_position, last_picture).to(device)
                last_ball_position = tensor_tennis_ball_pos.clone()
                last_picture = tensor_current_frame_data.clone()

                list_of_ball_positions.append( [frame, tensor_tennis_ball_pos[0].cpu().item(), tensor_tennis_ball_pos[1].cpu().item()] )

                if return_video:
                    tensor_tennis_ball_bb = self.draw_bb(tensor_current_frame_data, tensor_tennis_ball_pos)
                    tensor_frame_with_bb = tensor_tennis_ball_bb.clone()
                    tensor_mask = tensor_tennis_ball_bb != 255
                    tensor_frame_with_bb[tensor_mask] = tensor_current_frame_data[tensor_mask]
                    video[:, :, :, frame - frameFrom] = tensor_frame_with_bb
            else:
                print('Error')

        zed.close()

        self.ball_positions = list_of_ball_positions
        if return_video:
            self.make_mp4(video, camera)
            return video

        return list_of_ball_positions

    @staticmethod
    def make_mp4(video, camera, bildrate = 20, auflösung = (1920, 1080)):
        """
        Converts a video to a mp4 file.

        :param video (numpy array shape(x, y, 3, z): The video to convert.
        :param bildrate (int): The frame rate of the video. standard: 20
        :param auflösung (tuple): The resolution of the video. standard: (1920, 1080)

        :return: The converted video.
        """
        video_numpy = video.cpu().numpy()
        video_numpy = video_numpy.astype(np.uint8)
        datestring = datetime.now().strftime("%Y-%m-%d-%H-%M")
        filename = f'renderedVideo-{datestring}_{camera}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        video_komplett = cv2.VideoWriter(filename, fourcc, 20, (1920, 1080))  # Bildrate und Auflösung

        # Bilder zum Video hinzufügen
        for bild in tqdm(range(video_numpy.shape[3])):
            video_komplett.write(video_numpy[:, :, :, bild])

        # Video speichern
        video_komplett.release()

    @staticmethod
    def convert_BGR_to_RGB(image):

        """
        Converts a BGR image to a RGB image.

        :param image (numpy array shape(x, y, 3): The image to convert.

        :return: The converted image.
        """

        imageBlue = image[:, :, 0]
        imageGreen = image[:, :, 1]
        imageRed = image[:, :, 2]
        newRGBImage = torch.zeros_like(image)
        newRGBImage[:, :, 0] = imageRed
        newRGBImage[:, :, 1] = imageGreen
        newRGBImage[:, :, 2] = imageBlue

        returnImage = newRGBImage.type(torch.uint8)
        return returnImage


    @staticmethod
    def get_area_of_interest(tensor_image, detected_objects):
        """
        Returns the area of interest for the image. The area of interest is the area between the lowest and the highest person in the image.

        :param image (numpy array shape(x, y, 3): The image to get the area of interest from.

        :return: The area of interest as a mask.
        """

        mask = torch.zeros_like(tensor_image)

        tiefstePersonSum = np.inf
        tiefstePersonLocation = [1080, 1920]
        höchstePersonSum = 0
        höchstePersonLocation = [0, 0]

        # get the bounding box
        for person in detected_objects.object_list:
            if person.label == sl.OBJECT_CLASS.PERSON:
                bb = person.bounding_box_2d
                for i in range(len(bb)):
                    koordinateSum = bb[i][0] + bb[i][1]
                    if koordinateSum > höchstePersonSum:
                        höchstePersonSum = koordinateSum
                        höchstePersonLocation = bb[i]
                    if koordinateSum < tiefstePersonSum:
                        tiefstePersonSum = koordinateSum
                        tiefstePersonLocation = bb[i]

        # extend the bounding box by 100 pixels
        tiefstePersonLocation[0] -= 100
        tiefstePersonLocation[1] -= 100
        höchstePersonLocation[0] += 100
        höchstePersonLocation[1] += 100

        mask[max(0, int(tiefstePersonLocation[1])):max(0, int(höchstePersonLocation[1])),
        max(0, int(tiefstePersonLocation[0])):max(0, int(höchstePersonLocation[0]))] = 1

        return mask

    @staticmethod
    def find_yellow_areas(tensor_image, device, yellow_threshold = [(0, 180, 180), (200, 255, 255)]):
        """
        Finds the yellow areas in the image. The yellow areas are defined by the yellow_threshold.
        Attention: The ZED-Library returns BGR images and not RGB images.This function is therefore designed for BGR images.

        :param image (numpy array shape(x, y, 3): The image to find the yellow areas in.
        :param yellow_threshold (tuple): The threshold for the yellow areas. (Bluemin, Greenmin, Redmin), (Bluemax, Greenmax, Redmax)

        :return: A binary matrix with the yellow areas.
        """
        lower, upper = torch.tensor(yellow_threshold[0]).to(device), torch.tensor(yellow_threshold[1]).to(device)
        mask = torch.all((lower <= tensor_image) & (tensor_image <= upper), dim=-1)

        return mask.int()

    @staticmethod
    def sliding_window(tensor_image, window_size = (7, 7)):
        """
        Moves a window over the image and calculates the sum of the pixel values in the window.

        :param image (numpy array shape(x, y, 3): The image to move the window over.
        :param window_size (tuple): The size of the window. (height, width), standard: (7, 7)

        :return: A matrix with the sum of the pixel values in the window.
        """
        image_height, image_width = tensor_image.shape[0], tensor_image.shape[1]
        window_height, window_width = window_size

        windows = tensor_image.unfold(0, window_height, 1).unfold(1, window_width, 1)

        # Berechnung der Summe in jedem Fenster
        windows = windows.float()
        tensor_window_sums = windows.mean(dim=[2, 3])

        return tensor_window_sums

    @staticmethod
    def get_mask(index, tensor_image, detected_objects, bounding_box_extension = 30):
        """
        Returns a mask for the bounding box of the detected object.

        :param index (int): The index of the detected object.
        :param image (numpy array shape(x, y, 3): The image to get the mask from.
        :param bounding_box_extension (int): The extension of the bounding box in pixels. standard: 30

        :return: A mask for the bounding box of the detected object.
        """

        mask = torch.zeros_like(tensor_image)

        bb = detected_objects.object_list[index].bounding_box_2d

        bb[0, :] -= bounding_box_extension
        bb[1, :] += bounding_box_extension
        bb[2, :] += bounding_box_extension
        bb[3, :] -= bounding_box_extension

        # get the mask
        mask[int(bb[0, 1]):int(bb[2, 1]), int(bb[0, 0]):int(bb[1, 0])] = 1

        return mask

    def detect_tennis_ball(self, tensor_moving_pixels, detected_objects, tensor_original_picture, last_ball_position, previous_picture):
        """
        Detects the tennis ball in the image.

        :param moving_pixels (numpy array shape(x, y, 3): The moving pixels in the image.
        :param detected_objects (sl.Objects): The detected objects in the image.
        :param original_picture (numpy array shape(x, y, 3): The original picture.

        :return: The position of the tennis ball in the image. (tuple): (x, y)
        """
        tensor_original_picture_gpu = tensor_original_picture.to(device)
        tensor_moving_pixels_gpu = tensor_moving_pixels.to(device)
        tensor_moving_pixels_relative_previous = (tensor_original_picture_gpu - previous_picture).to(device)

        tensor_diff_pixels_relative_previous = torch.square(tensor_moving_pixels_relative_previous.type(torch.uint8)).sum(dim=2).to(device)
        tensor_diff_pixels = torch.square(tensor_moving_pixels_gpu.type(torch.uint8)).sum(dim=2).to(device)
        #diff_pixels = np.square(moving_pixels.astype(np.uint8)).sum(axis=2)

        tensor_final_diff_pixels = (tensor_diff_pixels * tensor_diff_pixels_relative_previous).to(device)

        tensor_areaOfInterest = self.get_area_of_interest(tensor_final_diff_pixels, detected_objects).to(device)
        tensor_filteredByInterest = (tensor_diff_pixels * tensor_areaOfInterest).to(device)

        #tensor_minPixelStrength = int(tensor_filteredByInterest.max() * 0.7)

        #tensor_diff_pixels_dsa = tensor_filteredByInterest.clone().to(device)
        #tensor_diff_pixels_dsa[tensor_diff_pixels_dsa < tensor_minPixelStrength] = 0

        tensor_diff_pixels_dsa = tensor_filteredByInterest.clone().to(device)
        if (last_ball_position[0] != 0 and last_ball_position[1] != 0):
            tensor_areaofBall = torch.zeros_like(tensor_filteredByInterest).to(device)
            tensor_areaofBall[last_ball_position[0] - 200:last_ball_position[0] + 200, last_ball_position[1] - 200:last_ball_position[1] + 200] = 1
            tensor_diff_pixels_dsa = (tensor_filteredByInterest * tensor_areaofBall).to(device)


        tensor_masked = tensor_diff_pixels_dsa.clone().to(device)

        for i in range(len(detected_objects.object_list)):
            mask = self.get_mask(i, tensor_diff_pixels_dsa, detected_objects, 15).to(device)
            tensor_masked[mask == 1.0] = 0.0

        tensor_yellow_areas = self.find_yellow_areas(tensor_original_picture_gpu, device).to(device)
        tensor_yellow_areas = (tensor_yellow_areas * tensor_masked).to(device)

        tensor_windows = self.sliding_window(tensor_yellow_areas, (7, 7)).to(device)

        max_index = torch.argmax(tensor_windows).to(device)
        y_coord = max_index // tensor_windows.shape[1]
        x_coord = max_index % tensor_windows.shape[1]

        tensor_coordinates = torch.tensor([y_coord, x_coord]).to(device)

        return tensor_coordinates

    @staticmethod
    def draw_bb(tensor_image, tensor_center):
        """
        Draws a bounding box around the center of the tennis ball.

        :param image (numpy array shape(x, y, 3): The image to draw the bounding box on.
        :param center (tuple): The center of the tennis ball. (x, y)

        :return: The image with the bounding box.
        """
        bb = torch.zeros_like(tensor_image).to(device)
        bb[tensor_center[0] - 15:tensor_center[0] + 15, tensor_center[1] - 15:tensor_center[1] + 15] = 255
        return bb

    def calculate_background(self, svo_path, camera = 'left', calibration_frames = 120, skip_frames = 30):
        """
        Calculates the median background and the depth background of the left camera.

        :param svo_path (str): The path to the svo file.
        :param calibration_frames (int): The number of frames to use for the calculation. standard: 120
        :param skip_frames (int): The number of frames to skip between the frames used for the calculation. standard: 30

        :return: The median background and the depth background of the left camera.

        """

        camera_lens = sl.VIEW
        if camera == 'left':
            camera_lens = sl.VIEW.LEFT
        elif camera == 'right':
            camera_lens = sl.VIEW.RIGHT
        else:
            raise Exception('camera must be either left or right')


        svo_path = svo_path
        zed = sl.Camera()

        # init parameters
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(svo_path)
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_maximum_distance = 40
        init_params.depth_minimum_distance = 1
        init_params.sdk_verbose = True

        zed.open(init_params)

        runtime_parameters = sl.RuntimeParameters()
        temp_image = sl.Mat()
        color_array = torch.zeros((1080, 1920, 3, calibration_frames)).to(device)
        range_count = calibration_frames * skip_frames
        for i in tqdm(range(range_count)):
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                if i % skip_frames != 0:
                    continue
                zed.retrieve_image(temp_image, camera_lens)
                current_frame = temp_image.get_data()[:, :, :3]
                color_array[:, :, :, i // skip_frames] = torch.from_numpy(current_frame)

        # get median depth
        median_depth = color_array.nanmedian(3).values
        if camera == 'left':
            self.median_background_l = median_depth
        else:
            self.median_background_r = median_depth
        zed.close()
        return median_depth

    def save_background(self, camera = 'left', path = 'median_background'):
        '''
        Saves the median background of a video to a local file.

        :param path (str): The path to the file. standard: 'median_background'
        '''
        if camera == 'left':
            torch.save(self.median_background_l, path +'_l.pt')
        elif camera == 'right':
            torch.save(self.median_background_r, path +'_r.pt')
        else:
            raise Exception('camera must be either left or right')

    def load_background(self, camera = 'left', path = 'median_background'):
        '''
        Loads the median background of a video from a local file.

        :param path (str): The path to the file. standard: 'median_background.pt'

        :return: The median background.
        '''

        if camera == 'left':
            self.median_background_l = torch.load(path +'_l.pt')
            return self.median_background_l
        elif camera == 'right':
            self.median_background_r = torch.load(path +'_r.pt')
            return self.median_background_r
        else:
            raise Exception('camera must be either left or right')

    def triangulation(self, pt1, pt2):

        """
        Trianguliert einen Punkt aus zwei Ansichten.

        :param K: Die Kameramatrix.
        :param R: Die Rotationsmatrix, die die Ausrichtung der zweiten Kamera relativ zur ersten beschreibt.
        :param t: Der Translationsvektor, der die Position der zweiten Kamera relativ zur ersten beschreibt.
        :param pt1: Der Punkt im ersten Bild (in Pixelkoordinaten).
        :param pt2: Der Punkt im zweiten Bild (in Pixelkoordinaten).
        :return: Die 3D-Koordinaten des Punktes im Weltkoordinatensystem.
        """

        zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.set_from_svo_file(self.svo_path)
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_maximum_distance = 40
        init_params.depth_minimum_distance = 1
        init_params.sdk_verbose = True

        zed.open(init_params)

        calib_params = zed.get_camera_information().camera_configuration.calibration_parameters
        intrinsics = calib_params.left_cam
        K = np.array([[intrinsics.fx, 0, intrinsics.cx],
                      [0, intrinsics.fy, intrinsics.cy],
                      [0, 0, 1]])

        R = calib_params.left_cam.disto.reshape(3,4)
                    #[x, y, z]
        #t = calib_params.T


        # Erstelle Projektionsmatrizen
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        #P2 = K @ np.hstack((R, t))
        P2 = K@ R

        print(P1)
        print(P2)

        pts1 = np.array([[pt1[0]], [pt1[1]]])
        pts2 = np.array([[pt2[0]], [pt2[1]]])

        print(pts1)
        print(pts2)

        # Trianguliere den Punkt
        point_3d_hom = cv2.triangulatePoints(projMatr1 = P1, projMatr2 = P2, projPoints1 = pts1, projPoints2 = pts2)

        # Konvertiere zurück in nicht-homogene Koordinaten
        point_3d = point_3d_hom[:3] / point_3d_hom[3]

        zed.close()
        return point_3d.ravel()






















