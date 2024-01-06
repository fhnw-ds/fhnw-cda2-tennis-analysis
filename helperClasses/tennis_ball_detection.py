from datetime import datetime
import pyzed.sl as sl
import numpy as np
from tqdm import tqdm
import cv2
from numpy.lib import stride_tricks


class BallDetection:

    def __init__(self, svo_path):
        self.svo_path = svo_path
        self.median_background_l = None

    def get_ball_by_frame(self, frameFrom, frameTo, return_video = False):

        """Returns the ball position in a frame

        Args:
            frameFrom (int): Frame to start from
            frameTo (int): Frame to end at
            svo_path (str): Path to the svo file
            return_video (bool): Whether to return the video with the bounding box or just a list of ball positions, standard: False

        Returns:
            - Ball position: 2*1 array of the ball position of pixels in the frame (x, y)

        Examples:
            from BallDetection import get_ball_by_frame

            ball_pos = get_ball_by_frame(0, 100, 'path/to/svo/file.svo', median_background_l, depth_background_l)
        """

        if self.median_background_l is None:
            self.median_background_l = np.load('median_background_l.npy')



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

        current_frame_left = sl.Mat()

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
        list_of_ball_positions = []
        video = np.zeros((1080, 1920, 3, frameTo - frameFrom))

        for frame in tqdm(range(frameFrom, frameTo)):
            zed.set_svo_position(frame)

            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Get Images
                zed.retrieve_image(current_frame_left, sl.VIEW.LEFT)
                zed.retrieve_objects(detected_objects, detection_parameters_rt)

                # Get Arrays
                current_frame_left_data = current_frame_left.get_data()[:, :, :3]

                # Get Moving Pixels
                moving_left = current_frame_left_data - self.median_background_l

                # Get Tennis Ball Position
                tennis_ball_pos = self.detect_tennis_ball(moving_left, detected_objects, current_frame_left_data)

                if not return_video:
                    list_of_ball_positions.append(tennis_ball_pos)

                else:
                    tennis_ball_bb = self.draw_bb(current_frame_left_data, tennis_ball_pos)
                    frame_with_bb = tennis_ball_bb.copy()
                    frame_with_bb[tennis_ball_bb != 255] = current_frame_left_data[tennis_ball_bb != 255]
                    video[:, :, :, frame - frameFrom] = frame_with_bb

        if return_video:
            np.save(f'videoRaw.npy', video)
            self.make_mp4(video)
            return video
        else:
            return list_of_ball_positions

    @staticmethod
    def make_mp4(video, bildrate = 20, auflösung = (1920, 1080)):
        """
        Converts a video to a mp4 file.

        :param video (numpy array shape(x, y, 3, z): The video to convert.
        :param bildrate (int): The frame rate of the video. standard: 20
        :param auflösung (tuple): The resolution of the video. standard: (1920, 1080)

        :return: The converted video.
        """

        datestring = datetime.now().strftime("%Y-%m-%d-%H-%M")
        filename = f'renderedVideo-{datestring}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        video_komplett = cv2.VideoWriter(filename, fourcc, 20, (1920, 1080))  # Bildrate und Auflösung

        # Bilder zum Video hinzufügen
        for bild in tqdm(range(video.shape[3])):
            video_komplett.write(video[:, :, :, bild].astype(np.uint8))

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
        newRGBImage = np.zeros_like(image)
        newRGBImage[:, :, 0] = imageRed
        newRGBImage[:, :, 1] = imageGreen
        newRGBImage[:, :, 2] = imageBlue

        returnImage = newRGBImage.astype(np.uint8)
        return returnImage


    @staticmethod
    def get_area_of_interest(image, detected_objects):
        """
        Returns the area of interest for the image. The area of interest is the area between the lowest and the highest person in the image.

        :param image (numpy array shape(x, y, 3): The image to get the area of interest from.

        :return: The area of interest as a mask.
        """

        mask = np.zeros_like(image)

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

        mask[int(tiefstePersonLocation[1]):int(höchstePersonLocation[1]),
        int(tiefstePersonLocation[0]):int(höchstePersonLocation[0])] = 1

        return mask

    @staticmethod
    def find_yellow_areas(image, yellow_threshold = [(0, 180, 180), (200, 255, 255)]):
        """
        Finds the yellow areas in the image. The yellow areas are defined by the yellow_threshold.
        Attention: The ZED-Library returns BGR images and not RGB images.This function is therefore designed for BGR images.

        :param image (numpy array shape(x, y, 3): The image to find the yellow areas in.
        :param yellow_threshold (tuple): The threshold for the yellow areas. (Bluemin, Greenmin, Redmin), (Bluemax, Greenmax, Redmax)

        :return: A binary matrix with the yellow areas.
        """
        lower, upper = np.array(yellow_threshold[0]), np.array(yellow_threshold[1])
        mask = np.all(np.logical_and(lower <= image, image <= upper), axis=-1)

        return mask.astype(int)

    @staticmethod
    def sliding_window(image, window_size = (7, 7)):
        """
        Moves a window over the image and calculates the sum of the pixel values in the window.

        :param image (numpy array shape(x, y, 3): The image to move the window over.
        :param window_size (tuple): The size of the window. (height, width), standard: (7, 7)

        :return: A matrix with the sum of the pixel values in the window.
        """
        image_height, image_width = image.shape[:2]
        window_height, window_width = window_size

        # Berechnung der neuen Shape und Strides
        new_shape = (image_height - window_height + 1, image_width - window_width + 1, window_height, window_width)
        new_strides = (image.strides[0], image.strides[1], image.strides[0], image.strides[1])

        # Erstellung eines 'strided' Arrays, das die Fenster darstellt
        strided_image = stride_tricks.as_strided(image, shape=new_shape, strides=new_strides)

        # Berechnung der Summe in jedem Fenster
        window_sums = np.sum(strided_image, axis=(2, 3))

        return window_sums

    @staticmethod
    def get_mask(index, image, detected_objects, bounding_box_extension = 30):
        """
        Returns a mask for the bounding box of the detected object.

        :param index (int): The index of the detected object.
        :param image (numpy array shape(x, y, 3): The image to get the mask from.
        :param bounding_box_extension (int): The extension of the bounding box in pixels. standard: 30

        :return: A mask for the bounding box of the detected object.
        """

        mask = np.zeros_like(image)

        bb = detected_objects.object_list[index].bounding_box_2d

        bb[0, :] -= bounding_box_extension
        bb[1, :] += bounding_box_extension
        bb[2, :] += bounding_box_extension
        bb[3, :] -= bounding_box_extension

        # get the mask
        mask[int(bb[0, 1]):int(bb[2, 1]), int(bb[0, 0]):int(bb[1, 0])] = 1

        return mask

    def detect_tennis_ball(self, moving_pixels, detected_objects, original_picture):
        """
        Detects the tennis ball in the image.

        :param moving_pixels (numpy array shape(x, y, 3): The moving pixels in the image.
        :param detected_objects (sl.Objects): The detected objects in the image.
        :param original_picture (numpy array shape(x, y, 3): The original picture.

        :return: The position of the tennis ball in the image. (tuple): (x, y)
        """

        diff_pixels = np.square(moving_pixels.astype(np.uint8)).sum(axis=2)

        areaOfInterest = self.get_area_of_interest(diff_pixels, detected_objects)
        filteredByInterest = diff_pixels * areaOfInterest

        minPixelStrength = int(filteredByInterest.max() * 0.7)
        diff_pixels_dsa = filteredByInterest.copy()
        diff_pixels_dsa[diff_pixels_dsa < minPixelStrength] = 0

        masked = diff_pixels_dsa.copy()
        for i in range(len(detected_objects.object_list)):
            mask = self.get_mask(i, diff_pixels_dsa, detected_objects)
            masked[mask == 1.0] = 0.0

        yellow_areas = self.find_yellow_areas(original_picture)
        yellow_areas = yellow_areas * masked

        windows = self.sliding_window(yellow_areas, (7, 7))

        tennisballs = np.unravel_index(np.argsort(windows, axis=None)[::-1], windows.shape)
        return [tennisballs[0][0], tennisballs[1][0]]

    @staticmethod
    def draw_bb(image, center):
        """
        Draws a bounding box around the center of the tennis ball.

        :param image (numpy array shape(x, y, 3): The image to draw the bounding box on.
        :param center (tuple): The center of the tennis ball. (x, y)

        :return: The image with the bounding box.
        """
        bb = np.zeros_like(image)
        bb[center[0] - 15:center[0] + 15, center[1] - 15:center[1] + 15] = 255
        return bb

    def get_background(self, svo_path, calibration_frames = 120, skip_frames = 30):
        """
        Calculates the median background and the depth background of the left camera.

        :param svo_path (str): The path to the svo file.
        :param calibration_frames (int): The number of frames to use for the calculation. standard: 120
        :param skip_frames (int): The number of frames to skip between the frames used for the calculation. standard: 30

        :return: The median background and the depth background of the left camera.

        """

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
        temp_image_left = sl.Mat()
        # %%
        color_array_l = np.zeros((1080, 1920, 3, calibration_frames))
        # %%
        for i in tqdm(range(calibration_frames * skip_frames)):
            zed.grab(runtime_parameters)
            if i % skip_frames != 0:
                continue
            zed.retrieve_image(temp_image_left, sl.VIEW.LEFT)
            current_frame_l = temp_image_left.get_data()[:, :, :3]
            color_array_l[:, :, :, i // skip_frames] = current_frame_l

        # get median depth
        median_depth_l = np.nanmedian(color_array_l, axis=3)
        np.save('median_background_l.npy', median_depth_l)
        self.median_background_l = median_depth_l

        return median_depth_l
        zed.close()

    def load_background(self, background_path):
        """
        Loads the previously calculated median background of the left camera.

        :param background_path (str): The path to the background file. Has to be a .npy file.
        """
        self.median_background_l = np.load(background_path)
        return self.median_background_l
