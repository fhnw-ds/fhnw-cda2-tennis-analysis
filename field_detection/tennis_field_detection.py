import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import pyzed.sl as sl

def calculate_slope_intercept(line):
    x1, y1, x2, y2 = line[0]
    if x2 - x1 == 0:  # To avoid division by zero
        return float('inf'), y1 - (float('inf') * x1)  # Infinite slope, intercept calculated using y1
    slope = (y2 - y1) / (x2 - x1)
    y_intercept = y1 - slope * x1
    return y_intercept, slope

def group_and_combine_lines(lines, width):
    line_properties = []
    for line in lines:
        y_intercept, slope = calculate_slope_intercept(line)
        line_properties.append([y_intercept, slope])
    
    clusters = DBSCAN(eps=10, min_samples=1, n_jobs=-1).fit(line_properties)

    # get left most and right most lines for each cluster
    combined_lines = []
    for cluster in set(clusters.labels_):

        # get all lines in cluster
        cluster_lines = []
        for cluster_label, line in zip(clusters.labels_, lines):
            if cluster_label == cluster:
                cluster_lines.append(line)
        
        # get left most and right most lines
        left_most_line = min(cluster_lines, key=lambda x: x[0][0])
        right_most_line = max(cluster_lines, key=lambda x: x[0][2])

        # create new line from left most and right most lines
        x1 = left_most_line[0][0]
        y1 = left_most_line[0][1]
        x2 = right_most_line[0][2]
        y2 = right_most_line[0][3]

        if x1 == 0 and x2 == width - 1:
            continue
    
        properties = calculate_slope_intercept([[x1, y1, x2, y2]])

        combined_lines.append((properties, (x1, y1, x2, y2)))

    return combined_lines

def get_intersection(line1, line2):
    y_intercept1, slope1 = line1
    y_intercept2, slope2 = line2
    x = round((y_intercept2 - y_intercept1) / (slope1 - slope2))
    y = round(slope1 * x + y_intercept1)
    return x, y

def detect_calibration_points(gray):
    edges = cv2.Canny(gray, 50, 200)
    # add dilation to close gaps in edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    lines = cv2.HoughLinesP(edges, 1, np.pi/360, 40, minLineLength=250, maxLineGap=7)

    filtered_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate the angle of the line using arctan2
        angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

        if ((0 <= angle <= 3) or (15 <= angle <= 25)):
            filtered_lines.append(line)

    combined_lines = group_and_combine_lines(filtered_lines, gray.shape[1])

    # get horizontal lines closest to bottom of image
    horizontal_lines = [horizontal_line for horizontal_line in combined_lines if np.abs(horizontal_line[0][1]) < 0.1]
    horizontal_lines.sort(key=lambda x: x[1][1], reverse=True)

    side_lines = [side_line for side_line in combined_lines if np.abs(side_line[0][1]) > 0.1]

    left_side_lines = [side_line for side_line in side_lines if side_line[0][1] < 0]
    left_side_lines.sort(key=lambda x: x[0][0])

    right_side_lines = [side_line for side_line in side_lines if side_line[0][1] > 0]
    right_side_lines.sort(key=lambda x: x[0][0])

    base_line = horizontal_lines[0]
    service_line = horizontal_lines[1]
    left_side_line = left_side_lines[0]
    right_side_line = right_side_lines[0]
    
    base_l = get_intersection(base_line[0], left_side_line[0])
    base_r = get_intersection(base_line[0], right_side_line[0])
    service_l = get_intersection(service_line[0], left_side_line[0])
    service_r = get_intersection(service_line[0], right_side_line[0])

    base_center = (base_l[0] + base_r[0]) // 2, (base_l[1] + base_r[1]) // 2
    service_center = (service_l[0] + service_r[0]) // 2, (service_l[1] + service_r[1]) // 2

    return {
        'base_center': base_center,
        'service_center': service_center,
        'base_l': base_l,
        'base_r': base_r,
        'service_l': service_l,
        'service_r': service_r
    }

if __name__ == '__main__':
    import time
    start_time = time.time()

    cv_window = 'Calibration Points'
    svo_path = '../../data/Schlieren_20230913/HD1080_SN35071549_10-32-34.svo'
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
    calibration_frames = 10
    skip_frames = 30
    runtime_parameters = sl.RuntimeParameters()
    temp_image_left = sl.Mat()
    temp_image_right = sl.Mat()
    color_array_l = np.zeros((1080, 1920, 3, calibration_frames))
    color_array_r = np.zeros((1080, 1920, 3, calibration_frames))
    for i in range(calibration_frames * skip_frames):
        zed.grab(runtime_parameters)
        if i % skip_frames != 0:
            continue
        frame_nr = zed.get_svo_position()
        # zed.retrieve_measure(image, sl.MEASURE.DEPTH)
        zed.retrieve_image(temp_image_left, sl.VIEW.LEFT)
        current_frame_l = temp_image_left.get_data()[:, :, :3]
        color_array_l[:, :, :, i // skip_frames] = current_frame_l
        zed.retrieve_image(temp_image_right, sl.VIEW.RIGHT)
        current_frame_r = temp_image_right.get_data()[:, :, :3]
        color_array_r[:, :, :, i // skip_frames] = current_frame_r

    median_l = np.nanmedian(color_array_l, axis=3)

    median_l = median_l.astype(np.uint8)

    src = median_l.copy()
    src_gray = cv2.cvtColor(median_l, cv2.COLOR_BGR2GRAY)

    # color top half of image black
    src_gray[:src_gray.shape[0]//7*3, :] = 0

    calibration_points = detect_calibration_points(src_gray)

    end_time = time.time()

    print(f'Calibration took {end_time - start_time} seconds')

    cv2.namedWindow(cv_window, )

    # draw calibration points
    for point in calibration_points.values():
        cv2.circle(src, point, 5, (0, 0, 255), -1)
        
    cv2.imshow(cv_window, src)

    # wait for q to quit
    while cv2.waitKey(0) != ord('q'):
        pass