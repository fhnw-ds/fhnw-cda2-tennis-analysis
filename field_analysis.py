# %%
import pyzed.sl as sl
import numpy as np
import pandas as pd

# %%
svo_path = '../data/Schlieren/svo/HD1080_SN35520970_11-03-24.svo'

# %%
# Define the known points in both the global and court's coordinate systems
calibration_points = np.array([
    [0.404, -1.434, -13.462], # service line center
    [3.263, -1.300, -11.118], # service x singles sideline
    [0.446, -1.444, -7.882] # baseline x singles sideline
])

# %%
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

# %%
# init detection parameters
detection_parameters = sl.ObjectDetectionParameters()

detection_parameters.image_sync = True
detection_parameters.enable_tracking = True
detection_parameters.enable_segmentation = True

detection_parameters.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM

if detection_parameters.enable_tracking:
    zed.enable_positional_tracking()

zed.enable_object_detection(detection_parameters)

detection_confidence = 20
detection_parameters_rt = sl.ObjectDetectionRuntimeParameters(detection_confidence)

detection_parameters_rt.object_class_filter = [sl.OBJECT_CLASS.PERSON, sl.OBJECT_CLASS.SPORT]

# %%
detected_objects = sl.Objects()
runtime_parameters = sl.RuntimeParameters()

detected_objects_list = []

while zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
    # Retrieve objects
    zed.retrieve_objects(detected_objects, detection_parameters_rt)

    frame_nr = zed.get_svo_position()

    for i, obj in enumerate(detected_objects.object_list):
        if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
            detected_objects_list.append({
                'obj': obj,
                'obj_nr': i,
                'frame_nr': frame_nr
            })

# %%
df_detected_objects = pd.DataFrame(
    data=[
        [
            obj['frame_nr'],
            obj['obj_nr'],
            obj['obj'].id,
            obj['obj'].label,
            obj['obj'].confidence,
            obj['obj'].tracking_state,
            obj['obj'].position[0],
            obj['obj'].position[1],
            obj['obj'].position[2],
            obj['obj'].velocity[0],
            obj['obj'].velocity[1],
            obj['obj'].velocity[2],
            obj['obj'].dimensions[0],
            obj['obj'].dimensions[1],
            obj['obj'].dimensions[2]
        ] for obj in detected_objects_list
    ],
    columns=['frame', 'object', 'object_id', 'object_label', 'confidence', 'tracking_state', 'x', 'y', 'z', 'vx', 'xy', 'vz', 'width', 'height', 'length']
)

# %%
df_detected_objects.to_csv('df_detected_objects.csv', index=False, float_format='%.5f')

# %%
df_detected_objects = pd.read_csv('df_detected_objects.csv')

# %%
def get_translation_funcs(global_points):
    court_points = np.array([
        [0, 0, 6.4],
        [4.11, 0, 6.4],
        [4.11, 0, 11.88]
    ])

    # Calculate the centroids of the points
    centroid_global = np.mean(global_points, axis=0)
    centroid_court = np.mean(court_points, axis=0)

    # Center the points by subtracting the centroids
    centered_global = global_points - centroid_global
    centered_court = court_points - centroid_court

    # Calculate the covariance matrix
    covariance_matrix = np.dot(centered_global.T, centered_court)

    # Perform singular value decomposition
    U, S, Vt = np.linalg.svd(covariance_matrix)

    # Calculate the rotation matrix
    rotation_matrix = np.dot(U, Vt)

    # If the determinant of the rotation matrix is -1, adjust for proper orientation
    if np.linalg.det(rotation_matrix) < 0:
        Vt[2, :] *= -1
        rotation_matrix = np.dot(U, Vt)

    # Calculate the translation vector
    translation_vector = centroid_court + np.dot(rotation_matrix, centroid_global)

    print(translation_vector)

    return lambda x: np.dot(rotation_matrix, x) + translation_vector, lambda x: np.dot(rotation_matrix, x)

translation_func, rotation_func = get_translation_funcs(calibration_points)

# %%
df_transformed_objects = df_detected_objects.copy()
transformed_xyz = df_transformed_objects[['x', 'y', 'z']].apply(lambda x: translation_func(x), axis=1)
df_transformed_objects['x'] = transformed_xyz.apply(lambda x: x[0] - 1.5)
df_transformed_objects['y'] = transformed_xyz.apply(lambda x: x[1])
df_transformed_objects['z'] = transformed_xyz.apply(lambda x: x[2] - 6.4)

# %%
df_transformed_objects.plot.scatter(x='frame', y='object_id', colormap='viridis')

# %%
df_transformed_objects['object_id'].value_counts().head(10)

# %%
import matplotlib.pyplot as plt
df_filtered = df_transformed_objects[~df_transformed_objects['object_id'].isin([0,1,59])]

fig, ax = plt.subplots(figsize=(10, 10))

plt.scatter(x=df_filtered['x'], y=df_filtered['z'], c=df_filtered['frame'], cmap='viridis')
plt.plot([4.11, 4.11], [0, 23.77], 'k', lw=2)
plt.plot([5.48, 5.48], [0, 23.77], 'k', lw=2)
plt.plot([-4.11, -4.11], [0, 23.77], 'k', lw=2)
plt.plot([-5.48, -5.48], [0, 23.77], 'k', lw=2)
plt.plot([0, 0], [5.48, 18.28], 'k', lw=2)
plt.plot([-5.48, 5.48], [11.88, 11.88], 'k', lw=2)
plt.plot([-5.48, 5.48], [0, 0], 'k', lw=2)
plt.plot([-5.48, 5.48], [23.77, 23.77], 'k', lw=2)
plt.plot([-4.11, 4.11], [5.48, 5.48], 'k', lw=2)
plt.plot([-4.11, 4.11], [18.28, 18.28], 'k', lw=2)

plt.colorbar()

ax.set_aspect('equal')

plt.show()

# %%
df_transformed_objects.plot.hexbin(x='x', y='z', cmap='viridis', gridsize=50)

# %%
zed.disable_object_detection()
zed.disable_positional_tracking()
zed.close()


