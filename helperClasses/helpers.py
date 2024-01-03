import numpy as np

def get_translation_funcs(global_points):
    court_points = np.array([
        [0, 0, -6.4],
        [4.11, 0, -6.4],
        [4.11, 0, -11.88]
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
    translation_vector = centroid_court - np.dot(rotation_matrix, centroid_global)

    return lambda x: np.dot(rotation_matrix, x) + translation_vector, lambda x: np.dot(rotation_matrix, x)
