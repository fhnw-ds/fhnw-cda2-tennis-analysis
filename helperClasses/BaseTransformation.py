import numpy as np

class BaseTransformation:

    def __init__(self, global_points):
        self.global_points = global_points

    @staticmethod
    def getTranslationFuncs(global_points):
        """Transforms global points to court points

        Args:
            global_points (np.array): 3x3 array of global points

        Returns:
            - Translation function: Function that takes in a global point and returns a court point
            - Rotation function: TODO

        Examples:
            from baseTransformation import get_translation_funcs

            global_points = np.array([
                [0.404, -1.434, -13.462], # service line center
                [3.263, -1.300, -11.118], # service x singles sideline
                [0.446, -1.444, -7.882] # baseline x singles sideline
            ])

            translate_func, rotate_func = get_translation_funcs(global_points)

        """

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

        return lambda x: np.dot(rotation_matrix, x) + translation_vector, lambda x: np.dot(rotation_matrix, x)



    def transformData(self, data, centralisation):
        """Transforms x, y and z coordinates to court coordiantesystem

        Args:
            data (pd.DataFrame): Dataframe with x, y and z coordinates
            centralisation (np.array): 3x1 array with x, y and z factor to add to the transformed coordinates

        Returns:
            data (pd.DataFrame): Dataframe with transformed x, y and z coordinates

        Examples:
            from baseTransformation import BaseTransformation

            global_points = np.array([
                [0.404, -1.434, -13.462], # service line center
                [3.263, -1.300, -11.118], # service x singles sideline
                [0.446, -1.444, -7.882] # baseline x singles sideline
            ])

            centralisation = np.array([-1.5, 0, -6.4])
            data = pd.read_csv('df_detected_objects.csv')

            base_transformation = BaseTransformation(global_points)
            transformed_data = base_transformation.transform_Data(data, centralisation)

        """
        translation_func, rotation_func = self.get_translation_funcs(self.global_points)
        transformed_xyz = data[['x', 'y', 'z']].apply(lambda x: translation_func(x), axis=1)
        data['x'] = transformed_xyz.apply(lambda x: x[0] + centralisation[0])
        data['y'] = transformed_xyz.apply(lambda x: x[1] + centralisation[1])
        data['z'] = transformed_xyz.apply(lambda x: x[2] + centralisation[2])

        return data
