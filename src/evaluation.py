import numpy as np
from sklearn import metrics


class AngleMAE:
    """Class for evaluation metrics AngleMAE which computes mean absolute
    error between set of joints angle of reference and test video keypoints

      Parameters:
        ref_keypoits : A numpy array [n, 1, 17, 3] containing keypoints
        representing poses in reference video
        test_keypoits : A numpy array [n, 1, 17, 3] containing keypoints
        representing poses in test video
    """

    def __init__(self):

        self.joints_array = np.array(
            [[11, 5, 7],
             [12, 6, 8],
             [6, 8, 10],
             [5, 7, 9],
             [11, 12, 14],
             [12, 11, 13],
             [12, 14, 16],
             [11, 13, 15],
             [5, 11, 13]]
            )

        self.joints_dict = {
            'left_shoulder_joint': ['left_hip', 'left_shoulder', 'left_elbow'],
            'right_shoulder_joint': ['right_hip', 'right_shoulder',
                                     'right_elbow'],
            'right_elbow_joint': ['right_shoulder', 'right_elbow',
                                  'right_wrist'],
            'left_elbow_joint': ['left_shoulder', 'left_elbow', 'left_wrist'],
            'right_hip_joint': ['left_hip', 'right_hip', 'right_knee'],
            'left_hip_joint': ['right_hip', 'left_hip', 'left_knee'],
            'right_knee_joint': ['right_hip', 'right_knee', 'right_ankle'],
            'left_knee_joint': ['left_hip', 'left_knee', 'left_ankle'],
            'waist_joint': ['left_shoulder', 'left_hip', 'left_knee']
            }

        self.angle_mae_joints_weightage_array = ([1, 1, 1, 1, 1, 1, 1, 1, 1])

    def mean_absolute_error(self, ref_keypoints, test_keypoints) -> object:
        """
        Calcultes MAE of given joints via index between reference and test
        frames

        Args:
          ref_keypoints: ndarray of shape (17,2) containing reference frame
          x, y coordinates
          test_keypoints: ndarray of shape (17,2) containing test frame x,
          y coordinates

        Returns:
          MAE: A float value representing angle based MAE
        """

        ref_angle = self.calculate_angle_atan2(ref_keypoints)
        test_angle = self.calculate_angle_atan2(test_keypoints)

        diff = np.abs(ref_angle - test_angle)

        mae = np.sum(diff * self.angle_mae_joints_weightage_array) / sum(
            self.angle_mae_joints_weightage_array
            )

        return mae

    def calculate_angle_atan2(self, kpts):
        """
        Calcultes angle of given joint

        Args:
          kpts: ndarray of shape (17,2) containing x, y coordinates

        Returns:
          angle: A float value representing angle in degrees
        """
        a = np.zeros((9, 2))
        b = np.zeros((9, 2))
        c = np.zeros((9, 2))

        for i, j in enumerate(self.joints_array):

            a[i] = kpts[j[0]]
            b[i] = kpts[j[1]]
            c[i] = kpts[j[2]]

        vector_b_a = b - a
        vector_b_c = b - c

        angle_0 = np.arctan2(
            vector_b_a[:, 1],
            vector_b_a[:, 0]
            )

        angle_2 = np.arctan2(
            vector_b_c[:, 1],
            vector_b_c[:, 0]
            )

        determinant = vector_b_a[:, 0] * vector_b_c[:, 1] - vector_b_a[:,
                                                            1] * vector_b_c[:,
                                                                 0]

        angle_diff = (angle_0 - angle_2)

        angle = np.degrees(angle_diff)
        joints_angle_array = angle * (determinant < 0) + (360 + angle) * (
            determinant > 0)

        return joints_angle_array % 360


class MAE:
    def __init__(self):
        pass

    @staticmethod
    def mean_absolute_error(ref_keypoints, test_keypoints):
        """
        Calcultes MAE of given keypoints between reference and test frames

        Args:
          ref_keypoints: ndarray of shape (17,2) containing reference frame
          x, y coordinates
          test_keypoints: ndarray of shape (17,2) containing test frame x,
          y coordinates

        Returns:
          MAE: A float value representing MAE
        """
        return metrics.mean_absolute_error(
            ref_keypoints.flatten(),
            test_keypoints.flatten(),
            )
