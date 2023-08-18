# from __future__ import absolute_import, division
from collections import defaultdict

import numpy as np

from src.evaluation import AngleMAE


class DTW:
    """Class for correspondence between two sequence of keypoints detected
    from videos

      Parameters:
        cost_weightage : dictionary containing weightage of MAE and AngleMAE
        to compute cost for DTW

    """

    def __init__(self, cost_weightage=None):
        self.anglemae_calculator = AngleMAE()

        self.cost_weightage = cost_weightage

    def cost(self, x, y):
        """computes cost for a set of keypoints through MAE or AngleMAE

        Args:
          x: A numpy array representing the keypoints of a reference frame
          y: A numpy array representing the keypoints of a test frame

        Returns:
          A float value representing the mae/angle mae score between
          reference and test pose
        """
        cost = 0
        if self.cost_weightage['angle_mae']:
            mae_angle = self.anglemae_calculator.mean_absolute_error(x, y) * \
                        self.cost_weightage['angle_mae']
            cost += mae_angle

        if self.cost_weightage['mae']:
            mae = self.MAE.mean_absolute_error(x, y) * self.cost_weightage[
                'mae']
            cost += mae

        return cost

    def find_correspondence(self, x, y):
        """applies Dynamic Time Warping algorithm to find correspondence
        between reference video and test video

        Args:
          x: A numpy array representing the keypoints of a reference video
          y: A numpy array representing the keypoints of a test video

        Returns:
          A tuple containing ref_frame_indices, test_frame_indices and costs
          where
            ref_frame_indices: A list of indices for reference video frames
            test_frame_indices : A list of indices for test video frames
            costs : A list of cost between reference and test keypoint
        """

        x = np.asanyarray(x, dtype='float')
        y = np.asanyarray(y, dtype='float')
        len_x, len_y = len(x), len(y)

        dtw_mapping = defaultdict(lambda: (float('inf'),))
        similarity_score = defaultdict(lambda: float('inf'), )

        dtw_mapping[0, 0] = (0, 0, 0)

        similarity_score[0, 0] = 0

        for i in range(1, len_x + 1):
            for j in range(1, len_y + 1):
                dt = self.cost(x[i - 1], y[j - 1])

                dtw_mapping[i, j] = min(
                    (dtw_mapping[i - 1, j][0] + dt, i - 1, j),
                    (dtw_mapping[i, j - 1][0] + dt, i, j - 1),
                    (dtw_mapping[i - 1, j - 1][0] + dt, i - 1, j - 1),
                    key=lambda a: a[0]
                    )
                similarity_score[i, j] = dt

        path = []
        i, j = len_x, len_y
        while not (i == j == 0):
            path.append(
                (i - 1, j - 1, dtw_mapping[i - 1, j - 1][0],
                 similarity_score[i - 1, j - 1])
                )
            i, j = dtw_mapping[i, j][1], dtw_mapping[i, j][2]
        path.reverse()

        ref_frame_idx, test_frame_idx, _, costs = DTW.get_ref_test_mapping(path)
        return (ref_frame_idx, test_frame_idx, costs)

    @staticmethod
    def get_ref_test_mapping(paths):
        """applies Dynamic Time Warping algorithm to find correspondence
        between reference video and test video

        Args:
          paths :  list of lists which consists of [i, j, ps, c] where i and
          j are index of x and y time series respectively which have the
          corespondence, ps is cummulative cost and c is cost between these
          two instances

        Returns:
          A tuple containing ref_frame_indices, test_frame_indices,path_score
          and costs where
            ref_frame_indices: A list of indices for reference video frames
            test_frame_indices : A list of indices for test video frames
            path_score : A list of path score calculated by DTW between
            reference and test keypoints
            costs : A list of cost between reference and test keypoint
        """

        path = np.array(paths)
        ref_2_test = {}
        for i in range(path.shape[0]):
            ref_2_test_val = ref_2_test.get(path[i][0], [])
            ref_2_test_val.append([path[i][1], path[i][2], path[i][3]])
            ref_2_test[path[i][0]] = ref_2_test_val
        ref_frames = []
        test_frames = []
        path_score = []
        costs = []

        for ref_frame, test_frame_list in ref_2_test.items():
            ref_frames.append(int(ref_frame))
            test_frame_list.sort(key=lambda x: x[2])
            test_frames.append(int(test_frame_list[0][0]))
            path_score.append(test_frame_list[0][1])
            costs.append(test_frame_list[0][2])

        return ref_frames, test_frames, path_score, costs
