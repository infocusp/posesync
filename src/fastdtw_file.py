from __future__ import absolute_import, division

from collections import defaultdict

import numpy as np
from pose_estimation import joints_angle


def fastdtw(x, y, radius=1, dist=None, method={}, angle_comp_method=''):
    ''' return the approximate distance between 2 time series with O(N)
        time and memory complexity

        Parameters
        ----------
        x : array_like
            input array 1
        y : array_like
            input array 2
        radius : int
            size of neighborhood when expanding the path. A higher value will
            increase the accuracy of the calculation but also increase time
            and memory consumption. A radius equal to the size of x and y
            will yield an exact dynamic time warping calculation.
        dist : function or int
            The method for calculating the distance between x[i] and y[j]. If
            dist is an int of value p > 0, then the p-norm will be used. If
            dist is a function then dist(x[i], y[j]) will be used. If dist is
            None then abs(x[i] - y[j]) will be used.

        Returns
        -------
        distance : float
            the approximate distance between the 2 time series
        path : list
            list of indexes for the inputs x and y

    '''
    x = np.asanyarray(x, dtype='float')
    y = np.asanyarray(y, dtype='float')

    return __dtw(x, y, None, angle_comp_method=angle_comp_method)


def cost(x, y, method, angle_comp_method=''):

    mae_angle = joints_angle.mean_absolute_error(
        x, y
        )
    cost = mae_angle

    return cost


def __dtw(x, y, method, angle_comp_method=''):
    len_x, len_y = len(x), len(y)

    dtw_mapping = defaultdict(lambda: (float('inf'),))
    similarity_score = defaultdict(lambda: float('inf'), )

    dtw_mapping[0, 0] = (0, 0, 0)

    similarity_score[0, 0] = 0

    for i in range(1, len_x + 1):
        for j in range(1, len_y + 1):
            dt = cost(
                x[i - 1], y[j - 1], method, angle_comp_method=angle_comp_method
                )
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
    return (dtw_mapping[len_x, len_y][0], path)
