import os

import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip


def get_video_frames(video_path):
    """Reads a video frame by frame

    Args:
      video_path: A video path

    Returns:
      A list of numpy arrays representing video frames
    """

    vid_cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = vid_cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            break

    return frames


class Plot:
    """Class for plotting keypoints on video frames and creating a video

    """
    KEYPOINT_EDGE_INDS_TO_COLOR = {
        (0, 1): "m",
        (0, 2): "c",
        (1, 3): "m",
        (2, 4): "c",
        (0, 5): "m",
        (0, 6): "c",
        (5, 7): "m",
        (7, 9): "m",
        (6, 8): "c",
        (8, 10): "c",
        (5, 6): "y",
        (5, 11): "m",
        (6, 12): "c",
        (11, 12): "y",
        (11, 13): "m",
        (13, 15): "m",
        (12, 14): "c",
        (14, 16): "c",
        }

    @staticmethod
    def add_keypoints_to_image(
        images_array, keypoints_list
        ):
        """
        Adds keypoints to the image

        Args:
          images_array: list of images to represent the keypoints
          keypoints_list : list of keypoints
          model: name of the model used to detect the keypoints
        Returns:
          None
        """

        output_overlay_array = images_array.astype(np.int32)
        output_overlay_list = Plot.draw_prediction_on_image(
            output_overlay_array, keypoints_list
            )
        return np.array(output_overlay_list)

    @staticmethod
    def draw_prediction_on_image(
        image_list, keypoints_list
        ):

        """Draws the keypoint predictions on image.

        Args:
          image_list: A numpy array with shape [n, height, width, channel]
          representing the
            pixel values of the input image where n is number of images.
          keypoints_list: A numpy array with shape [n, 17, 2] representing the
            coordinates of 17 keypoints where n is number of images.

        Returns:
          A numpy array with shape [n, out_height, out_width, channel]
          representing
          the list of
          images overlaid with keypoint predictions.
        """
        height, width, channel = image_list[0].shape

        keypoint_locs, keypoint_edges = Plot._keypoints_and_edges_for_display(
            keypoints_list,
            height, width
            )
        for img_i in range(keypoint_locs.shape[0]):
            for edge in keypoint_edges[img_i]:
                image = cv2.line(
                    image_list[img_i], (int(edge[0]), int(edge[1])),
                    (int(edge[2]), int(edge[3])), color=(0, 0, 255), thickness=3
                    )

            for center_x, center_y in keypoint_locs[img_i]:
                image = cv2.circle(
                    image_list[img_i], (int(center_x), int(center_y)), radius=5,
                    color=(255, 0, 0), thickness=-1
                    )

            image_list[img_i] = image

        return image_list

    @staticmethod
    def _keypoints_and_edges_for_display(
        keypoints_list, height, width,
        ):
        """Returns high confidence keypoints and edges for visualization.

        Args:
          keypoints_list: A numpy array with shape [1, 1, 17, 3] representing
          the keypoint coordinates and scores returned from the MoveNet model.
          height: height of the image in pixels.
          width: width of the image in pixels.
          keypoint_threshold: minimum confidence score for keypoint to be
          visualized.

        Returns:
          A (kpts_absolute_xy, edges_xy) containing:
            * array with shape [n, 17, 2] representing the coordinates of all
            keypoints of all detected entities in n images;
            * array with shape [n, 18, 4] representing the coordinates of all
            skeleton edges of all detected entities in n images;
        """
        kpts_x = width * keypoints_list[:, :, 0]
        kpts_y = height * keypoints_list[:, :, 1]

        edge_pair = np.array(list(Plot.KEYPOINT_EDGE_INDS_TO_COLOR.keys()))
        kpts_absolute_xy = np.stack(
            [kpts_x, kpts_y], axis=-1
            )

        x_start = kpts_x[:, edge_pair[:, 0]]
        y_start = kpts_y[:, edge_pair[:, 0]]
        x_end = kpts_x[:, edge_pair[:, 1]]
        y_end = kpts_y[:, edge_pair[:, 1]]

        edges = np.stack([x_start, y_start, x_end, y_end], axis=2)
        return kpts_absolute_xy, edges

    @staticmethod
    def resize_and_concat(ref_image_list, test_image_list):
        """Resizes either of reference frames list or test frames list to
        make both list of equal shape and merges both frames side by side

        Args:
          ref_image_list: A list of numpy array representing reference video
          frames
          test_image_list: A list of numpy array representing test video frames

        Returns:
          concat_img_list: A list of numpy array representing merged video
          frames

        """

        def pad_image(image_list, pad_axis, pad_len, odd_len_diff):
            """pads given number of pixels to image_list on given axis

            Args:
              image_list: A list of numpy array representing video frames
              pad_axis: A list of numpy array representing test video frames
              pad_len: number of pixels to pad on either side of frame
              odd_len_diff: 1 if difference between reference and test is odd
              else 0

            Returns:
              padded video frame

            """
            if pad_axis == 0:
                return np.pad(
                    image_list, (
                        (0, 0), (pad_len, pad_len + odd_len_diff), (0, 0),
                        (0, 0)), 'constant',
                    constant_values=(0)
                    )
            elif pad_axis == 1:
                return np.pad(
                    image_list, (
                        (0, 0), (0, 0), (pad_len, pad_len + odd_len_diff),
                        (0, 0)), 'constant',
                    constant_values=(0)
                    )

        ref_height, ref_width, _ = ref_image_list[0].shape
        test_height, test_width, _ = test_image_list[0].shape

        pad_height = abs(test_height - ref_height) // 2
        odd_height_diff = (test_height - ref_height) % 2

        if ref_height < test_height:
            ref_image_list = pad_image(
                ref_image_list, 0, pad_height, odd_height_diff
                )
        elif ref_height > test_height:
            test_image_list = pad_image(
                test_image_list, 0, pad_height, odd_height_diff
                )

        pad_width = abs(test_width - ref_width) // 2
        odd_width_diff = (test_width - ref_width) % 2

        if ref_width < test_width:
            ref_image_list = pad_image(
                ref_image_list, 1, pad_width, odd_width_diff
                )
        elif ref_width > test_width:
            test_image_list = pad_image(
                test_image_list, 1, pad_width, odd_width_diff
                )

        concat_img_list = np.concatenate(
            (ref_image_list, test_image_list), axis=2
            )
        return concat_img_list

    @staticmethod
    def overlay_score_on_images(image_list, scores):
        """writes score on given image list

        Args:
          image_list: A list of numpy array representing video frames
          scores: A list of score between reference and test keypoints

        Returns:
          A list of numpy array with score overlayed on it

        """
        for i in range(len(image_list)):
            image = image_list[i, :, :, :]

            txt = f"Score : {scores[i]}"

            image = cv2.putText(
                image, txt, (5, 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 0, 0), 1, cv2.LINE_AA
                )

            image_list[i, :, :, :] = image

        return image_list

    def plot_matching(
        ref_frames, test_frames, ref_keypoints, test_keypoints, ref_frames_idx,
        test_frames_idx, costs, output_path
        ):
        """creates a video of reference and test video frames with keypoints
        overlayed on them

        Args:
          ref_frames: A list of numpy array representing reference video frames
          test_frames: A list of numpy array representing test video frames
          ref_keypoints: A list of numpy array representing reference video
          keypoints
          test_keypoints: A list of numpy array representing reference video
          keypoints
          ref_frames_idx: A list of reference frame indices
          test_frames_idx: A list of test frame indices
          costs: A list of score between reference and test keypoints
          output_path: path at which output video to be stored

        """
        if (ref_frames_idx is not None) and (
            test_frames_idx is not None) and len(ref_frames_idx) == len(
            test_frames_idx
            ):

            ref_frames = ref_frames[ref_frames_idx]
            test_frames = test_frames[test_frames_idx]

            ref_keypoints = ref_keypoints[ref_frames_idx]
            test_keypoints = test_keypoints[test_frames_idx]

        if costs is None:
            costs = ['N/A'] * len(ref_frames)

        display_line = [
            f"Cost: {costs[i]}     Ref frame: {ref_frames_idx[i]}    Test " \
            f"frame: {test_frames_idx[i]}"
            for i in range(len(ref_frames_idx))]

        ref_image_list = Plot.add_keypoints_to_image(
            ref_frames, ref_keypoints
            )
        test_image_list = Plot.add_keypoints_to_image(
            test_frames, test_keypoints
            )

        comparison_img_list = Plot.resize_and_concat(
            ref_image_list, test_image_list
            )

        comparison_img_list = Plot.overlay_score_on_images(
            comparison_img_list, display_line
            )

        video = ImageSequenceClip(list(comparison_img_list), fps=5)
        video.write_videofile(output_path, fps=5)
