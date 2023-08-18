import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
from src.correspondence import DTW
from src.crop_video import YOLOCrop, TrackerCrop
from src.pose_detection import MovenetPoseDetection
from src import utils


def main(ref_video_path, test_video_path, output_video_path, crop_method="yolo"):

  ref_frames = utils.get_video_frames(ref_video_path)
  test_frames = utils.get_video_frames(test_video_path)

  if crop_method == 'Tracker':
    crop_object = TrackerCrop()
  else:
    crop_object = YOLOCrop()

  ref_crop_frames = crop_object.video_crop(ref_frames)
  test_crop_frames = crop_object.video_crop(test_frames)

  movenet = MovenetPoseDetection()
  ref_keypoints = movenet.run_inference(ref_crop_frames)
  test_keypoints = movenet.run_inference(test_crop_frames)

  dtw = DTW(cost_weightage={'mae' : 0, 'angle_mae' : 1})
  ref_frame_idx, test_frame_idx, costs = dtw.find_correspondence(ref_keypoints, test_keypoints)

  utils.Plot.plot_matching(ref_crop_frames, test_crop_frames, ref_keypoints, test_keypoints,ref_frame_idx, test_frame_idx, costs,output_video_path)

  return output_video_path


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description="run chained process",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
  parser.add_argument('--Ref_video', type=str, required=True)
  parser.add_argument('--Test_video', type=str, required=True)
  parser.add_argument('--Output_path', type=str, required=True)

  args = parser.parse_args()

  try:
    main(args.Ref_video, args.Test_video, args.Output_path)
  except NameError:
    print("Video file is not appropriate.")
  except ValueError:
    print("YOLO couldn't detect bounding box for given video.")
  except cv2.error:
    print(
      "Can not convert color from BGR to RGB. Please check the input frame."
      )
