import cv2
import numpy as np
import yolov5


class CropVideo:
    """Base class for cropping a video frame-by-frame using various object
    detection method such as YOLO or cv2.Tracker

    Warning: This class should not be used directly.
    Use derived classes instead.

      Parameters:
        method : name of the object detection method
        model_path : path to object detection model

    """

    def __init__(self, method=None):
        self.method = method

    def video_crop(self, video_frames):
        """Crops given list of frames by detecting object using different
        methods such as YOLO or cv2.Tracker.

        Args:
          video_frames: A list of numpy arrays representing the input images

        Returns:
          A numpy array containing cropped frames
        """
        raise NotImplementedError


class YOLOCrop(CropVideo):

    """Class for cropping a video frame-by-frame using YOLO object detection
    method


    Parameters :
        cropping_model_path : path to object detection model

    """

    def __init__(self, method=None, model_path=None):
        super().__init__('yolo')
        self.model_path = model_path or '../data/models/yolo/yolov5x.pt'
        self.load_model(self.model_path)

    def load_model(self, model_path):
        """Loads object detection model.
        """
        self.model = yolov5.load(model_path)
        self.model.classes = 0

    def get_yolo_bbox(self, frame):
        """Runs YOLO object detection on an input image.

        Args:
          frame: A [height, width, 3] numpy array representing the input image

        Returns:
          A list conating boundig box parameters [x_min, y_min, x_max, y_max]
        """

        results = self.model(frame)
        predictions = results.pred[0]

        boxes = predictions[:, :4].numpy().astype(np.int32)
        if len(boxes) == 0:
            return []
        elif len(boxes) == 1:
            return list(boxes[0])
        else:
            area = []
            for i in boxes:
                area.append(cv2.contourArea(np.array([[i[:2]], [i[2:]]])))
            largest_bbox = boxes[np.argmax(np.array(area))]
            return list(largest_bbox)

    def video_crop(self, video_frames):
        """Crops given list of frames by detecting object using YOLO

        Args:
          video_frames: A list of numpy arrays representing the input images

        Returns:
          A numpy array containing cropped frames
        """

        x_width_start = []
        y_height_start = []
        x_width_end = []
        y_height_end = []
        frame_height, frame_width = 0, 0

        widths = []
        heights = []
        for frame in video_frames:
            frame_height, frame_width, _ = frame.shape
            bbox = self.get_yolo_bbox(frame)

            if len(bbox) == 0:
                continue
            else:
                x_width_start.append(int(max(bbox[0] - 100, 0)))
                y_height_start.append(int(max(bbox[1] - 100, 0)))
                x_width_end.append(int(min(bbox[2] + 100, frame.shape[1])))
                y_height_end.append(int(min(bbox[3] + 100, frame.shape[0])))

                widths.append(x_width_end[-1] - x_width_start[-1])
                heights.append(y_height_end[-1] - y_height_start[-1])

        width = np.percentile(np.array(widths), 95)
        height = np.percentile(np.array(heights), 95)
        box_len = int(max(width, height))

        cropped_frames = []

        for i in range(len(widths)):
            frame = video_frames[i]
            xs = x_width_start[i]
            xe = x_width_start[i] + box_len
            ys = y_height_start[i]
            ye = y_height_start[i] + box_len

            if ye > frame_height:
                ye = frame_height
                ys = max(0, ye - box_len)

            if xe > frame_width:
                xe = frame_width
                xs = max(0, xe - box_len)

            cropped = frame[int(ys): int(ye), int(xs): int(xe), :]
            cropped_frames.append(np.array(cropped))

        return np.array(cropped_frames)


class TrackerCrop(YOLOCrop):
    def __init__(self, model_path=None):
        super().__init__(method='yolo')
        self.tracker = cv2.TrackerMIL.create()

    @staticmethod
    def expand_bbox(bbox, frame_shape):
        """Expands given bounding box by 50 pixels

        Args:
          bbox: A list [x,y, width, height] consits of bounding box
          parameters of
                object
          frame_shape: (height, width) of a frame

        """
        bbox[0] = max(bbox[0] - 50, 0)
        bbox[1] = max(bbox[1] - 50, 0)
        bbox[2] = min(bbox[3] + 50, frame_shape[1] - bbox[0] - 1)
        bbox[3] = min(bbox[3] + 50, frame_shape[0] - bbox[1] - 1)

    @staticmethod
    def pad_bbox(crop_frame, box_len):
        """Pads given cropped frame

        Args:
          crop_frame: A numpy array representing the cropped frame
          box_len: An integer value representing maximum out of width and height

        Returns:
          A numpy array containing cropped frame with padding
        """
        if box_len > crop_frame.shape[0] or box_len > crop_frame.shape[1]:
            crop_frame = np.pad(
                crop_frame, pad_width=(
                    (0, box_len - crop_frame.shape[0]),
                    (0, box_len - crop_frame.shape[1]), (0, 0))
                )
        return crop_frame

    @staticmethod
    def clip_coordinates(x, y, box_len, frame_shape):
        """Clips (x,y) coordinates representing the centre of bounding box

        Args:
          x: x-coordinate of the centre of bounding box
          y: y-coordinate of the centre of bounding box
          box_len: An integer value representing maximum out of width and height
          frame_shape: (height, width) of a frame

        Returns:
          (x,y) clipped coordinates
        """
        if x + box_len > frame_shape[1]:
            diff = x + box_len - frame_shape[1]
            x = max(0, x - diff)
        if y + box_len > frame_shape[0]:
            diff = y + box_len - frame_shape[0]
            y = max(0, y - diff)

        return (x, y)

    def video_crop(self, video_frames):
        """Crops given list of frames by detecting object using cv2.Tracker

        Args:
          video_frames: A list of numpy arrays representing the input images

        Returns:
          A numpy array containing cropped frames
        """

        frame = video_frames[0]
        bbox = self.get_yolo_bbox(frame)
        TrackerCrop.expand_bbox(bbox, frame.shape)
        self.tracker.init(frame, bbox)
        output_frame_list = []
        for frame in video_frames:
            _, bbox = self.tracker.update(frame)
            x, y, w, h = bbox
            box_len = max(w, h)
            x, y = TrackerCrop.clip_coordinates(x, y, box_len, frame.shape)
            crop_frame = np.array(frame[y:y + box_len, x:x + box_len, :])
            crop_frame = TrackerCrop.pad_bbox(crop_frame, box_len)
            output_frame_list.append(crop_frame)

        output_frame_array = np.array(output_frame_list)

        return output_frame_array
