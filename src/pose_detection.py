import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os


class PoseDetection:
    """Base class for pose detection in images using various algorithm such
    as Movenet

    Warning: This class should not be used directly.
    Use derived classes instead.

      Parameters:
        model_name : name of the pose detection method
        input_size : image size of input required for model
        model_path : path to pose detection model

    """

    def __init__(self, model_name=None, input_size=None, model_path=None):
        self.model_name = model_name
        self.input_size = input_size
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """Absrtact method to loads pose detection model.
        """
        raise NotImplementedError

    def preprocess_image(self, frame):
        """Absrtact method to preprocess a image before running pose
        detection inference on it.
        """
        raise NotImplementedError

    def run_inference(self, frames):
        """Absrtact method to run pose detection inference on it.
        """
        raise NotImplementedError


class MovenetPoseDetection(PoseDetection):
    """Class for pose detection in images using Movenet model

      Parameters:
        model_name : name of the pose detection method
        input_size : image size of input required for model
        model_path : path to pose detection model

    """

    def __init__(self, model_name=None, input_size=None, model_path=None):
        model_name = model_name or 'movenet'
        input_size = input_size or 256
        model_path = model_path or 'models/movenet/movenet_tf/1'
        super().__init__(model_name, input_size, model_path)

    def load_model(self):
        """Loads the pose detection model.

        """
        if self.model_name == "movenet":
            module = hub.load(self.model_path)
            self.model = module.signatures['serving_default']

    def preprocess_image(self, frame):
        """Preprocesses an image to transform it into required format for
        pose detection

        Args:
          frame: A numpy array representing the input image

        Returns:
          A tensor of 'int32' data type and resized input image.
        """
        input_image = tf.expand_dims(frame, axis=0)
        input_image = tf.image.resize_with_pad(
            input_image, self.input_size, self.input_size
            )
        input_image = tf.cast(input_image, dtype=tf.int32)

        return input_image

    def run_inference(self, frames):
        """Appllies pose dection model on a frame

        Args:
          frames: A list of numpy arrays representing the input images

        Returns:
          A [n, 1, 17, 3] float numpy array representing the keypoint
          coordinates
          and scores predicted by Movenet Model of list of images
        """
        keypoints = np.zeros((len(frames), 17, 2))
        for i, frame in enumerate(frames):
            frame = self.preprocess_image(frame)
            keypoints[i, :, :] = self.run_model(frame)
        return keypoints

    def run_model(self, input_image):
        """Runs detection on an input image.

        Args:
          input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and
            match the
            expected input resolution of the model before passing into this
            function.

        Returns:
          A [1, 1, 17, 3] float numpy array representing the keypoint
          coordinates
          and scores predicted by Movenet Model
        """

        outputs = self.model(input_image)
        outputs = outputs['output_0'].numpy()
        outputs[:, :, :, [0, 1]] = outputs[:, :, :, [1, 0]]

        return outputs[0, 0, :, :2]
