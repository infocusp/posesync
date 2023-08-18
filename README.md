## State-of-the-art Video Matching tool is here!! 

We present a tool to synchronize videos of people performing similar actions. This can be useful for applications like - 
* Comparing game play of two players
* Side by side view of dance moves
* Comparison of any moves that require side by side video

This open source tool allows you to compare any two videos by bringing them in sync using the state of the art models at its backend for performing pose estimation and matching the poses.


It consists of three stages:
* Cropping (Using YOLO v5 / tracker)
* Pose Detection (Movenet)
* Video Matching (Dynamic Time Warping)
    
### 1. Cropping
First, video is cropped frame by frame with help of object detection which is supported by two methods, YOLO and Tracker by OpenCV.

### 2. Pose Detection
Then, MoveNet Pose Detection is applied on cropped video which returns the keypoints.

### 3. Video Matching
Finally, DTW (Dynamic Time Warping) processes the sequences of pose detected from two videos.


## How to use posesync

#### 1. Clone the repo into posesync directory
```python
git clone https://github.com/InFoCusp/posesync.git
```

#### 2. Install all the requirements
```python
pip install -r requirements.txt
```

#### 3. Run the *posesync*
```python
python3 main.py
```

## Demo

![](https://github.com/InFoCusp/posesync/blob/main/data/PoseSync_Demo.gif)
