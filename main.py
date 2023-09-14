import os
import cv2
import gradio as gr
from src import run_chain


def video_process(ref_video, test_video, crop_method):

    run_chain.main(
        ref_video,
        test_video,
        'output_video.mp4',
        crop_method=crop_method
        )

    return 'output_video.mp4'


demo = gr.Interface(video_process,
                    inputs = [gr.Video(label='Reference Video'), gr.Video(label='Test Video'), gr.Radio(["YOLO", "Tracker"], label="Crop Method")],
                    outputs = [gr.PlayableVideo()]
                    )

if __name__ == "__main__":
    demo.launch()

