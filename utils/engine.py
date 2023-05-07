import cv2
import math
import json
import argparse
import numpy as np
from os import path
from tqdm import trange
from ultralytics import YOLO
from collections import deque
from matplotlib import pyplot as plt

from utils.deepsort import DeepSort
from utils.frame import Frame

DIAGRAM_PATH = "diagram.jpg"

class Engine():
    def __init__(self, mode="cli", config_file="config.json", **override_args):
        # Load config.json as attributes
        self.load_config(config_file)

        if mode == "cli":
            # Initialize parser
            self.init_parser()

            # Parse arguments
            args = vars(self.parser.parse_args())

            # Override config.json with args from command line
            self.override_config(args)

        # Override config.json with args from override_args
        if override_args is not None:
            self.override_config(override_args)

        # Fix paths
        self.source, self.save_path = path.abspath(
            self.source), path.abspath(self.save_path)

        # Load yolo models
        # For detecting players and referee
        self.yolov8 = YOLO(self.yolo_weights)
        # For instance segmentation on free throw area
        self.yolov8_seg = YOLO(self.yolo_seg_weights)
  
        # Load diagram
        self.diagram = cv2.imread(DIAGRAM_PATH)
  
        # To store the kmeans model
        self.kmeans = None
        self.prev_players = []

    def load_config(self, path):
        # Load config.json
        with open(path) as f:
            config = json.load(f)

        # Set attributes
        for key, value in config.items():
            setattr(self, key, value)

    def override_config(self, args):
        # Override config.json with override_config
        for key, value in args.items():
            setattr(self, key, value)

    def init_parser(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument(
            "--source",
            type=str,
            default=self.source,
            help="Path to basketball video"
        )
        self.parser.add_argument(
            "--save_path",
            type=str,
            default=self.save_path,
            help="Path to save the output video"
        )
        self.parser.add_argument(
            "--eval_court",
            action="store_true",
            default=self.eval_court,
            help="Run the court evaluation (frame must not contain empty detection)"
        )
        self.parser.add_argument(
            "--save",
            action="store_true",
            default=self.save,
            help="Save the output video"
        )
        self.parser.add_argument(
            "--save_diagram",
            action="store_true",
            default=self.save_diagram,
            help="Save the bird's eye view diagram"
        )
        self.parser.add_argument(
            "--show",
            action="store_true",
            default=self.show,
            help="Show the intermediate frames (for debugging)"
        )
        self.parser.add_argument(
            "--nframes",
            type=int,
            default=self.nframes,
            help="Number of frames to process (for debugging)"
        )
        self.parser.add_argument(
            "--skip_frames",
            type=int,
            default=self.skip_frames,
            help="Number of frames to skip (for debugging)"
        )

        # Hyperparameters #TODO: Review the help strings
        self.parser.add_argument(
            "--max_cos_dist",
            type=float,
            default=self.max_cos_dist,
            help="Maximum cosine distance for a track to be considered a match"
        )
        self.parser.add_argument(
            "--nn_budget",
            type=int,
            default=self.nn_budget,
            help="Maximum number of tracks to keep in the tracker")
        self.parser.add_argument(
            "--iou_th",
            type=float,
            default=self.iou_th,
            help="IOU threshold for non-max suppression"
        )
        self.parser.add_argument(
            "--conf_th",
            type=float,
            default=self.conf_th,
            help="Confidence threshold for non-max suppression"
        )
        self.parser.add_argument(
            "--max_age",
            type=int,
            default=self.max_age,
            help="Maximum number of frames to keep a track alive"
        )
        self.parser.add_argument(
            "--init_phase_frames",
            type=int,
            default=self.init_phase_frames,
            help="Number of frames a track is initialised for"
        )
        self.parser.add_argument(
            "--deepsort_iou_th",
            type=float,
            default=self.deepsort_iou_th,
            help="IOU threshold for DeepSort"
        )
  
    def override_and_load_yolo(self, override_args):
        if override_args is not None:
            yolo_prev = self.yolo_weights
            yolo_seg_prev = self.yolo_seg_weights

            self.override_config(override_args)

            if yolo_prev != self.yolo_weights:
                self.yolov8 = YOLO(self.yolo_weights)

            if yolo_seg_prev != self.yolo_seg_weights:
                self.yolov8_seg = YOLO(self.yolo_seg_weights)

    def detect_image(self, **override_args):
        self.override_and_load_yolo(override_args)

        # Assert that the image is not a video
        assert not self.source.endswith(
            ".mp4"), "Please use track() for videos or use a different source"
        assert not self.save_path.endswith(
            ".mp4"), "Please use track() for videos or use a different save_path"

        # Get the image from the source
        img = cv2.imread(self.source)

        # Extract the image dimensions
        self.width, self.height = img.shape[1], img.shape[0]

        frame = Frame(img, 0, self)

        # Get corners of free throw area if it exists
        frame.getCorners()

        # Use yolov8 to detect players and referee
        detection = frame.detect()

        # Detect players and referee
        frame.detect_image()

        # Write the frame to the output path
        return frame.save_img(self.save_path)

    def track(self, **override_args):
        self.override_and_load_yolo(override_args)

        # Initialize DeepSORT trackers for each players and referee
        self.player_tracker = DeepSort(
            self.deepsort_weights,
            self.max_cos_dist,
            self.deepsort_iou_th,
            self.nn_budget,
            self.max_age,
            self.init_phase_frames
        )
        self.referee_tracker = DeepSort(
            self.deepsort_weights,
            self.max_cos_dist,
            self.deepsort_iou_th,
            self.nn_budget,
            self.max_age,
            self.init_phase_frames
        )

        # Get the video from the source
        video = cv2.VideoCapture(self.source)

        # Extract the video w, h and fps
        self.width, self.height = int(
            video.get(
                cv2.CAP_PROP_FRAME_WIDTH)), int(
            video.get(
                cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))

        print(f"FPS: {fps}")

        if self.save:
            # Define the codec and create VideoWriter object
            codec = cv2.VideoWriter_fourcc(*"avc1")
            output = cv2.VideoWriter(self.save_path, codec, fps, (self.width, self.height))
   
        if self.save_diagram:
            # Get w and h of diagram
            diagram = cv2.imread(DIAGRAM_PATH)
      
            # Define the codec and create VideoWriter object
            self.save_diagram_path = self.save_path.replace(
                ".mp4", "_diagram.mp4")
            output_diagram = cv2.VideoWriter(self.save_diagram_path, codec, fps, diagram.shape[1::-1])

        if self.eval_court:
            # Initilise variable to store differences between corners
            self.court_errors = []

        # Get first frame from video by using skip_frames
        for _ in range(self.skip_frames + 1):
            ret, orig = video.read()

            # Check if video has ended
            if not ret:
                break

        # Queue for storing previous frames for optical flow
        queue = deque()
        prev_frame = None

        # Get total number of frames
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)
                           ) if self.nframes is None else self.nframes
        total_frames -= self.skip_frames if total_frames > self.skip_frames else 0

        # Loop through all frames
        for i in trange(total_frames, desc="Processing...", unit="frames"):
            # Check if video has ended
            if not ret:
                break

            frame = Frame(orig, i, self)

            # Get next frame from video
            ret, orig = video.read()

            # Get corners of free throw area if it exists
            frame.getCorners()
   
            # Update deepsort trackers for players and referee + plot tracks
            frame.track()

            # If mask is None then there is no free throw area so append frame
            # to queue
            if frame.index == total_frames - 1:
                queue.append(frame)
            elif len(frame.corners) != 4:
                queue.append(frame)
                continue

            # Frame has corners
            if prev_frame is None and len(frame.corners) == 4:
                # Only do backward optical flow from current frame to all qeued
                # frames
                next_frame = frame
                for i in range(len(queue) - 1, -1, -1):
                    qframe = queue[i]
                    Frame.updateCorners(qframe, next_frame)
                    next_frame = qframe
            elif len(frame.corners) != 4:
                # Only do forward optical flow from previous frame to current
                # frame
                queue.appendleft(prev_frame)
                for i in range(1, len(queue)):
                    qframe = queue[i]
                    Frame.updateCorners(qframe, prev_frame)
                    prev_frame = qframe
            elif len(queue) > 0:
                # If both frames have corners then do half forward and half
                # backward optical flow
                queue.appendleft(prev_frame)
                for i in range(1, len(queue) // 2):
                    qframe = queue[i]
                    Frame.updateCorners(qframe, prev_frame)
                    prev_frame = qframe

                # For the rest do backward optical flow
                next_frame = frame
                for i in range(len(queue) - 1, len(queue) // 2 - 1, -1):
                    qframe = queue[i]
                    Frame.updateCorners(qframe, next_frame)
                    next_frame = qframe
            else:
                queue.appendleft(prev_frame)

            while len(queue) > 0:
                f = queue.popleft()
                

                if self.save or self.show:
                    f.saveFrame(output=output)

                    if self.save_diagram:
                        output_diagram.write(f.diagram)
      
                if self.eval_court and prev_frame.index > self.init_phase_frames:
                    eval_corners = prev_frame.corners
                    prev_frame.corners = []
                    Frame.updateCorners(prev_frame, frame)
     
                    # Compare the distance between detected corners and the optical flow corners
                    for i in range(4):
                        xerror = (eval_corners[i][0] - prev_frame.corners[i][0])**2
                        yerror = (eval_corners[i][1] - prev_frame.corners[i][1])**2
                        self.court_errors.append(xerror)
                        self.court_errors.append(yerror)
      
                # Clear some memory
                f.clearMemory()

            prev_frame = frame
   
        if self.eval_court:
            self.court_errors = self.court_errors[:-1] # Remove last error as it is always 0
            print(f"Average court error: {sum(self.court_errors)/len(self.court_errors):.6f}")
            print(f"Standard deviation of court error: {np.std(self.court_errors):.6f}")
   
            # Combine every 4 errors into one
            self.court_errors = [sum(self.court_errors[i:i+8])/8 for i in range(0, len(self.court_errors), 8)]
   
            # Plot a graph of the cumulative error
            self.court_errors = np.cumsum(self.court_errors)
            self.court_errors = [x / i for i, x in enumerate(self.court_errors, 1)]
            # plt.plot(self.court_errors)
            # plt.xlabel("Frame number")
            # plt.ylabel("Error (px)")
            # plt.title("Error between detected corners and optical flow corners")
            # plt.show()

        cv2.destroyAllWindows()
        video.release()  # Release video
