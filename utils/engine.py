import cv2
import json
import argparse
import numpy as np
from os import path
from tqdm import trange
from collections import deque

from utils.deepsort import DeepSort
from ultralytics import YOLO

from utils.frame import Frame, updateCorners


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
			"--max_track_age",
			type=int,
			default=self.max_track_age,
			help="Maximum number of frames to keep a track alive"
		)
		self.parser.add_argument(
			"--init_phase_frames",
			type=int,
			default=self.init_phase_frames,
			help="Number of frames a track is initialised for"
		)

		# YOLOv7 Segmentation Hyperparameters
		self.parser.add_argument(
			"--conf_th_seg",
			type=float,
			default=self.conf_th_seg,
			help="Confidence threshold for segmentation"
		)
		self.parser.add_argument(
			"--iou_th_seg",
			type=float,
			default=self.iou_th_seg,
			help="IOU threshold for segmentation"
		)

	def detect_image(self, **override_args):
		# Override config.json with args from command line
		if override_args is not None:
			yolo_prev = self.yolo_weights
			yolo_seg_prev = self.yolo_seg_weights

			self.override_config(override_args)

			if yolo_prev != self.yolo_weights:
				self.yolov8 = YOLO(self.yolo_weights)

			if yolo_seg_prev != self.yolo_seg_weights:
				self.yolov8_seg = YOLO(self.yolo_seg_weights)

		# Assert that the image is not a video
		assert not self.source.endswith(
			".mp4"), "Please use track() for videos or use a different source"
		assert not self.save_path.endswith(
			".mp4"), "Please use track() for videos or use a different save_path"

		# Get the image from the source
		img = cv2.imread(self.source)

		# Extract the image dimensions
		self.w, self.h = img.shape[1], img.shape[0]

		diagram = cv2.imread("diagram.png")

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
		# Override config.json with args from command line
		if override_args is not None:
			yolo_prev = self.yolo_weights
			yolo_seg_prev = self.yolo_seg_weights

			self.override_config(override_args)

			if yolo_prev != self.yolo_weights:
				self.yolov8 = YOLO(self.yolo_weights)

			if yolo_seg_prev != self.yolo_seg_weights:
				self.yolov8_seg = YOLO(self.yolo_seg_weights)

		# Initialize DeepSORT trackers for each players and referee
		self.player_tracker = DeepSort(
			self.deepsort_weights,
			self.max_cos_dist,
			self.nn_budget,
			self.max_track_age,
			self.init_phase_frames
		)
		self.referee_tracker = DeepSort(
			self.deepsort_weights,
			self.max_cos_dist,
			self.nn_budget,
			self.max_track_age,
			self.init_phase_frames
		)

		# Read diagram image
		diagram_empty = cv2.imread('diagram.jpg')

		# Get the video from the source
		video = cv2.VideoCapture(self.source)

		# Extract the video w, h and fps
		self.w, self.h = int(
			video.get(
				cv2.CAP_PROP_FRAME_WIDTH)), int(
			video.get(
				cv2.CAP_PROP_FRAME_HEIGHT))
		fps = int(video.get(cv2.CAP_PROP_FPS))

		print(f"FPS: {fps}")

		if self.save:
			# Define the codec and create VideoWriter object
			codec = cv2.VideoWriter_fourcc(*"avc1")
			output = cv2.VideoWriter(self.save_path, codec, fps, (self.w, self.h))

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

			# Use yolov8 to detect players and referee
			detection = frame.detect()
   
			# Update deepsort trackers for players and referee + plot tracks
			frame.track(detection)

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
					updateCorners(qframe, next_frame)
					next_frame = qframe
			elif len(frame.corners) != 4:
				# Only do forward optical flow from previous frame to current
				# frame
				queue.appendleft(prev_frame)
				for i in range(1, len(queue)):
					qframe = queue[i]
					updateCorners(qframe, prev_frame)
					prev_frame = qframe
			elif len(queue) > 0:
				# If both frames have corners then do half forward and half
				# backward optical flow
				queue.appendleft(prev_frame)
				for i in range(1, len(queue) // 2):
					qframe = queue[i]
					updateCorners(qframe, prev_frame)
					prev_frame = qframe

				# For the rest do backward optical flow
				next_frame = frame
				for i in range(len(queue) - 1, len(queue) // 2 - 1, -1):
					qframe = queue[i]
					updateCorners(qframe, next_frame)
					next_frame = qframe
			else:
				queue.appendleft(prev_frame)

			while len(queue) > 0:
				f = queue.popleft()

				if self.save or self.show:
					f.saveFrame(
						diagram=diagram_empty.copy(),
						output=output,
					)

			prev_frame = frame

		cv2.destroyAllWindows()
		video.release()  # Release video
