import argparse
from os import path
from time import perf_counter
from tqdm import trange

import cv2
import numpy as np
from collections import deque

from tools.deepsort import DeepSort
from tools.frame import Frame, updateCorners, saveFrame

from ultralytics import YOLO
from ultralytics.tracker.trackers.bot_sort import BOTSORT
from ultralytics.yolo.utils import IterableSimpleNamespace

BOTSORT_CFG = {
    'tracker_type': 'botsort',
    'track_high_thresh': 0.5,
    'track_low_thresh': 0.1,
    'new_track_thresh': 0.7,
    'track_buffer': 60,
    'match_thresh': 1,
    'cmc_method': 'sparseOptFlow',
    'proximity_thresh': 0.5,
    'appearance_thresh': 0.25,
    'with_reid': False
}

def run(opt):
	# Debugging Arguments
	show_process, nframes, skip_frames = opt.show_process, opt.nframes, opt.skip_frames
	
	# Get arguments
	source, save_path, yolo_weights, yolo_seg_weights, deepsort_weights, device = path.abspath(opt.source), path.abspath(opt.save_path), opt.yolo_weights, opt.yolo_seg_weights, opt.deepsort_weights, opt.device
	
	# Get hyperparameters
	max_cos_dist, nn_budget, iou_th, conf_th, iou_th_seg, conf_th_seg, max_track_age, init_phase_frames = opt.max_cos_dist, opt.nn_budget, opt.iou_th, opt.conf_th, opt.iou_th_seg, opt.conf_th_seg, opt.max_track_age, opt.init_phase_frames

	# Initialise YOLO wrapper for player and referee detection
	yolov8 = YOLO(yolo_weights)

	# Initialize BOTSORT tracker
	cfg = IterableSimpleNamespace(**BOTSORT_CFG)
	tracker = BOTSORT(
    	args = cfg,
     	frame_rate = 30
    )

	# Initialise YoloV7 Instance Segmentation wrapper for court detection
	yolov8_seg = YOLO(yolo_seg_weights)

	# Get the video from the source
	video = cv2.VideoCapture(source)
	
	# Extract the video w, h and fps
	w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(video.get(cv2.CAP_PROP_FPS))
 
	# Define the codec and create VideoWriter object
	codec = cv2.VideoWriter_fourcc(*"avc1")
	output = cv2.VideoWriter(save_path, codec, fps, (w, h))

	# Initialise variables for tracking
	total_time = 0
 
	# Queue for storing previous frames for optical flow
	queue = deque()
 
	# Get first frame from video by using skip_frames
	for _ in range(skip_frames+1):
		ret, orig = video.read()
  
		# Check if video has ended
		if not ret:
			break

	# Get total number of frames
	total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) if nframes is None else nframes
 
	total_frames -= skip_frames if total_frames > skip_frames else 0

	prev_frame = None
 
	# Read diagram image
	diagram_empty = cv2.imread('diagram.jpg')

	# Loop through all frames
	for i in trange(total_frames, desc="Processing..."):
		# Start timer for current frame
		start = perf_counter()
  
		# Check if video has ended
		if not ret:
			break

		frame = Frame(orig, i, w, h)
  
		# Get next frame from video
		ret, orig = video.read()
  
		# Run court segmentation to find free throw area
		frame.detectMask(yolov8_seg)
  
		# Run player and referee detection + tracking (yolov7 + deepsort)
		frame.detectAndTrack(yolov8, tracker)
					
		# Update the time taken and increment frame count
		total_time += perf_counter() - start
		i += 1
  
		# If mask is None then there is no free throw area so append frame to queue
		if frame.n == total_frames - 1:
			queue.append(frame)
		elif frame.mask is None:
			queue.append(frame)
			continue
		
  		# Frame has corners
		if prev_frame is None and frame.corners is not None:
			# Only do backward optical flow from current frame to all qeued frames
			next_frame = frame
			for i in range(len(queue) - 1, -1, -1):
				qframe = queue[i]
				updateCorners(qframe, next_frame)
				next_frame = qframe
		elif frame.corners is None:
			# Only do forward optical flow from previous frame to current frame
			queue.appendleft(prev_frame)
			for i in range(1, len(queue)):
				qframe = queue[i]
				updateCorners(qframe, prev_frame)
				prev_frame = qframe
		elif len(queue) > 0:
			# If both frames have corners then do half forward and half backward optical flow
			queue.appendleft(prev_frame)
			for i in range(1, len(queue)//2):
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
			saveFrame(
				diagram = diagram_empty.copy(),
				frame = queue.popleft(),
				output = output,
				debug = show_process,
			)

		prev_frame = frame
   
	print(f"\nFPS: {i/total_time:.2f} Hz")
	cv2.destroyAllWindows()
	video.release() # Release video

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", type=str, default="./data/input.mp4", help="Path to a basketball video")
	parser.add_argument("--save_path", type=str, default="./data/output.mp4", help="Path to save the output video to")
	parser.add_argument("--yolo_weights", type=str, default="./weights/NinthYolo_x-nracheckpoint3.pt", help="Path to the YOLO weights file (.pt)")
	parser.add_argument("--yolo_seg_weights", type=str, default="./weights/SecondSeg.pt", help="Path to the YOLO segmentation weights file (.pt)")
	parser.add_argument("--deepsort_weights", type=str, default="./weights/market1501.pb", help="Path to the DeepSORT weights file (.pb)")
	parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument("--show_process", action="store_true", help="Show the intermediate frames (for debugging)")
	parser.add_argument("--nframes", type=int, default=None, help="Number of frames to process (for debugging)")
	parser.add_argument("--skip_frames", type=int, default=0, help="Number of frames to skip (for debugging)")
 
	# Hyperparameters #TODO: Review the help strings
	parser.add_argument("--max_cos_dist", type=float, default=0.4, help="Maximum cosine distance for a track to be considered a match")
	parser.add_argument("--nn_budget", type=int, default=None, help="Maximum number of tracks to keep in the tracker")
	parser.add_argument("--iou_th", type=float, default=0.2, help="IOU threshold for non-max suppression")
	parser.add_argument("--conf_th", type=float, default=0.2, help="Confidence threshold for non-max suppression")
	parser.add_argument("--max_track_age", type=int, default=300, help="Maximum number of frames to keep a track alive")
	parser.add_argument("--init_phase_frames", type=int, default=6, help="Number of frames a track is initialised for")
 
	# YOLOv7 Segmentation Hyperparameters
	parser.add_argument("--conf_th_seg", type=float, default=0.5, help="Confidence threshold for segmentation")
	parser.add_argument("--iou_th_seg", type=float, default=0.5, help="IOU threshold for segmentation")
	
	opt = parser.parse_args()
	print(opt)
	
	# Store start time
	start = perf_counter()
	
	# Run tracking
	run(opt)
	
	# Print time taken to run
	print(f"Time taken: {perf_counter()-start:.2f} s")
 
