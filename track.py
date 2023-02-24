import argparse
from os import path
from time import perf_counter
from tqdm import trange

import cv2
import numpy as np
from collections import deque

from tools.deepsort import DeepSortWrapper
from tools.yolov7 import Yolov7Wrapper
from tools.yolov7_seg import Yolov7SegWrapper
from tools.frame import Frame, updateCorners, saveFrame

def run(opt):
    # Debugging Arguments
	show_process, nframes, skip_frames = opt.show_process, opt.nframes, opt.skip_frames
    
	# Get arguments
	source, save_path, yolo_weights, yolo_seg_weights, deepsort_weights, device = path.abspath(opt.source), path.abspath(opt.save_path), opt.yolo_weights, opt.yolo_seg_weights, opt.deepsort_weights, opt.device
	
	# Get hyperparameters
	max_cos_dist, nn_budget, iou_th, conf_th, iou_th_seg, conf_th_seg, max_track_age, init_phase_frames = opt.max_cos_dist, opt.nn_budget, opt.iou_th, opt.conf_th, opt.iou_th_seg, opt.conf_th_seg, opt.max_track_age, opt.init_phase_frames

	# Initialise YOLOv7 wrapper for player and referee detection
	yolov7 = Yolov7Wrapper(device, yolo_weights, iou_th=iou_th, conf_th=conf_th)

	# initialize DeepSORT trackers for each players and referee
	player_tracker = DeepSortWrapper(deepsort_weights, max_cos_dist, nn_budget, max_track_age, init_phase_frames, class_name="player")
	referee_tracker = DeepSortWrapper(deepsort_weights, max_cos_dist, nn_budget, max_track_age, init_phase_frames, class_name="referee")

	# Initialise YoloV7 Instance Segmentation wrapper for court detection
	yolov7_seg = Yolov7SegWrapper(device, yolo_seg_weights, iou_th=iou_th_seg, conf_th=conf_th_seg)

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
 
	# Get total number of frames
	total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) if nframes is None else nframes
 
	# Get first frame from video by using skip_frames
	for _ in range(skip_frames+1):
		ret, first = video.read()
  
		# Check if video has ended
		if not ret:
			total_frames = 0
			break
	
	frame = Frame(first, None, w, h)
 
	# Convert frame to RGB
	frame.img = cv2.cvtColor(frame.img, cv2.COLOR_BGR2RGB)
 
	# Read diagram image
	diagram_empty = cv2.imread('diagram.jpg')

	# Loop through all frames
	for i in trange(total_frames, desc="1/3 - YOLOv7 + DeepSORT + YOLACT + Forward Optical Flow"):
		# Start timer for current frame
		start = perf_counter()

		# Get next frame from video
		ret, orig = video.read()
  
		# Check if video has ended
		if not ret:
			total_frames = i
			break

		frame = Frame(orig, i, w, h)
    
		# Convert frame to RGB
		frame.img = cv2.cvtColor(frame.img, cv2.COLOR_BGR2RGB)
  
		# Run court segmentation to find free throw area
		frame.detectMask(yolov7_seg)
  
		# Run player and referee detection + tracking (yolov7 + deepsort)
		frame.detectAndTrack(yolov7, [player_tracker, referee_tracker])
					
		# Update the time taken and increment frame count
		total_time += perf_counter() - start
		i += 1
  
		queue.append(frame)
  
		# If mask is None then there is no free throw area so append frame to queue
		if frame.mask is None and i != total_frames - 1:
			continue
  
		backwardsQueue = []

		if queue[-1].corners is not None and queue[0].corners is not None:
			backwards = True
			forwards = True
			loops = (len(queue) - 2) // 2
		elif queue[-1].corners is not None:
			backwards = True
			forwards = False
			loops = (len(queue) - 1)
		elif queue[0].corners is not None:
			backwards = False
			forwards = True
			loops = (len(queue) - 1)
		else:
			# throw error
			print("Error: No corners found")
   
		# Update corners using forward & Backwards optical flow for all frames in queue
		for _ in range(loops):
			if forwards:
				updateCorners(queue[1], queue[0])
				saveFrame(
					diagram = diagram_empty.copy(),
					frame = queue.popleft(),
					output = output,
					player_tracker = player_tracker,
					referee_tracker = referee_tracker,
					debug = show_process
				)
    
			if backwards:
				updateCorners(queue[-2], queue[-1])
				backwardsQueue.append(queue.pop())
   
		# If there is an odd number of frames in the queue then update the middle frame
		if len(queue) > 2 and len(queue) % 2 == 1:
			updateCorners(queue[1], queue[0])
			saveFrame(
				diagram = diagram_empty.copy(),
				frame = queue.popleft(),
				output = output,
				player_tracker = player_tracker,
				referee_tracker = referee_tracker,
				debug = show_process
			)
  
		if len(queue) >= 1 and backwards:
			while queue:
				backwardsQueue.append(queue.pop())
		elif len(queue) == 2:
			backwardsQueue.append(queue.popleft())
   
		# Run through backwards to save frames in correct order
		for _ in range(len(backwardsQueue)):
			saveFrame(
				diagram = diagram_empty.copy(),
				frame = backwardsQueue.pop(),
				output = output,
				player_tracker = player_tracker,
				referee_tracker = referee_tracker,
				debug = show_process
			)
   
	print(f"\nFPS: {i/total_time:.2f} Hz")
	cv2.destroyAllWindows()
	video.release() # Release video


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", type=str, default="./data/input.mp4", help="Path to a basketball video")
	parser.add_argument("--save_path", type=str, default="./data/output.mp4", help="Path to save the output video to")
	parser.add_argument("--yolo_weights", type=str, default="./weights/cleanedDatasetWeights.pt", help="Path to the YOLO weights file (.pt)")
	parser.add_argument("--yolo_seg_weights", type=str, default="./weights/freethrow.pt", help="Path to the YOLO segmentation weights file (.pt)")
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