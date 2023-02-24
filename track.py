import argparse
from os import path
from time import perf_counter
from tqdm import trange

import cv2
import numpy as np

from tools.deepsort import DeepSortWrapper
from tools.yolov7 import Yolov7Wrapper
from tools.yolov7_seg import Yolov7SegWrapper
from tools.frame import Frame

def run(opt):
    # Debugging Arguments
	show_process, nframes = opt.show_process, opt.nframes
    
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
 
	# Get total number of frames
	total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) if nframes is None else nframes
 
	# Get first frame from video
	_, first = video.read()

	frame = Frame(first, None, w, h)
 
	# Convert frame to RGB
	frame.img = cv2.cvtColor(frame.img, cv2.COLOR_BGR2RGB)

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

		frame = Frame(orig, frame, w, h)
    
		# Convert frame to RGB
		frame.img = cv2.cvtColor(frame.img, cv2.COLOR_BGR2RGB)
  
		# Run court segmentation to find free throw area
		frame.detectMask(yolov7_seg)
  
		# Run player and referee detection + tracking (yolov7 + deepsort)
		frame.detectAndTrack(yolov7, [player_tracker, referee_tracker])
  
		# Run forward optical flow to find player and referee positions
		if frame.mask is None:
			frame.updateCorners()
		
		# Draw white square in top left corner for text
		cv2.rectangle(frame.img, (0, 0), (185, 100), (255, 255, 255), -1)
		
		# Add number of objects to display frame, split into players and referees
		cv2.putText(frame.img, f"Players: {len(player_tracker.tracker.tracks)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
		cv2.putText(frame.img, f"Referees: {len(referee_tracker.tracker.tracks)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
		
		# Display frame and debug frames
		if show_process:
			frame.displayAll(debug = True)
  
		# Save the next frame for the next iteration
		frame.prev_frame.next_frame = frame
					
		# Update the time taken and increment frame count
		total_time += perf_counter() - start
		i += 1
  
	# Loop back through frames to add the player and referee positions to the diagram
	for _ in trange(total_frames, desc="2/3 - Backwards Optical Flow + Homography"):
		# Update the diagram with the player and referee positions
		frame.updateCornersBackwards()
		frame.updateDiagram()
  
		# Display frame and debug frames
		if show_process:
			frame.displayAll(debug = True)
   
		# Move to the previous frame
		frame = frame.prev_frame
  
	# Write the final diagram to the output video
	for _ in trange(total_frames, desc="3/3 - Writing Video"):
		# Place a downsized version of the diagram in the top right corner of the frame
		small_diagram = cv2.resize(frame.diagram, (0, 0), fx=0.25, fy=0.25)
	
		# Add the diagram to the top right corner of the frame
		frame.img[:small_diagram.shape[0], -small_diagram.shape[1]:] = small_diagram
  
		# Convert frame to BGR for writing to video
		frame.img = cv2.cvtColor(frame.img, cv2.COLOR_RGB2BGR)
  
		# Add corner age to display frame
		cv2.putText(frame.img, f"Corner Age: {frame.corners_age}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
	 
		# Write the frame to the output video
		output.write(frame.img)
  
		if show_process:
			# Display the frame
			cv2.imshow("Video", frame.img)
  
		# Update the frame
		frame = frame.next_frame

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