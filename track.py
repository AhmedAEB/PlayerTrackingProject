import argparse
from os import path
from time import perf_counter
from tqdm import trange

import cv2
import numpy as np

from tools.deepsort import DeepSortWrapper
from tools.yolov7 import Yolov7Wrapper
from tools.yolov7_seg import Yolov7SegWrapper

def run(opt):
	# Get arguments
	source, save_path, yolo_weights, yolo_seg_weights, deepsort_weights, device = path.abspath(opt.source), path.abspath(opt.save_path), opt.yolo_weights, opt.yolo_seg_weights, opt.deepsort_weights, opt.device
	
	# Get hyperparameters
	max_cos_dist, nn_budget, iou_th, conf_th, iou_th_seg, conf_th_seg, max_track_age, init_phase_frames = opt.max_cos_dist, opt.nn_budget, opt.iou_th, opt.conf_th, opt.iou_th_seg, opt.conf_th_seg, opt.max_track_age, opt.init_phase_frames

	# Initialise paused flag
	paused = False

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
	max_mask_area = 0
 
	# Get total number of frames
	total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

	# Loop through all frames
	for i in trange(total_frames, desc="YOLOv7 + DeepSORT + YOLACT"):
		# Start timer for current frame
		start = perf_counter()

		# Get next frame from video
		ret, frame = video.read()

		# Check if video has ended
		if not ret:
			break

		# Run court segmentation
		masked_frame = yolov7_seg.detect(frame)
  
		# Convert to binary image
		binary = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
		_, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY)
  
		# Calculate the percentage area of the detected mask (white pixels)
		mask_area = np.sum(binary == 255) / (w * h)
  
		# Display a line 10% of the way from the left and right of the frame
		cv2.line(frame, (w//20, 0), (w//20, h), (0, 255, 0), 2)
		cv2.line(frame, (w - w//20, 0), (w - w//20, h), (0, 255, 0), 2)
  
		# Find contours
		cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if len(cnts) == 2 else cnts[1]
		# Approximate the contours
		if len(cnts) > 0:
			epsilon = 0.01 * cv2.arcLength(cnts[-1], True)
			approx = cv2.approxPolyDP(cnts[-1], epsilon, True)
			# Draw the contours
			cv2.drawContours(frame, [approx], -1, (0, 0, 255), 2)
			# Draw polygon on a blank binary image
			blank = np.zeros_like(binary)
			cv2.drawContours(blank, [approx], -1, (255, 255, 255), -1)

			# Corner detection using good features to track
			corners = cv2.goodFeaturesToTrack(blank, 4, 0.01, 10)
			corners = np.int0(corners)
			for corner in corners:
				x, y = corner.ravel()
				cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

			# Display the polygon
			cv2.imshow("Polygon", blank)
   
		# Keep track of the maximum mask area frame if the first and last 5% lines do not contain any white pixels
		if mask_area > max_mask_area and (binary[:, w//20] == 0).all() and (binary[:, w - w//20] == 0).all():
			max_mask_area = mask_area
			max_mask_frame = frame
			cv2.imwrite("./data/max_mask_frame.jpg", max_mask_frame)
  
		# Display the binary image
		cv2.imshow("Mask", binary)
  
		# Convert frames to RGB
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
  
		# Use YOLO to detect players and referees
		detections = yolov7.detect(frame, draw_bboxes=False)
		
		if detections is None:
			bboxes, scores, classes = [], [], []
		else:
			# Split detections into bboxes, scores and classes
			bboxes, scores, classes = np.hsplit(detections, [4, 5])

			# Convert bboxes to xyxy format for DeepSORT #TODO: Double check deepsort format
			bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
			
		# Get indices of the players (0) and referees (1)
		classes = classes.flatten().astype(int)
		player_indices = np.where(classes == 0)[0]
		referee_indices = np.where(classes == 1)[0]
		
		# Get the bboxes and scores of the players and referee
		player_bboxes, player_scores = bboxes[player_indices], scores[player_indices]
		referee_bboxes, referee_scores = bboxes[referee_indices], scores[referee_indices]
		
		# Track the players and referee separately using DeepSORT
		player_tracker.update(frame, player_bboxes, player_scores.flatten())
		referee_tracker.update(frame, referee_bboxes, referee_scores.flatten())
		
		# Draw white square in top left corner for text
		cv2.rectangle(frame, (0, 0), (185, 130), (255, 255, 255), -1)
		
		# Add fps to display frame
		cv2.putText(frame, f"FPS: {1/(perf_counter()-start):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		
		# Add number of objects to display frame, split into players and referees
		cv2.putText(frame, f"Players: {len(player_tracker.tracker.tracks)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
		cv2.putText(frame, f"Referees: {len(referee_tracker.tracker.tracks)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
  
		# Add mask area to display frame
		cv2.putText(frame, f"Mask: {mask_area*100:.2f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
		
		# Write to output video
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		output.write(frame)
		
		# Display processing frame to user
		cv2.imshow("Processing...", frame)
  
		# Check if q is pressed to stop tracking
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		# Check if p is pressed to pause tracking
		elif cv2.waitKey(1) & 0xFF == ord('p'):
			paused = True
   
		# Handle pause state
		while paused:
			# Check if n is pressed to step to next frame
			if cv2.waitKey(1) & 0xFF == ord('n'):
				break
			# Check if r is pressed to resume tracking
			elif cv2.waitKey(1) & 0xFF == ord('r'):
				paused = False
				break
					
		# Update the time taken and increment frame count
		total_time += perf_counter() - start
		i += 1

	print(f"\nFPS: {i/total_time:.2f} Hz")
	cv2.destroyAllWindows()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", type=str, default="./data/input.mp4", help="Path to a basketball video")
	parser.add_argument("--save_path", type=str, default="./data/output.mp4", help="Path to save the output video to")
	parser.add_argument("--yolo_weights", type=str, default="./weights/cleanedDatasetWeights.pt", help="Path to the YOLO weights file (.pt)")
	parser.add_argument("--yolo_seg_weights", type=str, default="./weights/best.pt", help="Path to the YOLO segmentation weights file (.pt)")
	parser.add_argument("--deepsort_weights", type=str, default="./weights/market1501.pb", help="Path to the DeepSORT weights file (.pb)")
	parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
 
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