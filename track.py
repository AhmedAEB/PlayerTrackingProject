import argparse
from os import path
from time import perf_counter
from tqdm import trange

import cv2
import numpy as np

from tools.deepsort import DeepSortWrapper
from tools.yolov7 import Yolov7Wrapper
from tools.yolov7_seg import Yolov7SegWrapper

def convertMaskToPolygon(binaryMask):
	# Find contours
	contours, _ = cv2.findContours(binaryMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	# Get the largest contour
	largest_contour = max(contours, key=cv2.contourArea)
	
	# Approximate the contour
	epsilon = 0.01 * cv2.arcLength(largest_contour, True)
 
	# Get the polygon
	polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
 
	# Draw the polygon on a blank image
	blank = np.zeros_like(binaryMask)
	cv2.drawContours(blank, [polygon], 0, 255, -1)
	
	return blank

def getCornerPoints(polygon):
	# Run a line in each direction along the polygon and 
	# record the coordinates of the first non-zero pixels
	corners = []
	for i in range(polygon.shape[0]):
		# Check if the row contains any white pixels
		if polygon[i, :].any():
			corners.append((np.argmax(polygon[i, :]), i))
			break

	for i in range(polygon.shape[0]-1, -1, -1):
		# Check if the row contains any white pixels
		if polygon[i, :].any():
			corners.append((np.argmax(polygon[i, :]), i))
			break

	for i in range(polygon.shape[1]):
		# Check if the column contains any white pixels
		if polygon[:, i].any():
			corners.append((i, np.argmax(polygon[:, i])))
			break

	for i in range(polygon.shape[1]-1, -1, -1):
		# Check if the column contains any white pixels
		if polygon[:, i].any():
			corners.append((i, np.argmax(polygon[:, i])))
			break

	# Convert to numpy array
	corners = np.array(corners)
 
	# Put the corners in the correct order (top left, top right, bottom left, bottom right)
	corners = corners.reshape(-1, 2)
	top = corners[np.argsort(corners[:, 1])[:2]]
	bottom = corners[np.argsort(corners[:, 1])[2:]]
	top_left, top_right = top[np.argsort(top[:, 0])[:2]]
	bottom_left, bottom_right = bottom[np.argsort(bottom[:, 0])[:2]]
	corners = np.array([top_left, top_right, bottom_left, bottom_right])
	
	return corners

def applyTransformation(tracker, M, diagram):
	# Get referee positions (track_id, x, y)
	positions = tracker.get_positions()

	if len(positions) == 0:
		return diagram

	xy = positions[:, 1:]

	# Convert referee positions to diagram coordinates
	diagram_positions = cv2.perspectiveTransform(xy.reshape(-1, 1, 2), M)

	# Draw referee positions on diagram
	for i, position in enumerate(diagram_positions):
		colour = tracker.get_colour(int(positions[i, 0]))[::-1]
		x, y = position[0].astype(int)
		cv2.circle(diagram, (x, y), 15, colour, -1)
  
	return diagram
		
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
 
	# Get total number of frames
	total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

	# Loop through all frames
	for i in trange(total_frames, desc="YOLOv7 + DeepSORT + YOLACT"):
		# Start timer for current frame
		start = perf_counter()

		# Get next frame from video
		ret, orig = video.read()
  
		frame = orig.copy()

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
   
		# If the first and last 2.5% of the frame contains white pixels, then the frame is a center frame
		center = binary[:, :w//40].any() or binary[:, -w//40:].any() or mask_area < 0.01
  
		if center:
			state = "Center"
		else:
			# Convert the binary image to a polygon
			polygon = convertMaskToPolygon(binary)
   
			# Extract the top left, top right, bottom left and bottom right corners of the polygon
			corners = getCornerPoints(polygon)
	
			# Compare the top left and bottom left corners
			if corners[0][0] > corners[2][0]:
				state = "Left"
				# Free Throw Coordinates of 4 corners in order (top left, top right, bottom left, bottom right)
				# (53, 470) (424, 470) (53, 780) (424, 780)
				diagram_corners = np.array([[53, 470], [424, 470], [53, 780], [424, 780]], dtype=np.float32)
			else:
				state = "Right"
				# Free Throw Coordinates of 4 corners in order (top left, top right, bottom left, bottom right)
				# (1521, 473) (1890, 473) (1521, 782) (1890, 782) 
				diagram_corners = np.array([[1521, 473], [1890, 473], [1521, 782], [1890, 782]], dtype=np.float32)
	
			# Make the corners blue, red, green, yellow respectively
			cv2.circle(frame, (corners[0][0], corners[0][1]), 5, (255, 0, 0), -1)
			cv2.circle(frame, (corners[1][0], corners[1][1]), 5, (0, 0, 255), -1)
			cv2.circle(frame, (corners[2][0], corners[2][1]), 5, (0, 255, 0), -1)
			cv2.circle(frame, (corners[3][0], corners[3][1]), 5, (0, 255, 255), -1)

			# Convert corners to float32
			corners = corners.astype(np.float32)

			# Get the perspective transform matrix
			M = cv2.getPerspectiveTransform(corners, diagram_corners)

			# Display the polygon
			cv2.imshow("Polygon", polygon)

			# Display the binary image
			cv2.imshow("Mask", binary)
  
		# Convert frames to RGB
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
  
		# Use YOLO to detect players and referees
		detections = yolov7.detect(orig, draw_bboxes=False)
		
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
  
		if center:
			# Remove players and referees from the frame
			curr_frame = frame.copy()
   
			curr_frame = player_tracker.get_frame_wo_tracks(curr_frame)
			curr_frame = referee_tracker.get_frame_wo_tracks(curr_frame)
			prev_frame = player_tracker.get_frame_wo_tracks(prev_frame)
			prev_frame = referee_tracker.get_frame_wo_tracks(prev_frame)
	  
			# Run optical flow between prev and current frame to calculate average camera movement
			prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
			curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
   
			# Calculate optical flow
			flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 30, 3, 5, 1.2, 0)
   
			flow_frame = curr_frame.copy()
   
			# Loop through each pixel in the flow
			for y in range(0, flow.shape[0], 10):
				for x in range(0, flow.shape[1], 10):
					# Get the flow vector
					fx, fy = flow[y, x]
   
					# Draw the flow vector
					cv2.line(flow_frame, (x, y), (int(x + fx), int(y + fy)), (0, 255, 0))
	 
			# Calculate the average x and y movement
			avg_movement = np.mean(flow, axis=(0, 1))*2
   
			# Update corners with average movement
			avg_movement = avg_movement.astype(int)
			for i in range(len(corners)):
				corners[i] += avg_movement
	
			# Plot the corners
			corners = corners.astype(int)
			cv2.circle(flow_frame, (corners[0][0], corners[0][1]), 5, (255, 0, 0), -1)
			cv2.circle(flow_frame, (corners[1][0], corners[1][1]), 5, (0, 0, 255), -1)
			cv2.circle(flow_frame, (corners[2][0], corners[2][1]), 5, (0, 255, 0), -1)
			cv2.circle(flow_frame, (corners[3][0], corners[3][1]), 5, (0, 255, 255), -1)
   
			# Display the optical flow
			cv2.imshow("Optical Flow", flow_frame)
   
			# Compare the top left and bottom left corners
			if corners[0][0] > corners[2][0]:
				state = "Left"
				# Free Throw Coordinates of 4 corners in order (top left, top right, bottom left, bottom right)
				# (53, 470) (424, 470) (53, 780) (424, 780)
				diagram_corners = np.array([[53, 470], [424, 470], [53, 780], [424, 780]], dtype=np.float32)
			else:
				state = "Right"
				# Free Throw Coordinates of 4 corners in order (top left, top right, bottom left, bottom right)
				# (1521, 473) (1890, 473) (1521, 782) (1890, 782) 
				diagram_corners = np.array([[1521, 473], [1890, 473], [1521, 782], [1890, 782]], dtype=np.float32)
   
			# Convert corners to float32
			corners = corners.astype(np.float32)

			# Get the perspective transform matrix
			M = cv2.getPerspectiveTransform(corners, diagram_corners)
   
			center = False

  
		# Display referee positions on diagram using perspective transform matrix
		if not center:
			diagram = cv2.imread("diagram.jpg")
   
			# Add player and referee positions to diagram
			diagram = applyTransformation(referee_tracker, M, diagram)
			diagram = applyTransformation(player_tracker, M, diagram)
   
			# Draw corners on diagram in same colour as corner points
			diagram_corners = diagram_corners.astype(int)
			cv2.circle(diagram, (diagram_corners[0][0], diagram_corners[0][1]), 15, (255, 0, 0), -1)
			cv2.circle(diagram, (diagram_corners[1][0], diagram_corners[1][1]), 15, (0, 0, 255), -1)
			cv2.circle(diagram, (diagram_corners[2][0], diagram_corners[2][1]), 15, (0, 255, 0), -1)
			cv2.circle(diagram, (diagram_corners[3][0], diagram_corners[3][1]), 15, (0, 255, 255), -1)

			# Place a downsized version of the diagram in the top right corner of the frame
			diagram = cv2.resize(diagram, (0, 0), fx=0.25, fy=0.25)
   
			# Change colour space to BGR
			diagram = cv2.cvtColor(diagram, cv2.COLOR_RGB2BGR)
			frame[:diagram.shape[0], -diagram.shape[1]:] = diagram
		
		# Draw white square in top left corner for text
		cv2.rectangle(frame, (0, 0), (185, 160), (255, 255, 255), -1)
		
		# Add fps to display frame
		cv2.putText(frame, f"FPS: {1/(perf_counter()-start):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		
		# Add number of objects to display frame, split into players and referees
		cv2.putText(frame, f"Players: {len(player_tracker.tracker.tracks)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
		cv2.putText(frame, f"Referees: {len(referee_tracker.tracker.tracks)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

		# Add mask area to display frame
		cv2.putText(frame, f"Mask: {mask_area*100:.2f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
		
		# Add frame state to display frame
		cv2.putText(frame, f"State: {state}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
  
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

		prev_frame = orig.copy()
		# prev_frame = player_tracker.get_frame_wo_tracks(prev_frame)
		# prev_frame = referee_tracker.get_frame_wo_tracks(prev_frame)
					
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