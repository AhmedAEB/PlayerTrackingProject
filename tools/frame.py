from typing import List

import cv2
import numpy as np

from tools.deepsort import DeepSortWrapper
from tools.yolov7 import Yolov7Wrapper
from tools.yolov7_seg import Yolov7SegWrapper

from ultralytics import YOLO

from matplotlib import pyplot as plt

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255), (0, 0, 0)]

class Frame():
	def __init__(self, img, n, w, h):
		self.img = img
		self.n = n# frame number for debugging
		self.w, self.h = w, h
  
		self.corners = None
		self.diagram_corners = None
		self.corners_age = 1e+5
		self.mask = None
		self.polygon = None
		self.flow_frame = None

		self.positions = [] # x, y
		self.miny = h + 1
		self.maxy = 0
		self.colours = [] # r, g, b
  
	def clearMemory(self):
		# Remove all unnecessary variables to free up memory
		self.mask = None
		self.polygon = None
		self.flow_frame = None
  
	def detectAndTrack(self, yoloWrapper: YOLO, trackers: List[DeepSortWrapper]):
		# Use YOLO to detect players and referees
		# detections = yoloWrapper.detect(self.img, draw_bboxes=False)
		results = yoloWrapper(
      		source = self.img,
        	verbose = False
        )
		detections = results[0].boxes.numpy()
		
		if detections is None:
			bboxes, scores, classes = [], [], []
		else:
			# Split detections into bboxes, scores and classes
			bboxes, scores, classes = detections.xyxy, detections.conf, detections.cls

			# Convert bboxes to deepsort format
			bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
			
			classes = classes.flatten().astype(int)
			for i, tracker in enumerate(trackers):
				# Get the detections for the current class
				class_mask = classes == i
				class_bboxes = bboxes[class_mask]
				class_scores = scores[class_mask]
				
				# Update the tracker
				tracker.update(self.img, class_bboxes, class_scores)
	 
				# Get the positions of the players and their colours
				colours, positions = tracker.get_positions_c()
	
				# Check if there are any positions
				if len(positions) > 0:
					# Append the positions to the list of positions
					self.positions.extend(positions)
					self.colours.extend(colours)
	 
				# Update the min and max y values
				miny, maxy = tracker.get_min_max_y()
				self.miny = min(self.miny, miny)
				self.maxy = max(self.maxy, maxy)
	
	def detectMask(self, yoloWrapper: Yolov7SegWrapper):
		# Use yolo to detect the free throw area
		self.mask = yoloWrapper.detect(self.img)
  
		# Convert the mask to a binary image
		self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
		self.mask = cv2.threshold(self.mask, 0, 255, cv2.THRESH_BINARY)[1]
  
		# Calculate the percentage area of the detected mask (white pixels)
		mask_area = np.sum(self.mask == 255) / (self.w * self.h)
		
		# If the mask is too small, or is within 1% of the image right or left, remove it
		if mask_area < 0.025 or self.mask[:, :self.w//100].any() or self.mask[:, -self.w//100:].any():
			self.mask = None
		else:
			# Convert the mask to a polygon
			self.convertMaskToPolygon()
   
			# Get the corner points of the polygon
			self.getCornerPoints()
   
			self.corners_age = 0

	def convertMaskToPolygon(self):
		# Find contours
		contours, _ = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		
		# Get the largest contour
		largest_contour = max(contours, key=cv2.contourArea)
		
		# Approximate the contour
		epsilon = 0.01 * cv2.arcLength(largest_contour, True)
	
		# Get the polygon
		polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
	
		# Draw the polygon on a blank image
		self.polygon = np.zeros_like(self.mask)
		cv2.drawContours(self.polygon, [polygon], 0, 255, -1)

	def getCornerPoints(self):
		# Run a line in each direction along the polygon and 
		# record the coordinates of the first non-zero pixels
		self.corners = []
		for i in range(self.polygon.shape[0]):
			# Check if the row contains any white pixels
			if self.polygon[i, :].any():
				self.corners.append((np.argmax(self.polygon[i, :]), i))
				break

		for i in range(self.polygon.shape[0]-1, -1, -1):
			# Check if the row contains any white pixels
			if self.polygon[i, :].any():
				self.corners.append((np.argmax(self.polygon[i, :]), i))
				break

		for i in range(self.polygon.shape[1]):
			# Check if the column contains any white pixels
			if self.polygon[:, i].any():
				self.corners.append((i, np.argmax(self.polygon[:, i])))
				break

		for i in range(self.polygon.shape[1]-1, -1, -1):
			# Check if the column contains any white pixels
			if self.polygon[:, i].any():
				self.corners.append((i, np.argmax(self.polygon[:, i])))
				break

		# Convert to numpy array
		self.corners = np.array(self.corners, dtype=np.float32)
	
		# Put the corners in the correct order (top left, top right, bottom left, bottom right)
		self.corners = self.corners.reshape(-1, 2)
		
		# Order the corners by their y coordinate
		self.corners = self.corners[self.corners[:, 1].argsort()]
  
		# Order the top corners by their x coordinate
		self.corners[:2] = self.corners[:2][self.corners[:2, 0].argsort()]
  
		# Order the bottom corners by their x coordinate
		self.corners[2:] = self.corners[2:][self.corners[2:, 0].argsort()]
  
		# Compare top left and bottom left x coordinates
		if self.corners[0, 0] > self.corners[2, 0]:
			# (53, 470) (424, 470) (53, 780) (424, 780)
			self.diagram_corners = np.array([(53, 470), (424, 470), (53, 780), (424, 780)], dtype=np.float32)
		else:
			# (1521, 473) (1890, 473) (1521, 782) (1890, 782)
			self.diagram_corners = np.array([(1521, 473), (1890, 473), (1521, 782), (1890, 782)], dtype=np.float32)
  
		# Display the corners on the image
		for i, corner in enumerate(self.corners):
			cv2.circle(self.img, tuple(corner.astype(int)), 5, COLORS[i], -1)

	def applyHomography(self, diagram: np.ndarray):
		# if corners or diagram corners are not set, return
		if self.corners is None or self.diagram_corners is None:
			return diagram
	 
		# Get the transformation matrix
		M = cv2.getPerspectiveTransform(self.corners, self.diagram_corners)
  
		if len(self.positions) > 0:
			# Use self.positions and self.colours to draw the positions on the diagram
			positions = np.array(self.positions)[np.newaxis].astype(np.float32)
			positions = cv2.perspectiveTransform(positions, M)
			for i, position in enumerate(positions[0]):
				x, y = position.astype(int)
				cv2.circle(diagram, (x, y), 15, self.colours[i], -1)
	
		# Display corners on diagram
		for i, corner in enumerate(self.diagram_corners):
			cv2.circle(diagram, tuple(corner.astype(int)), 10, COLORS[i], -1)
   
		return diagram
  
	def displayAll(self, debug: bool = False):
		if not debug:
			return
	 
		# Display the optical flow frame
		if self.flow_frame is not None:
			cv2.imshow("Optical Flow", self.flow_frame)
  
		# Display the mask
		if self.mask is not None:
			cv2.imshow("Mask", self.mask)
   
		# Display the polygon
		if self.polygon is not None:
			cv2.imshow("Polygon", self.polygon)
   
		# Display the frame
		# img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
		cv2.imshow("Frame", self.img)
  
		if cv2.waitKey(1):
			return

def updateCorners(Frame1: Frame, Frame2: Frame):
	frame1_img = Frame1.img.copy()
	frame2_img = Frame2.img.copy()
 
	miny = min(Frame1.miny, Frame2.miny)
	maxy = max(Frame1.maxy, Frame2.maxy)

	# Add 3% to the miny and maxy
	miny, maxy = int(miny * 0.97), int(maxy * 1.03)
 
	# for tracker in trackers:
	# 	frame1_img = tracker.get_frame_wo_tracks(frame1_img)
	# 	frame2_img = tracker.get_frame_wo_tracks(frame2_img)
 
	# Apply canny edge detection to the curr and next frames
	frame1_img = cv2.Canny(frame1_img, 100, 200)
	frame2_img = cv2.Canny(frame2_img, 100, 200)
 
	# Calculate optical flow
	flow = cv2.calcOpticalFlowFarneback(frame2_img, frame1_img, None, 0.5, 10, 30, 3, 5, 1.5, 0)
 
	# Create a flow frame to display the flow vectors
	Frame1.flow_frame = Frame1.img.copy()
 
	# Reset flow vectors in range miny to maxy
	flow[miny:maxy, :, :] = 0
 
	# Reset top & bottom 15% of flow vectors
	flow[:int(flow.shape[0] * 0.15), :, :] = 0
	flow[int(flow.shape[0] * 0.85):, :, :] = 0
 
	# Calculate the average x and y movement ignoring vectors with magnitude less than 1
	avg_movement = np.array([0, 0], dtype=np.float32)
	count = 0
	
	# Loop through each pixel in the flow
	for y in range(0, flow.shape[0], 10):
		for x in range(0, flow.shape[1], 10):
			# Get the flow vector
			fx, fy = flow[y, x]
 
			# If the magnitude of the flow vector is greater than 1, add it to the average movement
			if np.sqrt(fx ** 2 + fy ** 2) > 1.5:
				avg_movement += np.array([fx, fy], dtype=np.float32)
				count += 1
 
			# Draw the flow vector in red if moving right, blue if moving left
			if fx > 0:
				cv2.line(Frame1.flow_frame, (x, y), (x + int(fx), y + int(fy)), (0, 0, 255), 1)
			else:
				cv2.line(Frame1.flow_frame, (x, y), (x + int(fx), y + int(fy)), (255, 0, 0), 1)

	# Divide the average movement by the number of vectors to get the average movement
	avg_movement /= count
	
	# Take a copy of the next frame's corners and increment their age
	Frame1.corners = Frame2.corners.copy()
	Frame1.diagram_corners = Frame2.diagram_corners.copy()
 
	Frame1.corners_age = Frame2.corners_age + 1
	
	# Increase avg_movement proportional to the age of the corners
	avg_movement *= (Frame1.corners_age ** 1.09) * 0.01 + 1
 
	# Update corners with average movement
	avg_movement = avg_movement.astype(int)
	
	for i in range(len(Frame1.corners)):
		Frame1.corners[i] += avg_movement
	
	# Plot the corners on the flow frame and frame
	for i, corner in enumerate(Frame1.corners):
		# Check if the corner is within the frame
		if corner[0] < 0 or corner[0] >= Frame1.img.shape[1] or corner[1] < 0 or corner[1] >= Frame1.img.shape[0]:
			continue # Skip the corner if it is not within the frame
		cv2.circle(Frame1.flow_frame, tuple(corner.astype(int)), 5, COLORS[i], -1)
		cv2.circle(Frame1.img, tuple(corner.astype(int)), 5, COLORS[i], -1)
	
	# Convert flow frame to BGR for display
	# Frame1.flow_frame = cv2.cvtColor(Frame1.flow_frame, cv2.COLOR_RGB2BGR)

def saveFrame(diagram: np.ndarray, frame: Frame, output: cv2.VideoWriter, debug: bool = False):
	# assert that the frame has 4 corners and the diagram has 4 corners
	assert len(frame.corners) == 4 and len(frame.diagram_corners) == 4
	
 	# Draw white square in top left corner for text
	cv2.rectangle(frame.img, (0, 0), (200, 70), (255, 255, 255), -1)

	# Get the homography for the current frame
	diagram = frame.applyHomography(diagram)

	# Place a downsized version of the diagram in the top right corner of the frame
	small_diagram = cv2.resize(diagram, (0, 0), fx=0.25, fy=0.25)

	# Add the diagram to the top right corner of the frame
	frame.img[:small_diagram.shape[0], -small_diagram.shape[1]:] = small_diagram

	# Display frame and debug frames
	if debug:
		frame.displayAll(debug = True)
  
	# Convert frame to BGR for writing to video
	# frame.img = cv2.cvtColor(frame.img, cv2.COLOR_RGB2BGR)

	# Add corner age to display frame
	cv2.putText(frame.img, f"Corner Age: {frame.corners_age}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

	# Add frame number to display frame
	cv2.putText(frame.img, f"Frame: {frame.n}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

	# Write the frame to the output video
	output.write(frame.img)
  
	# Clear some memory
	frame.clearMemory()