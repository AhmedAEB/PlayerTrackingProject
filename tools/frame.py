from typing import List

import cv2
import numpy as np

from tools.deepsort import DeepSort
from ultralytics import YOLO

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

class Frame():
	def __init__(self, img, n, w, h):
		self.img = img
		self.n = n# frame number for debugging
		self.w, self.h = w, h
  
		self.corners = []
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
  
	def detectAndTrack(self, yoloWrapper: YOLO, trackers: List[DeepSort]):
		# Use YOLO to detect players and referees
		results = yoloWrapper(
			source = self.img,
			verbose = False
		)
		
		result = results[0].cpu().numpy()
		
		if result.boxes is None:
			bboxes, scores, classes = [], [], []
		else:
			detections = splitTeams(result).boxes
		
			# Split detections into bboxes, scores and classes
			bboxes, scores, classes = detections.xyxy, detections.conf, detections.cls

			# Convert bboxes to deepsort format
			bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
			
			classes = classes.flatten().astype(int)
   
			# classes[classes == 3] = 2 # Change team b to team a # TODO: Remove this line when team b is detected properly
			# Split the detections into team a, team b and referees
			teama_boxes = bboxes[classes == 2]
			teamb_boxes = bboxes[classes == 3]
			ref_boxes = bboxes[classes == 1]
   
			teama_scores = scores[classes == 2]
			teamb_scores = scores[classes == 3]
			ref_scores = scores[classes == 1]
   
			# Update the trackers
			print(teama_boxes, teama_scores)
			trackers[0].update(self.img, teama_boxes, teama_scores)
			trackers[1].update(self.img, teamb_boxes, teamb_scores)
			trackers[2].update(self.img, ref_boxes, ref_scores)
	 
			for i, tracker in enumerate(trackers):
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
	
	def detectMask(self, yoloWrapper: YOLO):
		# Use yolo to detect the free throw area
		results = yoloWrapper(
			source = self.img,
			retina_masks = True,
			verbose = False
		)
  
		if results[0].masks is None:
			self.mask = None
			return
		
		self.mask = results[0].masks.masks[0].numpy().astype(np.uint8)
		
		# Change 1 to 255 for display purposes
		self.mask = self.mask * 255
  
		# Calculate the percentage area of the detected mask (white pixels)
		mask_area = np.sum(self.mask == 255) / (self.w * self.h)
		
		# If the mask is too small, or is within 1% of the image top or bottom, remove it
		if mask_area < 0.025 or self.mask[:self.w//100, :].any() or self.mask[-self.w//100:, :].any():
			self.mask = None
			return
		
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
		# Polygon is too close to the left side of the image
		if self.polygon[:, :self.w//50].any():
			# Loop from right to left until a non-zero pixel is found
			for i in range(self.polygon.shape[1] - 1, -1, -1):
				# Check if the column contains any white pixels
				if self.polygon[:, i].any():
					# Remove the polygon and return
					self.corners.append((i, np.argmax(self.polygon[:, i])))
					self.side = "left"
					return
			

		elif self.polygon[:, -self.w//50:].any():
			# Loop from left to right until a non-zero pixel is found
			for i in range(self.polygon.shape[1]):
				# Check if the column contains any white pixels
				if self.polygon[:, i].any():
					# Remove the polygon and return
					self.corners.append((i, np.argmax(self.polygon[:, i])))
					self.side = "right"
					return
			
	 
		# Run a line in each direction along the polygon and 
		# record the coordinates of the first non-zero pixels
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
  
	def displayAll(self):
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
		cv2.imshow("Frame", self.img)
  
		if cv2.waitKey(1):
			return

def updateCorners(Frame1: Frame, Frame2: Frame):
	if len(Frame1.corners) == 1:
		if Frame1.side == "left":
			# Compare the top right corner of Frame2 (frame [1]) with the corner of Frame1
			movement = Frame2.corners[1] - Frame1.corners[0]
			Frame1.corners = Frame2.corners.copy()
			for i in range(len(Frame1.corners)):
				Frame1.corners[i] -= movement
		else:
			# Compare the top left corner of Frame2 (frame [0]) with the corner of Frame1
			movement = Frame2.corners[0] - Frame1.corners[0]
			Frame1.corners = Frame2.corners.copy()
			for i in range(len(Frame1.corners)):
				Frame1.corners[i] -= movement
		Frame1.diagram_corners = Frame2.diagram_corners.copy()
		
		plotCorners(Frame1)
		return
			
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
	avg_movement *= (Frame1.corners_age ** 1.09) * 0.001 + 1
 
	# Update corners with average movement
	avg_movement = avg_movement.astype(int)
	
	for i in range(len(Frame1.corners)):
		Frame1.corners[i] += avg_movement
  
	plotCorners(Frame1)
	
def plotCorners(Frame1: Frame):
	# Plot the corners on the flow frame and frame
	for i, corner in enumerate(Frame1.corners):
		# Check if the corner is within the frame
		if corner[0] < 0 or corner[0] >= Frame1.img.shape[1] or corner[1] < 0 or corner[1] >= Frame1.img.shape[0]:
			continue # Skip the corner if it is not within the frame
		cv2.circle(Frame1.flow_frame, tuple(corner.astype(int)), 5, COLORS[i], -1)
		cv2.circle(Frame1.img, tuple(corner.astype(int)), 5, COLORS[i], -1)

def saveFrame(diagram: np.ndarray, frame: Frame, output: cv2.VideoWriter, debug: bool = False):
	# assert that the frame has 4 corners and the diagram has 4 corners
	assert len(frame.corners) == 4 and len(frame.diagram_corners) == 4
	
	# Draw white square in top left corner for text
	cv2.rectangle(frame.img, (0, 0), (200, 70), (255, 255, 255), -1)

	# Add corner age to display frame
	cv2.putText(frame.img, f"Corner Age: {frame.corners_age}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

	# Add frame number to display frame
	cv2.putText(frame.img, f"Frame: {frame.n}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
 
	# Get the homography for the current frame
	diagram = frame.applyHomography(diagram)

	# Place a downsized version of the diagram in the top right corner of the frame
	small_diagram = cv2.resize(diagram, (0, 0), fx=0.25, fy=0.25)

	# Add the diagram to the top right corner of the frame
	# frame.img[:small_diagram.shape[0], -small_diagram.shape[1]:] = small_diagram

	# Add corner age to display frame
	cv2.putText(frame.img, f"Corner Age: {frame.corners_age}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

	# Add frame number to display frame
	cv2.putText(frame.img, f"Frame: {frame.n}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

	# Display frame and debug frames
	if debug:
		frame.displayAll()

	# Write the frame to the output video
	output.write(frame.img)
  
	# Clear some memory
	frame.clearMemory()
 
TEAMODEL = YOLO("./weights/yolov8n-seg.pt")

def splitTeams(result):
	img = result.orig_img.copy()
	
	# Run a clustering algorithm on the xyxy coordinates of the players to split them into 2 teams
	# This is done by finding the 2 clusters with the most players in them
	
	players = []
 
	# Calculate largest box size
	maxw = int(np.max(result.boxes.xywh[:, 2]))
	maxh = int(np.max(result.boxes.xywh[:, 3]))
 
	for box in result.boxes:
		if box.cls == 0:
			x1, y1, x2, y2 = box.xyxy[0].astype(int)
			cutout = img[
       			int(y1+0.2*maxh) : int(y2-0.4*maxh),
          		int(x1+0.2*maxw) : int(x2-0.2*maxw)
            ]
   
			# cutoutR = TEAMODEL(
			# 	source = cutout,
			# 	retina_masks = True,
			# 	save = True,
			# 	classes = [0],
			# 	show = True,
			# 	verbose = False
			# )
			
			# if cutoutR[0].masks is not None:

			# 	mask = cutoutR[0].masks.masks[0].numpy().astype(np.uint8)
			# 	print(mask.shape, cutout.shape)
			# 	# cv2.imshow("MASK", mask)
			# 	cutout[mask == 0] = 0
   
			# cv2.imshow("CUTOUT", cutout)
	
   
			# Use opencv to calculate the histogram of the cutout
			hist = cv2.calcHist([cutout], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
			
   			# Apply gaussian blur to the histogram
			hist = cv2.GaussianBlur(hist, (9, 9), 0)
   
			# Normalize the histogram to 0-1
			hist = cv2.normalize(hist, hist).flatten()

			# Add the histogram to the players array
			players.append(hist)
   
	players = np.array(players)
	
	# Flatten to 2d array
	players = players.reshape((players.shape[0], -1))
 
	print(players.shape)
	
	# Run a kmeans clustering algorithm on the players
	kmeans = KMeans(n_clusters=2, random_state=0).fit(players)
	
	# Get the labels for each player
	labels = kmeans.labels_
 
	print(labels)
	# Set team a cls to 2 and team b cls to 3
	print("CLS", result.boxes.cls)
	i = 0
	for box in result.boxes:
		if box.cls == 0:
			box.boxes[0][-1] = labels[i] + 2
   
			# Draw box on img (Red for team 1, Blue for team 2)
			x1, y1, x2, y2 = box.xyxy[0].astype(int)
   
			fx1, fy1, fx2, fy2 = x1, y1, x2, y2
			fx = (x1 + x2) / 2
			fy = (y1 + y2) / 2
			fx1 = fx 
   
			if labels[i] == 1:
				cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
			else:
				cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
	
			i += 1
   
	cv2.imshow("TEAMS", img)
	cv2.waitKey(1)
			
	print("CLS", result.boxes.cls)
	return result