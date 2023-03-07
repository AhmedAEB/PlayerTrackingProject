from typing import List

import cv2
import numpy as np

from tools.deepsort import DeepSort
from tools.plot import plot_one_box
from ultralytics import YOLO

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

# grey, pastel blue, pastel red for referees, team A, team B
BOX_COLORS = [(128, 128, 128), (128, 128, 255), (255, 128, 128)]

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

		self.players = []
		self.referees = []
  
	def clearMemory(self):
		# Remove all unnecessary variables to free up memory
		self.mask = None
		self.polygon = None
		self.flow_frame = None
  
	def plotTracks(self, tracks, colours, labels):
		for track in tracks:
			colour = colours[track.team]
	  
			xyxy = track.to_tlbr().astype(int)
   
			plot_one_box(
				   xyxy,
				  self.img,
				label = f'{track.track_id} : {labels[track.team]}',
				 color = colour,
				  line_thickness = 2
			)
  
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
			# detections = splitTeams(result).boxes
			detections = result.boxes
		
			# Split detections into bboxes, scores and classes
			bboxes, scores, classes = detections.xyxy, detections.conf, detections.cls

			# Convert bboxes to deepsort format
			bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
			
			classes = classes.flatten().astype(int)
   
			# classes[classes == 3] = 2 # Change team b to team a # TODO: Remove this line when team b is detected properly
			# Split the detections into team a, team b and referees
			# teamb_boxes = bboxes[classes == 3]
			player_boxes = bboxes[classes == 0]
			ref_boxes = bboxes[classes == 1]
   
			# teamb_scores = scores[classes == 3]
			player_scores = scores[classes == 0]
			ref_scores = scores[classes == 1]
   
			prev_players = trackers[0].tracker.tracks
   
			# Update the trackers
			self.players = trackers[0].update(self.img, player_boxes, player_scores)
			# trackers[1].update(self.img, teamb_boxes, teamb_scores)
			self.referees = trackers[1].update(self.img, ref_boxes, ref_scores)
   
			self.players = splitTeams(prev_players, self.players, self)
   
			self.plotTracks(self.referees, BOX_COLORS, ['Referee'])
			self.plotTracks(self.players, BOX_COLORS[1:], ['Team A', 'Team B'])
	
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

		# Loop through the players and draw their positions on the diagram
		for track in self.players:
			# Get the player's position
			xyxy = track.to_tlbr().astype(int)
   
			# Get the player's centre
			x = (xyxy[0] + xyxy[2]) // 2
			y = xyxy[3]
   
			# Get the player's position on the diagram
			xy = np.array([[x, y]], dtype=np.float32)[np.newaxis]
	
			# Transform the player's position
			xy = cv2.perspectiveTransform(xy, M)[0][0]
   
			# Draw the player's position on the diagram based on their team
			cv2.circle(diagram, tuple(xy.astype(int)), 15, BOX_COLORS[track.team + 1], -1)
   
		# Do the same for the referees
		for track in self.referees:
			# Get the referee's position
			xyxy = track.to_tlbr().astype(int)
   
			# Get the referee's centre
			x = (xyxy[0] + xyxy[2]) // 2
			y = xyxy[3]
   
			# Get the referee's position on the diagram
			xy = np.array([[x, y]], dtype=np.float32)[np.newaxis]
	
			# Transform the referee's position
			xy = cv2.perspectiveTransform(xy, M)[0][0]
   
			# Draw the referee's position on the diagram based on their team
			cv2.circle(diagram, tuple(xy.astype(int)), 15, BOX_COLORS[track.team], -1)
	
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

	
	def remove_tracks(self, flow):
		for track in self.players + self.referees:
			x1, y1, x2, y2 = track.to_tlbr().astype(int)

			# Draw a solid black rectangle over the bbox + 100% padding
			x1, y1, x2, y2 = x1 - int((x2 - x1) / 2), y1 - int((y2 - y1) / 2), x2 + int((x2 - x1) / 2), y2 + int((y2 - y1) / 2)
			flow[y1:y2, x1:x2] = 0

		return flow

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
 
	# Apply canny edge detection to the curr and next frames
	frame1_img = cv2.Canny(frame1_img, 100, 200)
	frame2_img = cv2.Canny(frame2_img, 100, 200)
 
	# Calculate optical flow
	flow = cv2.calcOpticalFlowFarneback(frame2_img, frame1_img, None, 0.5, 10, 30, 3, 5, 1.5, 0)
 
	# Create a flow frame to display the flow vectors
	Frame1.flow_frame = Frame1.img.copy()
 
	# Reset flow vectors around detected players and referees
	flow = Frame1.remove_tracks(flow)
 
	# Reset top & bottom 15% of flow vectors
	flow[:int(flow.shape[0] * 0.15), :, :] = 0
	flow[int(flow.shape[0] * 0.85):, :, :] = 0
 
	# Calculate the average x and y movement ignoring vectors with magnitude less than 1
	avg_movement = np.array([0, 0], dtype=np.float32)
	count = 1
	
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
	# avg_movement *= (Frame1.corners_age ** 1.09) * 0.001 + 1
 
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
	frame.img[:small_diagram.shape[0], -small_diagram.shape[1]:] = small_diagram

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

def splitTeams(prev_tracks, tracks, frame):
	# Exit early if there are not enough tracks to split into 2 teams
	if len(tracks) < 2:
		return tracks
	
	# Run a clustering algorithm on the xyxy coordinates of the players to split them into 2 teams
	# This is done by finding the 2 clusters with the most players in them
	img = frame.img.copy()
 
	players = []
 
	for track in tracks:
		x1, y1, x2, y2 = track.to_tlbr().astype(int)
		w, h = x2 - x1, y2 - y1
  
		# Adjust x and y positions to make them closer to the player's jersey
		x1 += int(0.3 * w)
		y1 += int(0.25 * h)
		x2 -= int(0.4 * w)
		y2 -= int(0.45 * h)

		# Display a bounding box around the new positions
		cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
  
		cutout = img[y1:y2, x1:x2]

		# Use opencv to calculate the histogram of the cutout
		hist = cv2.calcHist([cutout], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		
		# Apply gaussian blur to the histogram
		hist = cv2.GaussianBlur(hist, (9, 9), 0)

		# Normalize the histogram to 0-1
		hist = cv2.normalize(hist, hist).flatten()

		# Add the histogram to the players array
		players.append(hist)
  
	cv2.imshow("Team Detection", img)

	players = np.array(players)
	
	# Flatten to 2d array
	players = players.reshape((players.shape[0], -1))
	
	# Run a kmeans clustering algorithm on the players
	kmeans = KMeans(
		 n_clusters = 2,
		  random_state = 0,
		   n_init = "auto"
	).fit(players)
	
	# Get the labels for each player
	labels = kmeans.labels_
 
	id_to_team = {}
	for track in prev_tracks:
		id_to_team[track.track_id] = track.team
	
	correct = 0
	total = 0
 
	for i, track in enumerate(tracks):
		track.team = labels[i]
  
		if track.track_id in id_to_team:
			if id_to_team[track.track_id] == track.team:
				correct += 1
			total += 1
   
	accuracy = correct / total if total > 0 else 0
 
	if accuracy < 0.5:
		# Swap the teams if the accuracy is less than 50%
		for track in tracks:
			track.team = 1 - track.team
   
	# Calculate final team for each track based on team prediction history
	for track in tracks:
		track.prev_teams.append(track.team)
	
		# Set the team to the most common team in the last 15 frames
		track.team = max(set(track.prev_teams[-15:]), key = track.prev_teams[-15:].count)
 
	return tracks