from typing import List

import cv2
import numpy as np

from utils.deepsort import DeepSort, BTrack
from ultralytics import YOLO

from sklearn.cluster import KMeans

# Colour constants for plotting tracks
CORNER_COLOURS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
TEAM_A_COLOUR = (128, 128, 255)  # pastel blue
TEAM_B_COLOUR = (255, 128, 128)  # pastel red
REFEREE_COLOUR = (128, 128, 128) # grey


class Frame():
	def __init__(self, img, index, engine):
		self.orig_img = img.copy()

		self.img = img
		self.index = index
		self.engine = engine

		self.corners = []
		self.xyxy = []
		self.players = []
		self.referees = []
		self.diagram_corners = None
		self.corners_age = 1e+5

		# Debug frames
		self.mask = None
		self.polygon = None
		self.flow_frame = None
		self.team_frame = None
		self.diagram = self.engine.diagram.copy()

	def clearMemory(self):
		# Remove all unnecessary variables to free up memory
		self.mask = None
		self.polygon = None
		self.flow_frame = None
		self.team_frame = None
		self.diagram = None

	def plotTracks(self, tracks, colours, labels):
		for track in tracks:
			colour = colours[track.team]

			x1, y1, x2, y2 = track.to_tlbr().astype(int)

			# Plot bounding box
			cv2.rectangle(self.img, (x1, y1), (x2, y2), colour, 2)

			# Calculate the extra width needed for the label
			extra_width = 12 * (len(str(track.track_id)) - 1)

			# Draw a box ontop of the player for the label
			cv2.rectangle(
				self.img,
				(x1 - 1, y1 - 20),
				(x1 + 95 + extra_width, y1),
				colour,
				-1
			)

			# Plot label text
			cv2.putText(
				self.img,
				f"{track.track_id} : {labels[track.team]}",
				(x1 + 2, y1 - 5),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				(0, 0, 0),
				1
			)

	def detect_players_and_referees(self):
		# Use YOLO to detect players and referees
		results = self.engine.yolov8(
			source=self.orig_img,
			verbose=False
		)
		
		# Return the detection as a Result object
		return results[0].cpu().numpy()
	
	def track(self):
		# Use yolov8 to detect players and referee
		detection = self.detect_players_and_referees()
     
		if detection.boxes is not None:
			boxes = detection.boxes

			# Split detections into bboxes, scores and classes
			bboxes, scores, classes = boxes.xyxy, boxes.conf, boxes.cls

			# Convert bboxes to deepsort format
			bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]

			# Convert classes to integers
			classes = classes.flatten().astype(int)

			# Split detections into players and referees based on class
			player_boxes = bboxes[classes == 0]
			ref_boxes = bboxes[classes == 1]
			player_scores = scores[classes == 0]
			ref_scores = scores[classes == 1]

			# Store previous player tracks
			prev_players = self.engine.player_tracker.tracker.tracks

			# Update the trackers
			self.players = self.engine.player_tracker.update(
				self.orig_img,
				player_boxes,
				player_scores
			)
			self.referees = self.engine.referee_tracker.update(
				self.orig_img,
				ref_boxes,
				ref_scores
			)

			# Save xyxy coordinates of all detections
			for track in self.players + self.referees:
				self.xyxy.append(track.to_tlbr().astype(int))

			# Apply team classification to players
			self.players = self.splitTeams(prev_players, self.players)

			self.plotTracks(self.referees, [REFEREE_COLOUR], ['Referee'])
			self.plotTracks(self.players, [TEAM_A_COLOUR, TEAM_B_COLOUR], ['Team A', 'Team B'])

	def detect_image(self):
		# Use YOLO to detect players and referees
		detection = self.detect_players_and_referees()

		if detections is not None:
			# Split detections into bboxes, scores and classes
			bboxes, classes = detections.xyxy, detections.cls

			classes = classes.flatten().astype(int)

			player_boxes = bboxes[classes == 0]
			ref_boxes = bboxes[classes == 1]

			for i, xyxy in enumerate(player_boxes):
				self.players.append(BTrack.fake_track(xyxy, i))

			for i, xyxy in enumerate(ref_boxes):
				self.referees.append(BTrack.fake_track(xyxy, i))

			self.players = self.splitTeams([], self.players)

			self.plotTracks(self.referees, [REFEREE_COLOUR], ['Referee'])
			self.plotTracks(self.players, [TEAM_A_COLOUR, TEAM_B_COLOUR], ['Team A', 'Team B'])

	def getCorners(self):
		# Use yolo to detect the free throw area
		results = self.engine.yolov8_seg(
			source=self.orig_img,
			retina_masks=True,
			verbose=False
		)

		if results[0].masks is None:
			return

		self.mask = results[0].masks.masks[0].cpu().numpy().astype(np.uint8)

		# Change 1 to 255 for display purposes
		self.mask = self.mask * 255

		# Calculate the percentage area of the detected mask (white pixels)
		mask_area = np.sum(self.mask == 255) / (self.engine.w * self.engine.h)

		# If the mask is too small, or is within 1% of the image top or bottom,
		# remove it
		if mask_area < 0.025 or \
				self.mask[:self.engine.w // 100, :].any() or self.mask[-self.engine.w // 100:, :].any():
			return

		# Convert the mask to a polygon
		self.convertMaskToPolygon()

		# Get the corner points of the polygon
		self.getCornerPoints()

		self.corners_age = 0

	def convertMaskToPolygon(self):
		# Find contours
		contours, _ = cv2.findContours(
			self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
		if self.polygon[:, :self.engine.w // 50].any():
			# Loop from right to left until a non-zero pixel is found
			for i in range(self.polygon.shape[1] - 1, -1, -1):
				# Check if the column contains any white pixels
				if self.polygon[:, i].any():
					# Remove the polygon and return
					self.corners.append((i, np.argmax(self.polygon[:, i])))
					self.side = "left"
					return

		elif self.polygon[:, -self.engine.w // 50:].any():
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

		for i in range(self.polygon.shape[0] - 1, -1, -1):
			# Check if the row contains any white pixels
			if self.polygon[i, :].any():
				self.corners.append((np.argmax(self.polygon[i, :]), i))
				break

		for i in range(self.polygon.shape[1]):
			# Check if the column contains any white pixels
			if self.polygon[:, i].any():
				self.corners.append((i, np.argmax(self.polygon[:, i])))
				break

		for i in range(self.polygon.shape[1] - 1, -1, -1):
			# Check if the column contains any white pixels
			if self.polygon[:, i].any():
				self.corners.append((i, np.argmax(self.polygon[:, i])))
				break

		# Convert to numpy array
		self.corners = np.array(self.corners, dtype=np.float32)

		# Put the corners in the correct order (top left, top right, bottom
		# left, bottom right)
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
			self.diagram_corners = np.array(
				[(53, 470), (424, 470), (53, 780), (424, 780)], dtype=np.float32)
		else:
			# (1521, 473) (1890, 473) (1521, 782) (1890, 782)
			self.diagram_corners = np.array(
				[(1521, 473), (1890, 473), (1521, 782), (1890, 782)], dtype=np.float32)

		# Display the corners on the image
		for i, corner in enumerate(self.corners):
			cv2.circle(self.img, tuple(corner.astype(int)), 5, CORNER_COLOURS[i], -1)

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
   
			# Get the player's team COLOUR
			COLOUR = [TEAM_A_COLOUR, TEAM_B_COLOUR][track.team]

			# Draw the player's position on the diagram based on their team
			cv2.circle(diagram, tuple(xy.astype(int)),
					   15, COLOUR, -1)

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

			# Draw the referee's position on the diagram
			cv2.circle(diagram, tuple(xy.astype(int)),
					   15, REFEREE_COLOUR, -1)

		# Display corners on diagram
		for i, corner in enumerate(self.diagram_corners):
			cv2.circle(diagram, tuple(corner.astype(int)), 10, CORNER_COLOURS[i], -1)

		return diagram

	def displayAll(self):
		# Display the mask
		if self.mask is not None:
			cv2.imshow("Mask", self.mask)

		# Display the polygon
		if self.polygon is not None:
			cv2.imshow("Polygon", self.polygon)

		# Display the optical flow frame
		if self.flow_frame is not None:
			cv2.imshow("Optical Flow", self.flow_frame)

		# Display the Team Split
		if self.team_frame is not None:
			cv2.imshow("Team Split", self.team_frame)

		# Display the birds eye view diagram
		if self.diagram is not None:
			cv2.imshow("Birds Eye View", self.diagram)

		# Display the frame
		cv2.imshow("Frame", self.img)

		if cv2.waitKey(1):
			return

	def remove_tracks(self, flow):
		for xyxy in self.xyxy:
			x1, y1, x2, y2 = xyxy

			x1 -= 20
			y1 -= 20
			x2 += 20
			y2 += 20

			# Set flow to 0 for the area around the player, flow is of shape
			# (h, w, 2)
			cv2.rectangle(flow, (x1, y1), (x2, y2), (0, 0), -1)

		return flow

	def splitTeams(self, prev_tracks, tracks):
		# Exit early if there are not enough tracks to split into 2 teams
		if len(tracks) < 2:
			return tracks

		# Run a clustering algorithm on the xyxy coordinates of the players to split them into 2 teams
		# This is done by finding the 2 clusters with the most players in them
		self.team_frame = self.orig_img.copy()

		players = []

		for track in tracks:
			x1, y1, x2, y2 = track.to_tlbr().astype(int)
			w, h = x2 - x1, y2 - y1

			# Adjust x and y positions to make them closer to the player's
			# jersey
			x1 += int(0.3 * w)
			y1 += int(0.25 * h)
			x2 -= int(0.4 * w)
			y2 -= int(0.45 * h)

			# Get the cutout of the player
			cutout = self.team_frame[y1:y2, x1:x2]

			# Display a bounding box around the new positions
			cv2.rectangle(self.team_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

			# Use opencv to calculate the histogram of the cutout
			hist = cv2.calcHist(
				[cutout],
				[0, 1, 2],
				None,
				[8, 8, 8],
				[0, 256, 0, 256, 0, 256]
			)

			# Apply gaussian blur to the histogram
			hist = cv2.GaussianBlur(hist, (9, 9), 0)

			# Normalize the histogram to 0-1
			hist = cv2.normalize(hist, hist).flatten()

			# Add the histogram to the players array
			players.append(hist)

		players = np.array(players)

		# Flatten to 2d array
		players = players.reshape((players.shape[0], -1))

		# Run a kmeans clustering algorithm on the players
		kmeans = KMeans(
			n_clusters=2,
			random_state=0,
			n_init="auto"
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

			# Set the team to the most common team in the last 20 frames
			track.team = max(
				set(track.prev_teams[-20:]), key=track.prev_teams[-20:].count)

		return tracks

	def plotCorners(self):
		# Plot the corners on the flow frame and frame
		for i, corner in enumerate(self.corners):
			# Check if the corner is within the frame
			if corner[0] < 0 or corner[0] >= self.img.shape[1] or corner[1] < 0 or corner[1] >= self.img.shape[0]:
				continue  # Skip the corner if it is not within the frame
			cv2.circle(
				self.flow_frame, tuple(
					corner.astype(int)), 5, CORNER_COLOURS[i], -1)
			cv2.circle(self.img, tuple(corner.astype(int)), 5, CORNER_COLOURS[i], -1)

	def saveFrame(
			self,
			output: cv2.VideoWriter,
			show: bool = False):
		# Run some validation checks on the frame

		# Check that the frame has 4 corners
		assert len(self.corners) == 4 and len(self.diagram_corners) == 4

		# Check that the frames corners age is less than 1e+5
		assert self.corners_age < 1e+5

		# Draw white square in top left corner for text
		cv2.rectangle(self.img, (0, 0), (200, 70), (255, 255, 255), -1)

		# Add corner age to display frame
		cv2.putText(
			self.img,
			f"Corner Age: {self.corners_age}",
			(10, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.8,
			(255, 0, 0),
			2
		)

		# Add frame number to display frame
		cv2.putText(
			self.img,
			f"Frame: {self.index}",
			(10, 60),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.8,
			(255, 0, 0),
			2
		)

		# Get the homography for the current frame
		self.diagram = self.applyHomography(self.diagram)

		# Calculate scale needed (diagram width/frame width)
		scale = self.img.shape[1] / self.diagram.shape[1] / 2.5

		# Place a downsized version of the diagram in the top right corner of
		# the frame
		small_diagram = cv2.resize(self.diagram, (0, 0), fx=scale, fy=scale)

		# Add the diagram to the top right corner of the frame
		self.img[:small_diagram.shape[0], -
				 small_diagram.shape[1]:] = small_diagram

		# Add corner age to display frame
		cv2.putText(
			self.img,
			f"Corner Age: {self.corners_age}",
			(10, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.8,
			(255, 0, 0),
			2
		)

		# Add frame number to display frame
		cv2.putText(
			self.img,
			f"Frame: {self.index}",
			(10, 60),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.8,
			(255, 0, 0),
			2
		)

		# Display frame and debug frames
		if self.engine.show:
			self.displayAll()

		# Write the frame to the output video
		output.write(self.img)

		# Clear some memory
		self.clearMemory()

	def save_img(self, path: str):
		# Check if frame has 4 corners
		if len(self.corners) == 4:
			self.diagram = self.applyHomography(self.engine.diagram.copy())

			# Calculate scale needed (diagram width/frame width)
			scale = self.img.shape[1] / self.diagram.shape[1] / 2.5

			# Place a downsized version of the diagram in the top right corner
			# of the frame
			small_diagram = cv2.resize(
				self.diagram, (0, 0), fx=scale, fy=scale)

			# Add the diagram to the top right corner of the frame
			self.img[:small_diagram.shape[0], -
					 small_diagram.shape[1]:] = small_diagram

		# Save the frame
		cv2.imwrite(path, self.img)

		return self.img

	@staticmethod
	def updateCorners(frame1, frame2):
		if len(frame1.corners) == 1:
			if frame1.side == "left":
				# Compare the top right corner of frame2 (frame [1]) with the
				# corner of frame1
				movement = frame2.corners[1] - frame1.corners[0]
				frame1.corners = frame2.corners.copy()
				for i in range(len(frame1.corners)):
					frame1.corners[i] -= movement
			else:
				# Compare the top left corner of frame2 (frame [0]) with the corner
				# of frame1
				movement = frame2.corners[0] - frame1.corners[0]
				frame1.corners = frame2.corners.copy()
				for i in range(len(frame1.corners)):
					frame1.corners[i] -= movement
			frame1.diagram_corners = frame2.diagram_corners.copy()

			frame1.plotCorners()
			return

		frame1_img = frame1.orig_img.copy()
		frame2_img = frame2.orig_img.copy()

		# Apply canny edge detection to the curr and next frames
		frame1_img = cv2.Canny(frame1_img, 100, 200)
		frame2_img = cv2.Canny(frame2_img, 100, 200)

		# Calculate optical flow
		flow = cv2.calcOpticalFlowFarneback(
			frame2_img, frame1_img, None, 0.5, 10, 30, 3, 5, 1.5, 0)

		# Create a flow frame to display the flow vectors
		frame1.flow_frame = frame1.orig_img.copy()

		# Reset flow vectors around detected players and referees
		flow = frame1.remove_tracks(flow)

		# Reset top & bottom 15% of flow vectors
		flow[:int(flow.shape[0] * 0.15), :, :] = 0
		flow[int(flow.shape[0] * 0.85):, :, :] = 0

		# Calculate the average x and y movement ignoring vectors with magnitude
		# less than 1
		avg_movement = np.array([0, 0], dtype=np.float32)
		count = 1

		# Loop through each pixel in the flow
		for y in range(0, flow.shape[0], 10):
			for x in range(0, flow.shape[1], 10):
				# Get the flow vector
				fx, fy = flow[y, x]

				# If the magnitude of the flow vector is greater than 1, add it to
				# the average movement
				if np.sqrt(fx ** 2 + fy ** 2) > 1.5:
					avg_movement += np.array([fx, fy], dtype=np.float32)
					count += 1

				# Draw the flow vector in red if moving right, blue if moving left
				if fx > 0:
					cv2.line(frame1.flow_frame, (x, y),
							(x + int(fx), y + int(fy)), (0, 0, 255), 1)
				else:
					cv2.line(frame1.flow_frame, (x, y),
							(x + int(fx), y + int(fy)), (255, 0, 0), 1)

		if frame1.engine.show:
			cv2.imshow("Flow", frame1.flow_frame)

		# Divide the average movement by the number of vectors to get the average
		# movement
		avg_movement /= count

		# Take a copy of the next frame's corners and increment their age
		frame1.corners = frame2.corners.copy()
		frame1.diagram_corners = frame2.diagram_corners.copy()

		frame1.corners_age = frame2.corners_age + 1

		# Increase avg_movement proportional to the age of the corners
		# avg_movement *= (frame1.corners_age ** 1.09) * 0.001 + 1

		# Update corners with average movement
		avg_movement = avg_movement.astype(int)

		for i in range(len(frame1.corners)):
			frame1.corners[i] += avg_movement

		frame1.plotCorners()
