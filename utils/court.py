# Utility Functions to help with detecting and tracking the basketball court
import numpy as np
import cv2

def convertMaskToPolygon(mask):
	# Find contours
	contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	# Get the largest contour
	largest_contour = max(contours, key=cv2.contourArea)
	
	# Approximate the contour
	epsilon = 0.01 * cv2.arcLength(largest_contour, True)

	# Get the polygon
	polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

	# Draw the polygon on a blank image
	out = np.zeros_like(mask)
	cv2.drawContours(out, [polygon], 0, 255, -1)
 
	return out
 
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
	corners = np.array(corners, dtype=np.float32)

	# Put the corners in the correct order (top left, top right, bottom left, bottom right)
	corners = corners.reshape(-1, 2)
	
	# Order the corners by their y coordinate
	corners = corners[corners[:, 1].argsort()]

	# Order the top corners by their x coordinate
	corners[:2] = corners[:2][corners[:2, 0].argsort()]

	# Order the bottom corners by their x coordinate
	corners[2:] = corners[2:][corners[2:, 0].argsort()]

	# Compare top left and bottom left x coordinates
	if corners[0, 0] > corners[2, 0]:
		# (53, 470) (424, 470) (53, 780) (424, 780)
		diagram_corners = np.array([(53, 470), (424, 470), (53, 780), (424, 780)], dtype=np.float32)
	else:
		# (1521, 473) (1890, 473) (1521, 782) (1890, 782)
		diagram_corners = np.array([(1521, 473), (1890, 473), (1521, 782), (1890, 782)], dtype=np.float32)

	return corners, diagram_corners

def calcCameraMovement(updateFrame, refFrame):
	# Make a copy of the curr and next frames
	updateFrame_img = updateFrame.copy()
	refFrame_img = refFrame.copy()
 
	# Apply canny edge detection to the curr and next frames
	updateFrame_img = cv2.Canny(updateFrame_img, 100, 200)
	refFrame_img = cv2.Canny(refFrame_img, 100, 200)
 
	# Calculate optical flow
	flow = cv2.calcOpticalFlowFarneback(refFrame_img, updateFrame_img, None, 0.5, 10, 30, 3, 5, 1.5, 0)
 
	# Create a flow frame to display the flow vectors
	flow_frame = updateFrame.copy()
 
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
				cv2.line(flow_frame, (x, y), (x + int(fx), y + int(fy)), (0, 0, 255), 1)
			else:
				cv2.line(flow_frame, (x, y), (x + int(fx), y + int(fy)), (255, 0, 0), 1)

	# Divide the average movement by the number of vectors to get the average movement
	avg_movement /= count
	
	return avg_movement