# Utility Functions to help with detecting and tracking the basketball court
import numpy as np
import cv2

def convertMaskToPolygon(mask):
	print(mask.dtype)
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

