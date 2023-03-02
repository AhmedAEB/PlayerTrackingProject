from ultralytics import YOLO
from utils.basketballPredictor import BasketballPredictor

import tqdm
import cv2
import torch
import argparse
import numpy as np
from time import perf_counter

from utils.court import convertMaskToPolygon, getCornerPoints

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

def run(opt):
	# Get arguments
	source, show_process, fast, skip_frames = opt.source, opt.show_process, opt.fast, opt.skip_frames

	# Find total number of frames in video
	video = cv2.VideoCapture(source)
 
	total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - skip_frames
 
	if total_frames < 0:
		raise ValueError("Video has no frames or skip_frames is too large!")

	video.release()

	print("🏀 - Loading Yolo Models...")
	if fast:
		 model = YOLO("./weights/FirstYoloV8.pt")
	else:
		model = YOLO("./weights/NinthYolo_x-nracheckpoint3.pt")
	
	segModel = YOLO("./weights/SecondSeg.pt")

	def on_predict_postprocess_end(predictor):
		if predictor.args.task == "detect":
			results = segModel.predict(
				source = predictor.results[0].orig_img,
				show = show_process,
				verbose = False,
				box = False,
				retina_masks = True,
			)

			# Display mask as a binary image
			binary = results[0].masks.masks.permute(1, 2, 0).numpy()
			binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY)[1]
   
			# Calculate the area covered by the mask
			mask_area = np.sum(binary == 255) / (binary.shape[0] * binary.shape[1])

			# If the area is too small or the mask is too close to the top/bottom of the image (by 2.5%), then ignore it
			if mask_area < 0.01 or binary[:binary.shape[0]//20, :].any() or binary[-binary.shape[0]//20:, :].any():
				predictor.results[0].corners, predictor.results[0].diagram_corners = None, None
			# Otherwise use the mask to find the corners
			else:
				# Convert the mask to a polygon with straight edges
				polygon = convertMaskToPolygon(binary.astype(np.uint8))
	
				# Extract corner points from polygon (4 points, 2 points or None)
				predictor.results[0].corners, predictor.results[0].diagram_corners = getCornerPoints(polygon)
	
				for i, corner in enumerate(predictor.results[0].corners):
					cv2.circle(predictor.results[0].orig_img, corner.astype(int), 5, COLORS[i], -1)
   
				if show_process:
					# Write mask area on binary image
					cv2.putText(binary, f"Mask area: {mask_area*100:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
		
					# Draw a white line at the top and bottom of the binary image
					cv2.line(binary, (0, binary.shape[0]//20), (binary.shape[1], binary.shape[0]//20), (255, 255, 255), 2)
					cv2.line(binary, (0, binary.shape[0] - binary.shape[0]//20), (binary.shape[1], binary.shape[0] - binary.shape[0]//20), (255, 255, 255), 2)
		
					# Display the image
					cv2.imshow("Binary", binary)
					cv2.imshow("Polygon", polygon)
	
			progress_bar.update(1)

	progress_bar = tqdm.tqdm(total = total_frames, desc = "🏀 - Processing...")

	model.add_callback("on_predict_postprocess_end", on_predict_postprocess_end)
	model.predictor = BasketballPredictor(skip_frames=skip_frames)
	for result in model.track(
		source = source,
		show = show_process,
		save = True,
		conf = 0.1,
		iou = 0.9,
		tracker = "botsort_custom.yaml",
		line_thickness = 2,
		max_det = 15, # For basketball, there is max 10 players on the court at a time, 3 refs, so 20 is a good number
		verbose = False,
	):
		print(result.corners)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", type=str, default="./data/alitest.mp4", help="Path to a basketball video")
	parser.add_argument("--show_process", action="store_true", help="Show the process of the tracking, including court detection.") 
	parser.add_argument("--fast", action="store_true", help="Run tracking at a faster speed, but with less accuracy.")
	parser.add_argument("--skip_frames", type=int, default=0, help="Skip the first n frames of the video.")
 
	opt = parser.parse_args()
	print(opt)
	
	# Store start time
	start = perf_counter()
	
	# Run tracking
	run(opt)
	
	# Print time taken to run
	print(f"Time taken: {perf_counter()-start:.2f} s")