from ultralytics import YOLO
from utils.basketballPredictor import BasketballPredictor

import tqdm
import cv2
import torch
import argparse
import numpy as np
from time import perf_counter

from utils.court import convertMaskToPolygon, getCornerPoints
from utils.data import BasketballAnnotater

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

def run(opt):
	# Get arguments
	source, save_path, fast = opt.source, opt.save_path, opt.fast
 
	# Debugging options
	show_process, skip_frames = opt.show_process, opt.skip_frames

	# Initialize video writer
	annotater = BasketballAnnotater(source, save_path)
 
	# Calculate total frames including skipped frames
	total_frames = annotater.frame_count - skip_frames - 1
 
	# Check if there are any frames left
	if total_frames < 0:
		raise ValueError("Video has no frames or skip_frames is too large!")

	print("🏀 - Loading Yolo Models...")
	if fast:
		 model = YOLO("./weights/FirstYoloV8.pt")
	else:
		model = YOLO("./weights/NinthYolo_x-nracheckpoint3.pt")
	
	segModel = YOLO("./weights/SecondSeg.pt")
 
	prev_result = None

	def on_predict_postprocess_end(predictor):
		nonlocal prev_result
		if predictor.args.task == "detect":
			predictor.results[0].prev_result = prev_result
			prev_result = predictor.results[0]

			progress_bar.update(1)
   
			results = segModel.predict(
				source = predictor.results[0].orig_img,
				show = show_process,
				verbose = False,
				box = False,
				retina_masks = True,
			)
   
			if results[0].masks is None:
				predictor.results[0].corners, predictor.results[0].diagram_corners = None, None
				return

			# Display mask as a binary image
			binary = results[0].masks.masks[0].numpy().astype(np.uint8)
   
			# Change 1s to 255s (for display purposes)
			binary[binary == 1] = 255
   
			# Calculate the area covered by the mask
			mask_area = np.sum(binary == 255) / (binary.shape[0] * binary.shape[1])

			# If the area is too small or the mask is too close to the top/bottom of the image (by 2.5%), then ignore it
			if mask_area < 0.01 or binary[:binary.shape[0]//20, :].any() or binary[-binary.shape[0]//20:, :].any():
				predictor.results[0].corners, predictor.results[0].diagram_corners = None, None
				return
			
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

	progress_bar = tqdm.tqdm(total = total_frames, desc = "🏀 - Processing...")

	model.add_callback("on_predict_postprocess_end", on_predict_postprocess_end)
	model.predictor = BasketballPredictor(skip_frames=skip_frames)
	results = model.track(
		source = source,
		show = show_process,
		conf = 0.1,
		iou = 0.9,
		tracker = "botsort_custom.yaml",
		line_thickness = 2,
		max_det = 15, # For basketball, there is max 10 players on the court at a time, 3 refs, so 20 is a good number
		verbose = False
  	)
 
	# TODO: Loop through results and update corners + apply homography (both forward and backward)
	for result in results[::-1]:
		annotater.write(result)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", type=str, default="./data/alitest.mp4", help="Path to a basketball video")
	parser.add_argument("--save_path", type=str, default="./data/output.mp4", help="Path to save the output video")
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