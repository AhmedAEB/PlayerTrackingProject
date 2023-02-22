import random

import numpy as np
import torch

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device

class Yolov7Wrapper():
	def __init__(self, device, weights, imgsz = 640, conf_th = 0.25, iou_th = 0.45):
		self.device = select_device(device)
		self.conf_th = conf_th
		self.iou_th = iou_th
  
		self.half = self.device.type != "cpu"
		
		self.model = attempt_load(weights, map_location=self.device)
		self.stride = int(self.model.stride.max())
		self.imgsz = check_img_size(imgsz, self.stride)

		if self.half:
			self.model.half()
   
		# Get names and colors
		self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
		self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

		if self.device.type != "cpu":  # Run once before inference
			self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))

	@torch.no_grad()
	def detect(self, img, draw_bboxes = True):
		# Make a copy of the original image
		orig = img.copy()
	 
		# From yolov7/detect.py
		img = letterbox(img, self.imgsz, stride=self.stride, scaleup=False)[0]
		img = np.ascontiguousarray(img.transpose(2, 0, 1))
		img = torch.from_numpy(img).to(self.device)
		img = img.half() if self.half else img.float()  # uint8 to fp16/32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		# Run inference and non-maximum suppression
		pred = self.model(img, augment=False)[0]
		pred = non_max_suppression(pred, self.conf_th, self.iou_th)

		# Extract detections
		dets = pred[0]

		# If no detections, return original image or None
		if not len(dets):
			return orig if draw_bboxes else None

		# Rescale boxes from img_size to im0 size
		dets[:, :4] = scale_coords(img.shape[2:], dets[:, :4], orig.shape).round()
  
		if draw_bboxes:
			# From yolov7/detect.py
			for *xyxy, conf, cls in reversed(dets):
				label = f"{self.names[int(cls)]} {conf:.2f}"
				plot_one_box(xyxy, orig, self.colors[int(cls)], label, line_thickness=1)
    
		# Return detections or original image
		return orig if draw_bboxes else dets.detach().cpu().numpy()