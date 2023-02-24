import random
from os import path
import sys

import cv2

import numpy as np
import torch

from seg.utils.augmentations import letterbox
from seg.utils.torch_utils import select_device, smart_inference_mode
from seg.utils.general import Profile, check_img_size, non_max_suppression, scale_coords
from seg.models.common import DetectMultiBackend
from seg.utils.segment.general import process_mask, scale_masks
from seg.utils.segment.plots import plot_masks, colors, Annotator

class Yolov7SegWrapper():
	@smart_inference_mode()
	def __init__(self, device, weights, imgsz = (640, 640), conf_th = 0.25, iou_th = 0.45):
		self.device = select_device(device)
		self.conf_th = conf_th
		self.iou_th = iou_th
		
		self.model = DetectMultiBackend(weights, device=self.device, fp16=False)
		self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
		self.imgsz = check_img_size(imgsz, self.stride)
   
		# Batch size
		self.bs = 1
  
		# Run once
		self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))  # warmup
		self.dt = (Profile(), Profile(), Profile())
	
	@smart_inference_mode()
	def detect(self, img, draw_masks = True):
		# Make a copy of the original image
		orig = img.copy()

		img = letterbox(img, self.imgsz, stride=self.stride, auto=self.pt)[0]
		img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])
  
		with self.dt[0]:
			img = torch.from_numpy(img).to(self.device)
			img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
			img /= 255  # 0 - 255 to 0.0 - 1.0
			if len(img.shape) == 3:
				img = img[None]  # expand for batch dim

		# Inference
		with self.dt[1]:
			pred, out = self.model(img, augment=False, visualize=False)
			proto = out[1]

		# NMS
		with self.dt[2]:
			pred = non_max_suppression(pred, self.conf_th, self.iou_th, nm=32)

		dets = pred[0]

		# If no detections, return original image or None
		if not len(dets):
			# Return black image same size as original image
			return torch.zeros(orig.shape, dtype=torch.uint8).numpy()

		# Process detections and extract masks
		masks = process_mask(proto[0], dets[:, 6:], dets[:, :4], img.shape[2:], upsample=True)  # HWC
  
  		# Rescale boxes from img_size to original size
		dets[:, :4] = scale_coords(img.shape[2:], dets[:, :4], orig.shape).round()
	
		# Draw masks on image
		if draw_masks:
			# Create black image same size as img[0] using torch
			black_img = torch.zeros(img[0].shape, dtype=torch.uint8, device=self.device)
			im_masks = plot_masks(black_img, masks, [(255, 255, 255)], alpha=1)  # image with masks shape(imh,imw,3)
			final_img = scale_masks(img.shape[2:], im_masks, orig.shape)  # scale to original h, w
		
		return final_img if draw_masks else orig