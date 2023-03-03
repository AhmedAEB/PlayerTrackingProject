from ultralytics.yolo.v8.detect import DetectionPredictor
from ultralytics.yolo.data import load_inference_source
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops, DEFAULT_CFG
from ultralytics.yolo.utils.checks import check_imgsz

class BasketballResults(Results):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.corners = None
		self.diagram_corners = None
		self.prev_result = None

class BasketballPredictor(DetectionPredictor):
	def __init__(self, cfg=DEFAULT_CFG, overrides=None, skip_frames=0):
		super().__init__(cfg, overrides)
		self.skip_frames = skip_frames
	
	def setup_source(self, source):
		self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
		if self.args.task == 'classify':
			transforms = getattr(self.model.model, 'transforms', classify_transforms(self.imgsz[0]))
		else:  # predict, segment
			transforms = None
		self.dataset = load_inference_source(source=source,
											 transforms=transforms,
											 imgsz=self.imgsz,
											 vid_stride=self.args.vid_stride,
											 stride=self.model.stride,
											 auto=self.model.pt)
  
		#* CUSTOM CODE
  
		# Skip frames
		for batch in self.dataset:
			if self.skip_frames > 0:
				self.skip_frames -= 1
				continue
			else:
				break

		#* END OF CUSTOM CODE

		self.source_type = self.dataset.source_type
		self.vid_path, self.vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs
  
	def postprocess(self, preds, img, orig_img):
		preds = ops.non_max_suppression(preds,
										self.args.conf,
										self.args.iou,
										agnostic=self.args.agnostic_nms,
										max_det=self.args.max_det,
										classes=self.args.classes)

		results = []
		for i, pred in enumerate(preds):
			orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
			shape = orig_img.shape
			pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
			path, _, _, _, _ = self.batch
			img_path = path[i] if isinstance(path, list) else path
			results.append(BasketballResults(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
		return results
