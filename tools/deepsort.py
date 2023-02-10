import random

from deepsort.deep_sort.detection import Detection
from deepsort.tools.generate_detections import create_box_encoder
from deepsort.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deepsort.deep_sort.track import Track
from deepsort.deep_sort.tracker import Tracker

# Use yolov7.utils.plots to draw bboxes
from yolov7.utils.plots import plot_one_box

class DeepSortWrapper():
	def __init__(self, deepsort_weights, cosine_thresh, nn_budget, max_track_age, class_name):
		self.encoder = create_box_encoder(deepsort_weights, batch_size=1)
		self.tracker = Tracker(NearestNeighborDistanceMetric("cosine", cosine_thresh, nn_budget), max_age=max_track_age)
		self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(1000)]
		self.class_name = class_name

	def update(self, frame, bboxes, scores):
		feats = self.encoder(frame, bboxes)
		dets = [Detection(*args) for args in zip(bboxes, scores, feats)]

		self.tracker.predict()
		self.tracker.update(dets)
  
		for track in self.tracker.tracks:
			if not track.is_confirmed() or track.time_since_update > 1:
				continue
			
			xyxy = track.to_tlbr().astype(int)
   
			label = f"{track.track_id} : {self.class_name.title()}"

			plot_one_box(xyxy, frame, label=label, color=self.colors[track.track_id], line_thickness=2)