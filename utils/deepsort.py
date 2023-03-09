from deepsort.detection import Detection
from deepsort.generate_detections import create_box_encoder
from deepsort.nn_matching import NearestNeighborDistanceMetric
from deepsort.track import Track
from deepsort.tracker import Tracker

class DeepSort():
	def __init__(self, deepsort_weights, cosine_thresh, nn_budget, max_track_age, init_phase_frames):
		self.encoder = create_box_encoder(deepsort_weights, batch_size=1)
		self.tracker = Tracker(NearestNeighborDistanceMetric("euclidean", cosine_thresh, nn_budget), max_age=max_track_age, n_init=init_phase_frames)

	def update(self, frame, bboxes, scores):
		tracks = []
    
		feats = self.encoder(frame, bboxes)
		dets = [Detection(*args) for args in zip(bboxes, scores, feats)]

		self.tracker.predict()
		self.tracker.update(dets)
  
		for track in self.tracker.tracks:
			# Skip tracks that are not confirmed or have not been updated for a long time
			if not track.is_confirmed() or track.time_since_update > 1:
				continue

			tracks.append(track)
   
		return tracks