from deepsort.detection import Detection
from deepsort.generate_detections import create_box_encoder
from deepsort.nn_matching import NearestNeighborDistanceMetric
from deepsort.track import Track
from deepsort.tracker import Tracker

class BTrack(Track):
	def __init__(self, mean, covariance, track_id, n_init, max_age, feature):
		super().__init__(mean, covariance, track_id, n_init, max_age, feature)

  		# Add a new attribute to the track class
		self.team = 0
		self.prev_teams = []
  
	@staticmethod
	def fake_track(xyxy, track_id, prev_teams = []):
		track = BTrack(
			mean=None,
			covariance=None,
			track_id=track_id,
			n_init=1,
			max_age=1,
			feature=None
		)
     
		track.to_tlbr = lambda: xyxy
		track.prev_teams = prev_teams
  
		return track


class BTracker(Tracker):
	def _initiate_track(self, detection):
		mean, covariance = self.kf.initiate(detection.to_xyah())
		self.tracks.append(
			BTrack( # Change the track class to BTrack
				mean, 
			 	covariance,
			  	self._next_id,
			   	self.n_init,
				self.max_age,
				detection.feature,
			)
		)
		self._next_id += 1
		

class DeepSort():
	def __init__(
		self,
		deepsort_weights,
		cosine_thresh,
		max_iou_distance,
		nn_budget,
		max_age,
		init_phase_frames
	):
		self.encoder = create_box_encoder(deepsort_weights, batch_size=1)
		self.tracker = BTracker(
			NearestNeighborDistanceMetric(
				"euclidean",
				cosine_thresh,
				nn_budget
			),
			max_iou_distance=max_iou_distance,
			max_age=max_age,
			n_init=init_phase_frames
		)

	def update(self, frame, bboxes, scores):
		tracks = []

		feats = self.encoder(frame, bboxes)
		dets = [Detection(*args) for args in zip(bboxes, scores, feats)]

		self.tracker.predict()
		self.tracker.update(dets)

		for track in self.tracker.tracks:
			# Skip tracks that are not confirmed or have not been updated for a
			# long time
			if not track.is_confirmed() or track.time_since_update > 3: # TODO: mess with this value
				continue

			tracks.append(track)

		return tracks
	

	
