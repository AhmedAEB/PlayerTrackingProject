import cv2

class BasketballAnnotater():
	def __init__(self, source, save_path):
		# Get video writer
		video = cv2.VideoCapture(source)
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		fps = video.get(cv2.CAP_PROP_FPS)
		size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  
		self.frame_count = frame_count
		self.output = cv2.VideoWriter(save_path, fourcc, fps, size)
  
		video.release()

	def write(self, result):
		self.output.write(result.orig_img)