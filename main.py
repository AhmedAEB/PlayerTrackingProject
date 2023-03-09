from time import perf_counter

from utils.engine import Engine

if __name__ == "__main__":
	# Store start time
	start = perf_counter()
	
	# Initialize engine
	engine = Engine()
 
	# Run tracking
	# engine.track()
 
	# Run detection on image
	# img = engine.detect_image(
	# 	source="./data/input2.png",
	# 	save_path="./data/output2.png",
	#   save=True,
	# )
	
	# Print time taken to run
	print(f"Time taken: {perf_counter()-start:.2f} s")