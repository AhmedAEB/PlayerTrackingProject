from time import perf_counter

from utils.engine import Engine
from utils.performance import performance_logger

if __name__ == "__main__":
	# Store start time
	start = perf_counter()
	performance_logger.start_timer("total_execution")

	# Initialize engine
	performance_logger.start_timer("engine_initialization")
	engine = Engine()
	performance_logger.stop_timer("engine_initialization")

	# Run tracking
	engine.track(show=True)
	
	# Run detection on image
	# img = engine.detect_image(
	# 	source="./data/input.png",
	# 	save_path="./data/output2.png",
	#   	save=True,
	# )

	# Print time taken to run
	performance_logger.stop_timer("total_execution")
	print(f"Time taken: {perf_counter()-start:.2f} s")
