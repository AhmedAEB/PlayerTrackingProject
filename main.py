from time import perf_counter

from utils.engine import Engine

if __name__ == "__main__":
	# Store start time
	start = perf_counter()
	
	# Initialize engine
	engine = Engine()
	
	# Run tracking
	engine.track()
	
	# Print time taken to run
	print(f"Time taken: {perf_counter()-start:.2f} s")