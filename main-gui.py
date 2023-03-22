import tkinter as tk
from time import perf_counter

from utils.engine import Engine
from utils.gui import GUI

def main():
	# Store start time
	start = perf_counter()

	# Disable button and change text to "Processing..."
	GUI.button.config(text="Processing...", state="disabled")
	GUI.root.update()

	# Run tracking using values from entries
	ENGINE.track(
		source = source.get(),
		save_path = save_path.get(),
		nframes = int(nframes.get()) if nframes.get() != "-1" else None,
		skip_frames = int(skip_frames.get()),
		show = show.get(),
		# gui = gui,
	)

	# Enable button and change text to "Run"
	GUI.button.config(text="Run", state="normal")

	# Move gui to top
	GUI.root.wm_attributes("-topmost", 1)

	# Print time taken to run
	print(f"Time taken: {perf_counter()-start:.2f} s")

if __name__ == "__main__":
	# Initialize engine
	ENGINE = Engine()

	# Initialize GUI
	GUI = GUI(main)

	# Create entries
	source = GUI.create_entry("Source", "./data/input.mp4")
	save_path = GUI.create_entry("Save Path", "./data/output.mp4")
	nframes = GUI.create_entry("Number of Frames", "-1")
	skip_frames = GUI.create_entry("Frames to Skip", "0")
	show = GUI.create_checkbox("Show Process", True)
 
	# Run GUI mainloop
	GUI.run(main)