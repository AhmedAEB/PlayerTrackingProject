import tkinter as tk
from time import perf_counter

from utils.engine import Engine

# Run detection on image
# img = engine.detect_image(
# 	source="./data/input2.png",
# 	save_path="./data/output2.png",
#   save=True,
# )

# Initialize engine
engine = Engine(
    show=True,
)


def run():
    # Store start time
    start = perf_counter()

    # Get values from entries
    source = source_entry.get()
    save_path = save_path_entry.get()
    nframes = int(nframes_entry.get())
    nframes = nframes if nframes != -1 else None
    skip_frames = int(skip_frames_entry.get())

    # Disable button and change text to "Processing..."
    run_button.config(text="Processing...", state="disabled")
    gui.update()

    # Run tracking
    engine.track(
        source=source,
        save_path=save_path,
        nframes=nframes,
        skip_frames=skip_frames,
        show=show.get(),
        # gui = gui,
    )

    # Enable button and change text to "Run"
    run_button.config(text="Run", state="normal")

    # Move gui to top
    gui.wm_attributes("-topmost", 1)

    # Print time taken to run
    print(f"Time taken: {perf_counter()-start:.2f} s")


gui = tk.Tk()
gui.title("🏀 - Player Tracking Project")

# Make window non-resizable
gui.resizable(False, False)

# Make background white
gui.configure(background="white")

# Create widgets
source_label = tk.Label(gui, text="Source:", bg="white")
source_entry = tk.Entry(gui, highlightthickness=1, border=1)
source_entry.insert(0, "./data/input.mp4")
save_path_label = tk.Label(gui, text="Save Path:", bg="white")
save_path_entry = tk.Entry(gui, highlightthickness=1, border=1)
save_path_entry.insert(0, "./data/output.mp4")
nframes_label = tk.Label(gui, text="Number of Frames:", bg="white")
nframes_entry = tk.Entry(gui, highlightthickness=1, border=1)
nframes_entry.insert(0, "-1")
skip_frames_label = tk.Label(gui, text="Frames to Skip:", bg="white")
skip_frames_entry = tk.Entry(gui, highlightthickness=1, border=1)
skip_frames_entry.insert(0, "0")

# Create a checkbox to show the output
show = tk.BooleanVar()
show_label = tk.Label(gui, text="Show Process:", bg="white")
show_checkbox = tk.Checkbutton(
    gui,
    bg="white",
    onvalue=True,
    offvalue=False,
    variable=show)

run_button = tk.Button(
    gui,
    text="Run",
    command=run,
    highlightbackground="white")

# Place widgets
source_label.grid(row=0, column=0)
source_entry.grid(row=0, column=1)
save_path_label.grid(row=1, column=0)
save_path_entry.grid(row=1, column=1)
nframes_label.grid(row=2, column=0)
nframes_entry.grid(row=2, column=1)
skip_frames_label.grid(row=3, column=0)
skip_frames_entry.grid(row=3, column=1)
show_label.grid(row=4, column=0)
show_checkbox.grid(row=4, column=1, sticky="w")
run_button.grid(row=5, column=0, columnspan=2)

# Make window appear in bottom right corner of screen
gui.geometry(
    f"+{gui.winfo_screenwidth() - (gui.winfo_reqwidth() * 2)}+{gui.winfo_screenheight() - gui.winfo_reqheight()}")

# Run mainloop
gui.mainloop()
