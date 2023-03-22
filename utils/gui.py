import tkinter as tk

class GUI():
	def __init__(self, callback):
		self.root = self.create_root()
  
		self.button = tk.Button(self.root, text = "Run", command = callback, highlightbackground = "white")

		self.rows = 0
  
	def create_root(self):
		root = tk.Tk()
		
		root.title("🏀 - Player Tracking Project")
  
		# Make window non-resizable
		root.resizable(False, False)
	
		# Make background white
		root.configure(background = "white")
  
		return root

	def create_entry(self, label, default):
		label = tk.Label(self.root, text = f"{label.title()}:", bg = "white")
		entry = tk.Entry(self.root, highlightthickness = 1, border = 1)
		entry.insert(0, default)
  
		label.grid(row = self.rows, column = 0, sticky = "w")
		entry.grid(row = self.rows, column = 1, sticky = "ew")
  
		self.rows += 1
  
		return entry

	def create_checkbox(self, label, default):
		var = tk.BooleanVar()
		var.set(default)
		label = tk.Label(self.root, text = f"{label.title()}:", bg = "white")
		checkbox = tk.Checkbutton(self.root, bg = "white", onvalue = True, offvalue = False, variable = var)

		label.grid(row = self.rows, column = 0, sticky = "w")
		checkbox.grid(row = self.rows, column = 1, sticky = "w")
  
		self.rows += 1
  
		return var

	def run(self, callback):  
		self.button.grid(row = self.rows, column = 0, columnspan = 2, sticky = "ew")
  
		# Place in bottom left corner of screen
		self.root.geometry(
      		"+{}+{}".format(
            	0,
                self.root.winfo_screenheight() - self.root.winfo_reqheight()
            )
        )
     
		# Run mainloop
		self.root.mainloop()
