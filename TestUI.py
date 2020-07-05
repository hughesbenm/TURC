from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.colorchooser import *

root = Tk()
root.minsize(200, 200)

frame = Frame(root, bg = 'green', height = 200, width = 200)
frame.pack(expand = True, fill = BOTH)
frame_0_0 = Frame(frame, bg = 'red', height = 50, width = 50)
frame_0_0.grid(row = 0, column = 0, sticky = W)
frame_0_1 = Frame(frame, bg = 'purple', height = 50, width = 50)
frame_0_1.grid(row = 0, column = 1, sticky = W)
frame_1_012 = Frame(frame, bg = 'blue', height = 50, width = 150)
frame_1_012.grid(row = 1, column = 0, columnspan = 3, sticky = W)
frame.columnconfigure(2, weight = 1)
# frame_0_2 = Frame(frame, bg = 'black', height = 50, width = 50)
# frame_0_2.grid(row = 0, column = 2, sticky = W)

root.mainloop()
