from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.colorchooser import *
from main import *

COORDS = 50, 50, 100, 100
class Neuron:
    # default constructor
    def __init__(self):
        canvas.create_oval(COORDS)


    def set_background(self, event = None):
        color = askcolor()
        canvas.itemconfig(self.node, fill = color)
    

    def circle_click(self, event = None):
        set_background(self)

