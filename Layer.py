from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.colorchooser import *
import numpy as np
from main import *
import Neuron

class Layer:
    def __init__(self):
        layer = [Neuron()]