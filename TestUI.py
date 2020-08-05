from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.colorchooser import *
import numpy as np
import math as math
from operator import add
from operator import sub
from operator import truediv
from operator import mul

root = Tk()
root.minsize(1200, 600)
canvas = Canvas(root)
canvas.place(anchor = CENTER, relheight = .95, relwidth = 0.95, relx = 0.5, rely = 0.5)

input_size = (3, 3, 4)

MAX_WIDTH = 150

pixels_per_unit = MAX_WIDTH / input_size[2]

class Prism:
    ANGLE = 45 / 360 * 2 * math.pi
    ANGLE_CONSTANT = math.sin(ANGLE)

    def __init__(self, center, input_size):
        self.input_size = input_size

        # Width of the Prism
        self.x = float(input_size[0] * pixels_per_unit)

        # Points Right and Up
        self.y_vector = [float(input_size[1] * math.cos(Prism.ANGLE)) * pixels_per_unit, -float(input_size[1] * math.sin(Prism.ANGLE) * pixels_per_unit)]

        # Height of the Cube
        self.z = float(input_size[2] * pixels_per_unit)

        self.front_top_left = list(map(add, list(map(add, center, [-self.x / 2, -self.z / 2])),
                                       list(map(truediv, self.y_vector, [2, 2]))))
        self.front_top_right = list(map(add, self.front_top_left, [self.x, 0]))
        self.front_bottom_left = list(map(add, self.front_top_left, [0, self.z]))
        self.front_bottom_right = list(map(add, self.front_bottom_left, [self.x, 0]))
        self.back_top_left = list(map(add, self.front_top_left, self.y_vector))
        self.back_top_right = list(map(add, self.front_top_right, self.y_vector))
        self.back_bottom_right = list(map(add, self.front_bottom_right, self.y_vector))
        self.front = canvas.create_polygon([self.front_top_left, self.front_top_right, self.front_bottom_right, self.front_bottom_left],
                                         fill = 'black', outline = 'green', width = 1)
        self.top = canvas.create_polygon([self.front_top_left, self.back_top_left, self.back_top_right, self.front_top_right],
                                         fill = 'black', outline = 'green', width = 1)
        self.right = canvas.create_polygon([self.front_top_right, self.back_top_right, self.back_bottom_right, self.front_bottom_right],
                                         fill = 'black', outline = 'green', width = 1)


Prism([200, 200], [4, 4, 5])
Prism([500, 200], [3, 2, 3])
root.mainloop()
