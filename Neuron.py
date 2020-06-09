from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.colorchooser import *

global canvas
global root
global menu
global WIN_WIDTH
global WIN_HEIGHT
global DEFAULT_Y

WIN_WIDTH = 1200
WIN_HEIGHT = 600
DEFAULT_Y = 50

root = Tk()
root.minsize(WIN_WIDTH, WIN_HEIGHT)
menu = Menu(root)
canvas = Canvas(root, height=500, width=500)

canvas.pack()

coords = [50, 50, 100, 100]

class Neuron:
    # default constructor
    def __init__(self, color, x, y):
        self.node = canvas.create_oval(x, y, x + 50, y + 50, fill = color)
        canvas.tag_bind(self.node, '<Button-3>', self.circle_click)


    def set_background(self, color, event = None):
        canvas.itemconfig(self.node, fill = color)
    

    def circle_click(self, event = None):
        self.set_background('green')
    

class Layer:
    def __init__(self, x, y):
        self.CONST_X = x
        self.last_y = y
        self.color = 'black'
        self.layer = [Neuron(self.color, self.CONST_X, self.last_y)]


    def set_color(self, event = None):
        self.color = askcolor()[1]
        for i in self.layer:
            i.set_background(self.color)


    def add_neuron(self):
        self.next_location()
        self.layer.append(Neuron(self.color, self.CONST_X, self.last_y))

    def next_location(self):
        self.last_y += 100
    

class NeuralNetwork:
    def __init__(self):
        self.input = Layer(50, DEFAULT_Y)
        self.output = Layer(250, DEFAULT_Y)
        self.hidden = [Layer(150, DEFAULT_Y)]
        self.last_x = 250

        self.network = [self.input, self.output, self.hidden[0]]
        root.bind('<Return>', self.hidden[0].set_color)

    def add_layer(self):
        self.last_x += 100
        self.network.append(Layer(self.last_x, DEFAULT_Y))
        canvas.update()

    def get_network(self):
        return self.network


app = NeuralNetwork()
layer_button = Button(text = 'add', command = app.add_layer).pack()
add_neuron_button = Button(text = 'add input neuron', command = app.network[0].add_neuron).pack()
add_neuron_button = Button(text = 'add out neuron', command = app.network[1].add_neuron).pack()
add_neuron_button = Button(text = 'add hidden neuron', command = app.network[2].add_neuron).pack()
root.mainloop()