from tkinter import *
from tkinter.colorchooser import *


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

    def set_background(self, color):
        canvas.itemconfig(self.node, fill = color)

    def get_tag(self):
        return self.node
    
    def move_x(self, x):
        canvas.move(self.node, x, 0)


class Layer:
    def __init__(self, x, y):
        self.CONST_X = x
        self.last_y = y
        self.color = 'black'
        self.layer = [Neuron(self.color, self.CONST_X, self.last_y)]
        self.num_neurons = 1
        canvas.tag_bind(self.layer[0].get_tag(), '<Button-3>', self.set_color)


    def set_color(self, event = None):
        self.color = askcolor()[1]
        for i in self.layer:
            i.set_background(self.color)


    def add_neuron(self):
        self.num_neurons += 1
        self.next_location()
        self.layer.append(Neuron(self.color, self.CONST_X, self.last_y))
        canvas.tag_bind(self.layer[self.num_neurons - 1].get_tag(), '<Button-3>', self.set_color)


    def next_location(self):
        self.last_y += 100
    

    def get_x(self):
        return self.CONST_X

    def set_x(self, x):
        self.CONST_X = x

    def move_x(self):
        for i in self.layer:
            i.move_x(100)



class NeuralNetwork:
    def __init__(self):
        self.input = Layer(50, DEFAULT_Y)
        self.input_index = 0 #not necessary

        self.output = Layer(250, DEFAULT_Y)
        self.output_index = 1

        self.last_x = 250
        
        self.network = [self.input, self.output]

    def add_layer(self):
        self.last_x += 100
        self.network.append(self.network[self.output_index])
        # self.network[self.output_index] = Layer(self.last_x, DEFAULT_Y)
        self.update_output()
        self.output_index += 1
        self.network[self.output_index].set_x(self.last_x)
        canvas.update()
            
    def update_output(self):
        self.network[self.output_index].move_x()

    def get_network(self):
        return self.network

    def get_output_index(self):
        return self.output_index


app = NeuralNetwork()
Button(text = 'add hidden layer', command = app.add_layer).pack()
Button(text = 'add input neuron').pack()
Button(text = 'add out neuron', command = app.network[app.get_output_index()].add_neuron).pack()
Button(text = 'add hidden neuron').pack()
root.mainloop()
