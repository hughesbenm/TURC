from tkinter import *
from tkinter.colorchooser import *


WIN_WIDTH = 1200
WIN_HEIGHT = 600
DEFAULT_Y = 50

root = Tk()
root.minsize(WIN_WIDTH, WIN_HEIGHT)
menu = Menu(root)
canvas = Canvas(root, height=600, width=1200)

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

    def move_forward_x(self):
        for i in self.layer:
            i.move_x(50)
    
    def move_backward_x(self):
        for i in self.layer:
            i.move_x(-50)


class NeuralNetwork:
    def __init__(self):
        self.input = Layer((WIN_WIDTH/2) - 50, DEFAULT_Y)
        self.input_index = 0 #not necessary

        self.output = Layer((WIN_WIDTH/2) + 50, DEFAULT_Y)
        self.output_index = 1

        self.last_x = self.output.get_x()
        self.hidden_x = self.input.get_x() # initial value of 550
        
        self.network = [self.input, self.output]

    def add_layer(self):
        # adjust layers before output
        for i in self.network:
            if (i == self.network[self.output_index]):
                break
            i.move_backward_x()
            

        # move output layer
        self.network[self.output_index].move_forward_x() # moves layer

        # insert new layer
        self.hidden_x += 50
        self.network.insert(self.output_index, Layer(self.hidden_x, DEFAULT_Y)) # inserts hidden layer in next position (aka last output index)

        # adjust variables
        self.last_x += 50 # updates last_x which denotes where the x location of layer is
        self.output_index += 1 # updates output_index, which stores the last index of network array
        self.network[self.output_index].set_x(self.last_x) # updates x location for output layer/fixes "add out neuron"

        canvas.update()

    def get_network(self):
        return self.network

    def get_output_index(self):
        print(self.output_index)
        return self.output_index


app = NeuralNetwork()
Button(text = 'add hidden layer', command = app.add_layer).pack()
Button(text = 'add input neuron').pack()
Button(text = 'add out neuron', command = app.network[app.get_output_index()].add_neuron).pack()
Button(text = 'add hidden neuron').pack()
root.mainloop()
