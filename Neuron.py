from tkinter import *
from tkinter.colorchooser import *


WIN_WIDTH = 1200
WIN_HEIGHT = 600
CAN_WIDTH = 1100
CAN_HEIGHT = 500
DEFAULT_Y = 50

root = Tk()
root.minsize(WIN_WIDTH, WIN_HEIGHT)
menu = Menu(root)
canvas = Canvas(root, height=CAN_HEIGHT, width=CAN_WIDTH)

canvas.pack()


def coords(x, y, radius):
    return x - radius, y - radius, x + radius, y + radius


class Neuron:
    # default constructor
    def __init__(self, color, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.node = canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill = color)

    def set_background(self, color):
        canvas.itemconfig(self.node, fill = color)

    def get_tag(self):
        return self.node

    def move_x(self, x):
        canvas.move(self.node, x, 0)


class Layer:
    def __init__(self, x, y, color='black'):
        self.CONST_X = x
        self.y_interval = y
        self.color = color
        self.layer = [Neuron(self.color, self.CONST_X, self.y_interval, 25)]
        self.num_neurons = 1
        canvas.tag_bind(self.layer[0].get_tag(), '<Button-3>', self.set_color)

    def set_color(self, event = None):
        self.color = askcolor()[1]
        for i in self.layer:
            i.set_background(self.color)

    def add_neuron(self):
        self.layer.append(Neuron(self.color, self.CONST_X, self.y_interval, 25))
        self.num_neurons += 1
        self.y_interval = CAN_HEIGHT / (self.num_neurons + 1)
        for i in range(self.num_neurons):
            canvas.coords(self.layer[i].get_tag(), coords(self.CONST_X, self.y_interval * (i + 1), 25))
        canvas.tag_bind(self.layer[self.num_neurons - 1].get_tag(), '<Button-3>', self.set_color)

    def next_location(self):
        self.y_interval += self.y_interval

    def get_x(self):
        return self.CONST_X

    def set_x(self, x):
        self.CONST_X = x

    def move_forward_x(self):
        for i in self.layer:
            i.move_x(50)
            self.CONST_X += 50

    def move_backward_x(self):
        for i in self.layer:
            i.move_x(-50)
            self.CONST_X -= 50

    def move_x_num(self, x):
        for i in self.layer:
            i.move_x(x)
            self.CONST_X += x


class NeuralNetwork:
    def __init__(self):
        self.input = Layer((WIN_WIDTH/2) - 75, DEFAULT_Y)
        self.input_index = 0 #not necessary

        self.output = Layer((WIN_WIDTH/2) + 25, DEFAULT_Y)
        self.output_index = 1

        self.last_x = self.output.get_x()
        self.hidden_x = self.input.get_x() # initial value of 550

        self.network = [self.input, self.output]

    def add_layer(self):
        if ((self.output_index + 1) < 12):
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

        else:
            width = (self.network[self.output_index].get_x() - self.network[0].get_x())
            spacing = width / (self.output_index + 1)
            for i in range(1, len(self.network) - 1):
                self.network[i].move_x_num(self.network[0].get_x() + (spacing * i) - self.network[i].get_x())
            self.network.insert(self.output_index, Layer(self.network[0].get_x() + (spacing * self.output_index), DEFAULT_Y))
            self.output_index += 1

        canvas.update()

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
