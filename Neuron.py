from tkinter import *
from tkinter.colorchooser import *


WIN_WIDTH = 1200
WIN_HEIGHT = 600
CAN_WIDTH = 1100
CAN_HEIGHT = 500
DEFAULT_Y = 50


root = Tk()
root.minsize(WIN_WIDTH, WIN_HEIGHT)
root.config()
menu = Menu(root)
canvas = Canvas(root, height=CAN_HEIGHT, width=CAN_WIDTH)

canvas.pack()
canvas.place(anchor = CENTER, relheight = .95, relwidth = 0.95, relx = 0.5, rely = 0.5)
canvas.update()


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
        self.y_interval = canvas.winfo_height() / (self.num_neurons + 1)
        for i in range(self.num_neurons):
            canvas.coords(self.layer[i].get_tag(), coords(self.CONST_X, self.y_interval * (i + 1), 25))
        canvas.tag_bind(self.layer[self.num_neurons - 1].get_tag(), '<Button-3>', self.set_color)

    def next_location(self):
        self.y_interval += self.y_interval


class NeuralNetwork:
    def __init__(self):
        self.input = Layer(canvas.winfo_width() / 4, canvas.winfo_height() / 2, 'red')
        self.hidden = [Layer(canvas.winfo_width() / 2, canvas.winfo_height() / 2)]
        self.output = Layer(canvas.winfo_width() / 4 * 3, canvas.winfo_height() / 2, 'blue')
        self.last_x = WIN_WIDTH / 4 * 3

        self.input.add_neuron()
        self.hidden[0].add_neuron()
        self.hidden[0].add_neuron()

        self.network = [self.input, self.output, self.hidden[0]]

    def add_layer(self):
        self.last_x += WIN_WIDTH
        self.network.append(Layer(self.last_x, DEFAULT_Y))
        canvas.update()

    def get_network(self):
        return self.network

    def popup_settings(self):
        settings = Tk()
        settings.minsize(500, 500)
        buttons = Frame(settings, width=200, height=500, bg='red')
        buttons.pack(side=BOTTOM, fill = BOTH)
        settings.focus_force()
        buttons.grid_columnconfigure(0, weight = 1)
        butt1 = Button(buttons, text = 'Apply', command=settings.destroy)
        butt1.grid(column = 1, row = 0, padx = 10, pady = 5)
        butt2 = Button(buttons, text='Close', command=settings.destroy)
        butt2.grid(column = 2, row = 0, padx = 10, pady = 5)
        settings.mainloop()


app = NeuralNetwork()
root.update_idletasks()
menu.add_command(label="Settings", command=app.popup_settings)
root.config(menu=menu)
Button(canvas, text = 'add', command = app.add_layer).pack(side = BOTTOM)
Button(canvas, text = 'add input neuron', command = app.network[0].add_neuron).pack(side = BOTTOM)
Button(canvas, text = 'add out neuron', command = app.network[1].add_neuron).pack(side = BOTTOM)
Button(canvas, text = 'add hidden neuron', command = app.network[2].add_neuron).pack(side = BOTTOM)
root.mainloop()
