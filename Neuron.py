from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.colorchooser import *

global canvas
global root
global menu

root = Tk()
menu = Menu(root)
canvas = Canvas(root, height=500, width=500)
canvas.pack()

coords = [50, 50, 100, 100]

class Neuron:
    # default constructor
    def __init__(self):
        self.node = canvas.create_oval(coords, fill = 'black')
        canvas.tag_bind(self.node, '<Button-3>', self.circle_click)


    def set_background(self, event = None):
        color = askcolor()
        canvas.itemconfig(self.node, fill = color[1])
    

    def circle_click(self, event = None):
        self.set_background()
    

class Layer:
    def __init__(self):
        self.layer = [Neuron()]
        self.next_location()


    def set_color(self, color):
        for i in self.layer:
            i.set_background(color)
    

    def add_neuron(self):
        self.layer.append(Neuron())
    
    def next_location(self):
        coords[1] += 60
        coords[3] += 60
    

class NeuralNetwork:
    def __init__(self):
        self.network = [Layer()]

    
    def add_layer(self):
        self.network.append(Layer())
        canvas.update()


    def get_network(self):
        return self.network


app = NeuralNetwork()
test = Button(text = 'add', command = app.add_layer).pack()
root.mainloop()