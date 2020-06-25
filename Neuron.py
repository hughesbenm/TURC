from tkinter import *
from tkinter.colorchooser import *

# Main window's dimensions
WIN_WIDTH = 1200
WIN_HEIGHT = 600

# Main Canvas's dimensions, not sure if working
CAN_WIDTH = 1100
CAN_HEIGHT = 500

# Create the standard window
root = Tk()
root.minsize(WIN_WIDTH, WIN_HEIGHT)

# Create and add the menu bar at the top of the window
menu = Menu(root)
root.config(menu=menu)

# Create and add the canvas that takes up the entire main window
canvas = Canvas(root, height=CAN_HEIGHT, width=CAN_WIDTH)
canvas.pack()
canvas.place(anchor = CENTER, relheight = .95, relwidth = 0.95, relx = 0.5, rely = 0.5)

# Necessary for canvas.winfo_ATTRIBUTE to be updated, see NeuralNetwork.__init__() and Layer.orient_neurons()
canvas.update()

# Importing images for various buttons and things
up_arrow = PhotoImage(master = root, file = r"C:\Users\Ben\Desktop\Pics\Up.png")
down_arrow = PhotoImage(master = root, file = r"C:\Users\Ben\Desktop\Pics\Down.png")


# Simple function to turn (center_x, center_y, radius) into (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
def coords(x, y, radius):
    return x - radius, y - radius, x + radius, y + radius


# Most basic unit of the net, used mostly for displaying the logical neurons than any actual functionality
class Neuron:
    # default constructor, creates a separate oval for each node
    def __init__(self, color, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.node = canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill = color)

    # Set the color of this node, should always be the same for an entire layer
    def set_background(self, color):
        canvas.itemconfig(self.node, fill = color)

    # Returns the tag of the created oval for the individual node, allows for Layer to change things
    def get_tag(self):
        return self.node


# Collection of Neurons
class Layer:
    # default constructor, each Layer starts with a single neuron
    def __init__(self, x, y, color='black'):
        self.CONST_X = x
        self.y_interval = y
        self.color = color
        self.layer = [Neuron(self.color, self.CONST_X, self.y_interval, 25)]
        self.num_neurons = 1
        self.desired_neurons = 1
        canvas.tag_bind(self.layer[0].get_tag(), '<Button-3>', self.popup_layer_settings)

    # Runs when "Apply" is clicked in a layer's settings
    # Changes the layer based on changes made in the settings menu
    def apply_layer(self, settings, num_neurons_var):
        self.desired_neurons = num_neurons_var.get()
        settings.destroy()
        if self.desired_neurons > self.num_neurons:
            for i in range(self.desired_neurons - self.num_neurons):
                self.layer.append(Neuron(self.color, self.CONST_X, self.y_interval, 25))
                canvas.tag_bind(self.layer[-1].get_tag(), '<Button-3>', self.popup_layer_settings)
        if self.desired_neurons < self.num_neurons:
            for i in range(self.num_neurons - self.desired_neurons):
                canvas.delete(self.layer.pop().get_tag())
        self.num_neurons = self.desired_neurons
        self.orient_neurons()

    # Runs when "Close" is clicked in a layer's settings
    # Disregards any and all changes made in the settings menu
    def close_layer(self, settings, num_neurons_var):
        settings.destroy()
        num_neurons_var.set(self.num_neurons)
        self.desired_neurons = self.num_neurons

    # Set a layer's color
    def set_color(self, event = None):
        self.color = askcolor()[1]
        for i in self.layer:
            i.set_background(self.color)

    # Increase the number of neurons in the layer by one, bind it to settings on right click, reorient the layer
    def add_neuron(self):
        self.layer.append(Neuron(self.color, self.CONST_X, self.y_interval, 25))
        self.num_neurons += 1
        self.desired_neurons += 1
        canvas.tag_bind(self.layer[self.num_neurons - 1].get_tag(), '<Button-3>', self.popup_layer_settings)
        self.orient_neurons()

    # Decrease the number of neurons in the layer by one, reorient the layer
    def subtract_neuron(self):
        canvas.delete(self.layer.pop().get_tag())
        self.num_neurons -= 1
        self.desired_neurons -= 1
        self.orient_neurons()

    # Spread the neurons evenly throughout the layer
    def orient_neurons(self):
        self.y_interval = canvas.winfo_height() / (self.num_neurons + 1)
        for i in range(self.num_neurons):
            canvas.coords(self.layer[i].get_tag(), coords(self.CONST_X, self.y_interval * (i + 1), 25))

    # Increase the number of desired neurons by one, only called by settings menu
    def add_desired(self, num_neurons_var):
        self.desired_neurons += 1
        num_neurons_var.set(self.desired_neurons)

    # Decrease the number of desired neurons by one, only called by settings menu, bounce back up to 1 if below 0
    def subtract_desired(self, num_neurons_var):
        self.desired_neurons -= 1
        if self.desired_neurons <= 0:
            self.desired_neurons = 1
        num_neurons_var.set(self.desired_neurons)

    # Code for the layer settings menu that appears on right clicking a neuron
    def popup_layer_settings(self, event = None):
        settings = Toplevel()
        settings.focus_force()
        settings.minsize(500, 500)

        sett_frame = Frame(settings, width = 200, height = 500)
        sett_frame.pack(side = TOP, fill = BOTH)

        num_neurons_label = Label(sett_frame, text = 'Number of Neurons')
        num_neurons_var = IntVar(settings, self.desired_neurons)
        # add_neuron = Button(sett_frame, text='\u22C0', command = lambda: self.add_desired(num_neurons_var))
        # subtract_neuron = Button(sett_frame, text='\u22C1', command = lambda: self.subtract_desired(num_neurons_var))
        add_neuron = Button(sett_frame, image = up_arrow, command = lambda: self.add_desired(num_neurons_var), height = 10)
        subtract_neuron = Button(sett_frame, image = down_arrow, command = lambda: self.subtract_desired(num_neurons_var), height = 10)
        num_neurons_entry = Entry(sett_frame, textvariable = num_neurons_var, width = 18)
        
        num_neurons_label.grid(row = 0, column = 1)
        add_neuron.grid(column = 0, row = 1, padx = 7)
        subtract_neuron.grid(column = 0, row = 2, padx = 7)
        num_neurons_entry.grid(column = 1, row = 1, rowspan = 2)

        buttons = Frame(settings, width=200, height=500)
        buttons.pack(side=BOTTOM, fill = BOTH)
        buttons.grid_columnconfigure(0, weight = 1)

        settings_apply = Button(buttons, text = 'Apply', command = lambda: self.apply_layer(settings, num_neurons_var))
        settings_close = Button(buttons, text='Close', command = lambda: self.close_layer(settings, num_neurons_var))
        
        settings_apply.grid(column = 1, row = 0, padx = 10, pady = 5)
        settings_close.grid(column = 2, row = 0, padx = 10, pady = 5)

        settings.mainloop()


# The entire network itself, a collection of layers
class NeuralNetwork:
    # Default constructor for the network
    # Starts with an input layer with 2 nodes, a single hidden layer with 3 nodes, and an output  layer with two nodes
    def __init__(self):
        self.input = Layer(canvas.winfo_width() / 4, canvas.winfo_height() / 2, 'red')
        self.hidden = [Layer(canvas.winfo_width() / 2, canvas.winfo_height() / 2)]
        self.output = Layer(canvas.winfo_width() / 4 * 3, canvas.winfo_height() / 2, 'blue')
        self.last_x = WIN_WIDTH / 4 * 3
        self.num_hidden = 1
        self.hidden_desired = 1
        self.input.add_neuron()
        self.hidden[0].add_neuron()
        self.hidden[0].add_neuron()

        self.network = [self.input, self.output, self.hidden[0]]

    # Increase the number of hidden layers by one
    def add_hidden(self):
        self.last_x += WIN_WIDTH
        self.num_hidden += 1
        self.hidden_desired += 1
        self.network.append(Layer(self.last_x, CAN_HEIGHT / 2))
        self.hidden.append(Layer(self.last_x, CAN_HEIGHT / 2))
        canvas.update()

    # Decrease the number of hidden layers by one
    def subtract_hidden(self, layer):
        self.num_hidden -= 1
        self.hidden_desired -= 1
        self.network.remove(layer)

    # Return the network itself as an array of its Layers
    def get_network(self):
        return self.network


app = NeuralNetwork()
menu.add_command(label="Settings")
Button(canvas, text = 'add', command = app.add_hidden).pack(side = BOTTOM)
Button(canvas, text = 'add input neuron', command = app.network[0].add_neuron).pack(side = BOTTOM)
Button(canvas, text = 'add out neuron', command = app.network[1].add_neuron).pack(side = BOTTOM)
Button(canvas, text = 'add hidden neuron', command = app.network[2].add_neuron).pack(side = BOTTOM)
root.mainloop()
