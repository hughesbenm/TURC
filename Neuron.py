from tkinter import *
import tkinter.ttk as ttk
# from keras import *
from tkinter.colorchooser import *
import os.path

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
root.resizable(False, False)

# Create and add the canvas that takes up the entire main window
canvas = Canvas(root, height=CAN_HEIGHT, width=CAN_WIDTH)
canvas.pack()
canvas.place(anchor = CENTER, relheight = .95, relwidth = 0.95, relx = 0.5, rely = 0.5)

# Necessary for canvas.winfo_ATTRIBUTE to be updated, see NeuralNetwork.__init__() and Layer.orient_neurons()
canvas.update()

# Importing images for various buttons and things

up_arrow = PhotoImage(master = root, file = os.path.join(os.path.dirname(__file__), "Images/Up.png"))
down_arrow = PhotoImage(master = root, file = os.path.join(os.path.dirname(__file__), "Images/Down.png"))


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
        self.node = canvas.create_oval(coords(x, y, radius), fill = color)

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
        self.desired_color = self.color
        canvas.tag_bind(self.layer[0].get_tag(), '<Button-3>', self.popup_layer_settings)

    # Runs when "Apply" is clicked in a layer's settings
    # Changes the layer based on changes made in the settings menu
    def apply_layer(self, settings, num_neurons_var):
        self.desired_neurons = num_neurons_var.get()
        self.color = self.desired_color
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
        self.set_color()

    # Runs when "Close" is clicked in a layer's settings
    # Disregards any and all changes made in the settings menu
    def close_layer(self, settings, num_neurons_var):
        settings.destroy()
        num_neurons_var.set(self.num_neurons)
        self.desired_neurons = self.num_neurons
        self.desired_color = self.color

    # Set a layer's desired color in settings
    def set_desired_color(self, color_button, sett_frame, event = None):
        self.desired_color = askcolor(parent = sett_frame)[1]
        color_button.config(bg = self.desired_color)

    # Set a layer's color
    def set_color(self):
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
        # Main window for settings
        settings = Toplevel()
        settings.resizable(False, False)
        settings.focus_force()
        settings.minsize(width = 250, height = 350)

        # Frame to include buttons for settings
        sett_frame = Frame(settings)
        # sett_frame.grid_propagate(False)
        sett_frame.pack(side = TOP, fill = X)

        # Buttons for settings
        num_neurons_label = Label(sett_frame, text = 'Number of Neurons')
        num_neurons_var = IntVar(settings, self.desired_neurons)
        add_neuron = Button(sett_frame, image = up_arrow, height = 10)
        add_neuron.config(command = lambda: self.add_desired(num_neurons_var))
        subtract_neuron = Button(sett_frame, image = down_arrow, height = 10)
        subtract_neuron.config(command = lambda: self.subtract_desired(num_neurons_var))
        num_neurons_entry = Entry(sett_frame, textvariable = num_neurons_var, width = 18)
        color_label = Label(sett_frame, text = 'Layer Color')
        color_button = Button(sett_frame, bg = self.color, width = 2)
        color_button.config(command = lambda: self.set_desired_color(color_button, sett_frame))
        var = StringVar(sett_frame)
        function_dropdown = ttk.Combobox(sett_frame, textvariable = var, width = 10)
        function_dropdown.bind("Hello")

        # Arranges all buttons
        num_neurons_label.grid(row = 0, column = 0, columnspan = 5, sticky = W)
        add_neuron.grid(row = 1, column = 0, padx = 7)
        subtract_neuron.grid(row = 2, column = 0, padx = 7)
        num_neurons_entry.grid(row = 1, column = 1, rowspan = 2, columnspan = 4)
        sett_frame.grid_rowconfigure(3, minsize = 20)
        color_label.grid(row = 4, column = 0, columnspan = 4, sticky = W)
        sett_frame.grid_rowconfigure(5)
        color_button.grid(row = 5, column = 0, padx = 5)
        sett_frame.grid_rowconfigure(6, minsize = 20)
        function_dropdown.grid(row = 7, column = 0)

        # Frame for Apply and Close
        buttons = Frame(settings, width=200, height=500)
        buttons.pack(side=BOTTOM, fill = BOTH)
        buttons.grid_columnconfigure(0, weight = 1)

        # Apply and Close Buttons
        settings_apply = Button(buttons, text = 'Apply', command = lambda: self.apply_layer(settings, num_neurons_var))
        settings_close = Button(buttons, text='Close', command = lambda: self.close_layer(settings, num_neurons_var))

        # Place Apply and Close
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
        self.predict_inputs = None
        self.training_data = None
        self.run_button = Button(canvas, text = 'Run Network', bg = 'green', command = self.run)
        self.run_button.pack(side = BOTTOM)

        self.network = [self.input, self.output, self.hidden[0]]
        self.net_model = None  # should be Sequential()

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

    # Bring up a window to ask the user to select a file for data to train on
    def prompt_training_data(self):
        pass

    # Bring up a window to ask the user to select a file for data to predict results for
    def prompt_predict_inputs(self):
        pass

    def compile_network(self):
        for hidden_layer in self.hidden:
            # Check for different layer types
            pass

            # Add the correct layer to the model
            pass

    def run(self, event = None):
        if self.training_data is not None:
            self.prompt_predict_inputs()
        if self.predict_inputs is not None:
            self.prompt_training_data()

        self.compile_network()

        # Train the completed model on the predetermined training data
        pass

        # Predict the results for the predetermined inputs
        pass


app = NeuralNetwork()

Button(canvas, text = 'add', command = app.add_hidden).pack(side = BOTTOM)
Button(canvas, text = 'add input neuron', command = app.network[0].add_neuron).pack(side = BOTTOM)
Button(canvas, text = 'add out neuron', command = app.network[1].add_neuron).pack(side = BOTTOM)
Button(canvas, text = 'add hidden neuron', command = app.network[2].add_neuron).pack(side = BOTTOM)
root.mainloop()
