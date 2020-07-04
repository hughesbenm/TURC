from tkinter import *
import tkinter.ttk as ttk
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

# Constants for the keras side of things
FUNCTIONS = ('linear', 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential')
LAYERS = ('Dense', 'Dropout')
INITIALIZERS = ('random_normal', 'random_uniform', 'truncated_normal', 'zeros', 'ones', 'glorot_normal',
                'glorot_uniform', 'identity', 'orthogonal', 'constant', 'variance_Scaling')


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
        self.layer_type = LAYERS[0]     # Dense
        self.function = FUNCTIONS[0]    # Linear
        self.bias = INITIALIZERS[0]     # Zeros
        self.desired_function = self.function
        canvas.tag_bind(self.layer[0].get_tag(), '<Button-3>', self.open_settings)
        self.settings = Toplevel()
        self.settings.protocol('WM_DELETE_WINDOW', self.close_layer)
        self.settings.resizable(False, False)
        self.settings.minsize(width = 300, height = 350)
        self.settings.maxsize(width = 300, height = 350)
        self.settings.withdraw()
        self.sett_frame = Frame(self.settings)
        self.sett_frame.pack(expand = True, fill = BOTH)
        self.layer_type_label = Label(self.sett_frame, text = 'Layer Type')
        # Buttons for settings
        self.layer_type_label = Label(self.sett_frame, text = 'Layer Type')
        self.layer_type_var = StringVar(self.sett_frame)
        self.layer_type_var.set(self.layer_type)
        self.layer_dropdown = ttk.Combobox(self.sett_frame, textvariable = self.layer_type_var, width = 10,
                                           values = LAYERS, state = 'readonly')
        self.layer_dropdown.bind('<<ComboboxSelected>>', self.arrange_settings)
        self.num_neurons_label = Label(self.sett_frame, text = 'Number of Neurons')
        self.num_neurons_var = IntVar(self.settings, self.desired_neurons)
        self.add_neuron_arrow = Button(self.sett_frame, image = up_arrow, height = 10, command = self.add_desired)
        self.subtract_neuron = Button(self.sett_frame, image = down_arrow, height = 10, command = self.subtract_desired)
        self.num_neurons_entry = Entry(self.sett_frame, textvariable = self.num_neurons_var, width = 9)
        self.color_label = Label(self.sett_frame, text = 'Layer Color')
        self.color_button = Button(self.sett_frame, bg = self.color, width = 2, command = self.set_desired_color)
        self.function_var = StringVar(self.sett_frame)
        self.function_var.set(self.function)
        self.function_label = Label(self.sett_frame, text = 'Activation Function')
        self.function_dropdown = ttk.Combobox(self.sett_frame, textvariable = self.function_var, width = 10,
                                              values = FUNCTIONS, state = 'readonly')
        self.bias_var = StringVar(self.sett_frame)
        self.bias_var.set(self.bias)
        self.use_bias_label = Label(self.sett_frame, text = 'Use Bias')
        self.bias_check_var = BooleanVar(self.sett_frame)
        self.bias_check_var.set(True)
        self.bias_check = Checkbutton(self.sett_frame, variable = self.bias_check_var,
                                      command = self.switch_bias_dropdown)
        self.bias_label = Label(self.sett_frame, text = 'Bias Initializer')
        self.bias_dropdown = ttk.Combobox(self.sett_frame, textvariable = self.bias_var, width = 18,
                                          values = INITIALIZERS, state = 'readonly')
        self.bias_dropdown_flag = True

        self.apply_close_frame = Frame(self.sett_frame)
        self.settings_apply = Button(self.apply_close_frame, text = 'Apply', command = self.apply_layer)
        self.settings_close = Button(self.apply_close_frame, text = 'Close', command = self.close_layer)
        self.settings_apply.grid(column = 1, row = 0, padx = 7, pady = 7, sticky = W)
        self.settings_close.grid(column = 2, row = 0, padx = 7, pady = 7, sticky = W)
        self.apply_close_frame.columnconfigure(0, weight = 1)

    # Runs when "Apply" is clicked in a layer's settings
    # Changes the layer based on changes made in the settings menu
    def apply_layer(self):
        self.layer_type = self.layer_type_var.get()
        self.function = self.function_var.get()
        self.bias = self.bias_var.get()
        self.desired_neurons = self.num_neurons_var.get()
        self.color = self.desired_color
        if self.desired_neurons > self.num_neurons:
            for i in range(self.desired_neurons - self.num_neurons):
                self.layer.append(Neuron(self.color, self.CONST_X, self.y_interval, 25))
                canvas.tag_bind(self.layer[-1].get_tag(), '<Button-3>', self.open_settings)
        if self.desired_neurons < self.num_neurons:
            for i in range(self.num_neurons - self.desired_neurons):
                canvas.delete(self.layer.pop().get_tag())
        self.num_neurons = self.desired_neurons
        self.orient_neurons()
        self.set_color()
        self.settings.withdraw()

    # Runs when "Close" is clicked in a layer's settings
    # Disregards any and all changes made in the settings menu
    def close_layer(self):
        self.settings.withdraw()
        self.num_neurons_var.set(self.num_neurons)
        self.desired_neurons = self.num_neurons
        self.desired_color = self.color

    # Set a layer's desired color in settings
    def set_desired_color(self, event = None):
        self.desired_color = askcolor(parent = self.sett_frame)[1]
        self.color_button.config(bg = self.desired_color)

    # Set a layer's color
    def set_color(self):
        for i in self.layer:
            i.set_background(self.color)

    def switch_bias_dropdown(self):
        if self.bias_dropdown_flag:
            self.bias_dropdown.config(state = 'disable')
        else:
            self.bias_dropdown.config(state = 'enable')
        self.bias_dropdown_flag = not self.bias_dropdown_flag

    # Increase the number of neurons in the layer by one, bind it to settings on right click, reorient the layer
    def add_neuron(self):
        self.layer.append(Neuron(self.color, self.CONST_X, self.y_interval, 25))
        self.num_neurons += 1
        self.desired_neurons += 1
        canvas.tag_bind(self.layer[self.num_neurons - 1].get_tag(), '<Button-3>', self.open_settings)
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
    def add_desired(self):
        self.desired_neurons += 1
        self.num_neurons_var.set(self.desired_neurons)

    # Decrease the number of desired neurons by one, only called by settings menu, bounce back up to 1 if below 0
    def subtract_desired(self):
        self.desired_neurons -= 1
        if self.desired_neurons <= 0:
            self.desired_neurons = 1
        self.num_neurons_var.set(self.desired_neurons)

    def arrange_settings(self, event = None):
        self.layer_type = self.layer_type_var.get()
        for widget in self.sett_frame.winfo_children():
            widget.grid_forget()

        self.sett_frame.update()

        self.sett_frame.columnconfigure(0, weight = 1)
        self.sett_frame.columnconfigure(1, weight = 1)
        self.layer_type_label.grid(row = 0, column = 0, sticky = W)
        self.layer_dropdown.grid(row = 1, column = 0, padx = 7, sticky = W)
        self.sett_frame.rowconfigure(3, minsize = 20)
        ttk.Separator(self.sett_frame, orient = HORIZONTAL).grid(row = 3, column = 0, padx = 7, columnspan = 2,
                                                                 sticky = EW)
        self.sett_frame.rowconfigure(100, weight = 1)
        ttk.Separator(self.sett_frame, orient = HORIZONTAL).grid(row = 101, column = 0, padx = 7, columnspan = 2,
                                                                 sticky = EW)
        self.apply_close_frame.grid(row = 102, column = 1, padx = 7, sticky = E)

        if self.layer_type == 'Dense':
            self.num_neurons_label.grid(row = 4, column = 0, sticky = W)
            self.num_neurons_entry.grid(row = 5, column = 0, rowspan = 2, padx = 32, sticky = W)
            self.add_neuron_arrow.grid(row = 5, column = 0, padx = 7, sticky = W)
            self.subtract_neuron.grid(row = 6, column = 0, padx = 7, sticky = W)

            self.function_label.grid(row = 4, column = 1, sticky = W)
            self.function_dropdown.grid(row = 5, column = 1, padx = 7, sticky = W)

            self.sett_frame.grid_rowconfigure(7, minsize = 20)

            self.use_bias_label.grid(row = 8, column = 0, sticky = W)
            self.bias_check.grid(row = 9, column = 0, padx = 7, sticky = W)

            self.bias_label.grid(row = 8, column = 1, sticky = W)
            self.bias_dropdown.grid(row = 9, column = 1, padx = 7, sticky = W)

            self.color_label.grid(row = 10, column = 0, sticky = W)
            self.color_button.grid(row = 11, column = 0, padx = 7, sticky = W)

        elif self.layer_type == 'Dropout':
            pass

    # Code for the layer settings menu that appears on right clicking a neuron
    def open_settings(self, event = None):
        self.arrange_settings()
        self.settings.deiconify()
        self.settings.focus_force()
        self.num_neurons_var.set(self.desired_neurons)
        self.settings.mainloop()


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

Button(canvas, text = 'add', command = app.add_hidden).pack(side = BOTTOM)
Button(canvas, text = 'add input neuron', command = app.network[0].add_neuron).pack(side = BOTTOM)
Button(canvas, text = 'add out neuron', command = app.network[1].add_neuron).pack(side = BOTTOM)
Button(canvas, text = 'add hidden neuron', command = app.network[2].add_neuron).pack(side = BOTTOM)
root.mainloop()
