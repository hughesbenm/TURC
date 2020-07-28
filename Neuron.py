from tkinter import *
from tkinter import filedialog
from tensorflow import keras
import tkinter.ttk as ttk
from tkinter.colorchooser import *
import os.path
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd

# Main window's dimensions
WIN_WIDTH = 1200
WIN_HEIGHT = 600
CAN_WIDTH = WIN_WIDTH * 0.95
CAN_HEIGHT = WIN_HEIGHT * 0.95
DEFAULT_Y = WIN_HEIGHT / 2
MAROON = '#700000'
FOREST = '#007016'
ACTIVATION_COLOR = '#0F891C'
CONVOLUTIONAL_COLOR = '#B40E0E'
DENSE_COLOR = '#741062'
DROPOUT_COLOR = 'black'
FLATTEN_COLOR = '#1DB1CE'
NORMALIZATION_COLOR = '#1B2678'
POOLING_COLOR = '#5B5654'
INPUT_COLOR = '#EACC23'
OUTPUT_COLOR = '#CF5416'

# Create the standard window
root = Tk()
root.minsize(WIN_WIDTH, WIN_HEIGHT)

# Create and add the menu bar at the top of the window
menu = Menu(root)
canvas = Canvas(root, height = WIN_HEIGHT, width = WIN_WIDTH)
root.config(menu = menu)
root.resizable(False, False)
root.focus_force()

# Create and add the canvas that takes up the entire main window
canvas.place(anchor = CENTER, relheight = .95, relwidth = 0.95, relx = 0.5, rely = 0.5)

# Necessary for canvas.winfo_ATTRIBUTE to be updated, see NeuralNetwork.__init__() and Layer.orient_neurons()
canvas.update()

# Importing images for various buttons and things
up_arrow = PhotoImage(master = root, file = os.path.join(os.path.dirname(__file__), "Images/Up.png"))
down_arrow = PhotoImage(master = root, file = os.path.join(os.path.dirname(__file__), "Images/Down.png"))

# Constants for the keras side of things
FUNCTIONS = [None, 'elu', 'exponential', 'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh']

LAYERS = ['Activation', 'Convolutional', 'Dense', 'Dropout', 'Flatten', 'Normalization', 'Pooling']

LAYER_COLORS = [ACTIVATION_COLOR, CONVOLUTIONAL_COLOR, DENSE_COLOR, DROPOUT_COLOR, FLATTEN_COLOR, NORMALIZATION_COLOR,
                POOLING_COLOR]

INITIALIZERS = ['zeros', 'constant', 'identity', 'glorot_normal', 'glorot_uniform', 'ones', 'orthogonal',
                'random_normal', 'random_uniform', 'truncated_normal', 'variance_scaling']


# Simple function to turn (center_x, center_y, radius) into (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
def coords(x, y, radius):
    return x - radius, y - radius, x + radius, y + radius


# Most basic unit of the net, used mostly for displaying the logical neurons than any actual functionality
class Neuron:
    # default constructor, creates a separate oval for each node
    def __init__(self, color, x, y, radius):
        self.node = canvas.create_oval(coords(x, y, radius), fill = color)

    # Set the color of this node, should always be the same for an entire layer
    def set_background(self, color):
        canvas.itemconfig(self.node, fill = color)

    # Returns the tag of the created oval for the individual node, allows for Layer to change things
    def get_tag(self):
        return self.node

    def move_x(self, x):
        canvas.move(self.node, x, 0)


# Collection of Neurons
class Layer:
    # default constructor, each Layer starts with a single neuron
    def __init__(self, color = LAYER_COLORS[0]):
        self.prev_layer = None
        self.next_layer = None
        self.x = 0
        self.y_interval = 0
        self.color = color
        self.layer = [Neuron(self.color, self.x, self.y_interval, 25)]
        self.num_neurons = 1
        self.desired_neurons = 1
        self.layer_type = LAYERS[0]     # Activation
        self.desired_type = self.layer_type
        self.function = FUNCTIONS[0]    # Linear
        self.bias = INITIALIZERS[0]     # Zeros
        self.dropout_rate = 0.25
        self.desired_rate = self.dropout_rate
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

        # Layer type
        self.layer_type_frame = Frame(self.sett_frame)
        self.layer_type_label = Label(self.layer_type_frame, text = 'Layer Type')
        self.layer_type_var = StringVar(self.sett_frame)
        self.layer_type_var.set(self.layer_type)
        self.layer_dropdown = ttk.Combobox(self.layer_type_frame, textvariable = self.layer_type_var, width = 13,
                                           values = LAYERS, state = 'readonly')
        self.layer_dropdown.bind('<<ComboboxSelected>>', self.arrange_settings)
        self.layer_type_label.grid(row = 0, column = 0, sticky = W)
        self.layer_dropdown.grid(row = 1, column = 0, padx = 7, sticky = W)

        # Number of Neurons: Dense
        def check_num_neuron_entry(inp):
            if (inp.isdigit() and inp != '0') or inp == '':
                return True
            else:
                return False
        num_reg = self.settings.register(check_num_neuron_entry)
        self.num_neurons_frame = Frame(self.sett_frame)
        self.num_neurons_label = Label(self.num_neurons_frame, text = 'Number of Neurons')
        self.num_neurons_var = IntVar(self.settings, self.desired_neurons)
        self.add_neuron_arrow = Button(self.num_neurons_frame, image = up_arrow, height = 10,
                                       command = self.add_desired)
        self.subtract_neuron_arrow = Button(self.num_neurons_frame, image = down_arrow, height = 10,
                                            command = self.subtract_desired)
        self.num_neurons_entry = Entry(self.num_neurons_frame, textvariable = self.num_neurons_var, width = 9,
                                       validate = 'key', validatecommand = (num_reg, '%P'))
        self.num_neurons_label.grid(row = 0, column = 0, columnspan = 3, sticky = W)
        self.num_neurons_entry.grid(row = 1, column = 1, rowspan = 2)
        self.add_neuron_arrow.grid(row = 1, column = 0, padx = 7, sticky = W)
        self.subtract_neuron_arrow.grid(row = 2, column = 0, padx = 7, sticky = W)
        self.num_neurons_frame.columnconfigure(2, weight = 1)

        # Activation Function: Activation/Dense
        self.function_frame = Frame(self.sett_frame)
        self.function_var = StringVar(self.sett_frame)
        self.function_var.set(self.function)
        self.function_label = Label(self.function_frame, text = 'Activation Function')
        self.function_dropdown = ttk.Combobox(self.function_frame, textvariable = self.function_var, width = 11,
                                              values = FUNCTIONS, state = 'readonly')
        self.function_label.grid(row = 0, column = 0, sticky = W)
        self.function_dropdown.grid(row = 1, column = 0, padx = 7, sticky = W)

        # Bias Check Mark: Dense
        self.bias_check_frame = Frame(self.sett_frame)
        self.use_bias_label = Label(self.bias_check_frame, text = 'Use Bias')
        self.bias_check_var = BooleanVar(self.bias_check_frame)
        self.bias_check_var.set(True)
        self.bias_check = Checkbutton(self.bias_check_frame, variable = self.bias_check_var,
                                      command = self.switch_bias_dropdown)
        self.use_bias_label.grid(row = 0, column = 0, sticky = W)
        self.bias_check.grid(row = 1, column = 0, padx = 7, sticky = W)

        # Bias Initializer: Dense
        self.bias_initializer_frame = Frame(self.sett_frame)
        self.bias_var = StringVar(self.sett_frame)
        self.bias_var.set(self.bias)
        self.bias_label = Label(self.bias_initializer_frame, text = 'Bias Initializer')
        self.bias_dropdown = ttk.Combobox(self.bias_initializer_frame, textvariable = self.bias_var, width = 18,
                                          values = INITIALIZERS, state = 'readonly')
        self.bias_dropdown_flag = True
        self.bias_label.grid(row = 0, column = 0, sticky = W)
        self.bias_dropdown.grid(row = 1, column = 0, padx = 7, sticky = W)

        # Pooling Type: Pooling
        self.pooling_type_frame = Frame(self.sett_frame)

        # Rate: Dropout
        def check_dropout_entry(inp):

            if inp == '':
                return True
            elif inp.replace('.', '1', 1).isdigit() and 0 <= float(inp) <= 1:
                return True
            else:
                return False
        drop_reg = self.settings.register(check_dropout_entry)
        self.dropout_rate_frame = Frame(self.sett_frame)
        self.dropout_rate_label = Label(self.dropout_rate_frame, text = 'Rate')
        self.dropout_rate_var = DoubleVar(self.settings, self.dropout_rate)
        self.increase_rate_arrow = Button(self.dropout_rate_frame, image = up_arrow, height = 10,
                                          command = self.increase_dropout_rate)
        self.decrease_rate_arrow = Button(self.dropout_rate_frame, image = down_arrow, height = 10,
                                          command = self.decrease_dropout_rate)
        self.dropout_rate_entry = Entry(self.dropout_rate_frame, textvariable = self.dropout_rate_var, width = 9,
                                        validate = 'key', validatecommand = (drop_reg, '%P'))
        self.dropout_rate_label.grid(row = 0, column = 0, columnspan = 3, sticky = W)
        self.dropout_rate_entry.grid(row = 1, column = 1, rowspan = 2)
        self.increase_rate_arrow.grid(row = 1, column = 0, padx = 7, sticky = W)
        self.decrease_rate_arrow.grid(row = 2, column = 0, padx = 7, sticky = W)
        self.dropout_rate_frame.columnconfigure(2, weight = 1)

        # Layer Apply Close
        self.apply_close_frame = Frame(self.sett_frame)
        self.settings_apply = Button(self.apply_close_frame, text = 'Apply', command = self.apply_layer)
        self.settings_close = Button(self.apply_close_frame, text = 'Close', command = self.close_layer)
        self.settings_apply.grid(column = 1, row = 0, padx = 7, pady = 7, sticky = W)
        self.settings_close.grid(column = 2, row = 0, padx = 7, pady = 7, sticky = W)
        self.apply_close_frame.columnconfigure(0, weight = 1)

    def set_neurons(self, num = None):
        if num is not None:
            self.desired_neurons = num
        if self.desired_neurons > self.num_neurons:
            for i in range(self.desired_neurons - self.num_neurons):
                self.layer.append(Neuron(self.color, self.x, self.y_interval, 25))
                canvas.tag_bind(self.layer[-1].get_tag(), '<Button-3>', self.open_settings)
        if self.desired_neurons < self.num_neurons:
            for i in range(self.num_neurons - self.desired_neurons):
                canvas.delete(self.layer.pop().get_tag())
        self.num_neurons = self.desired_neurons
        self.orient_neurons()

    # Runs when "Apply" is clicked in a layer's settings
    # Changes the layer based on changes made in the settings menu
    def apply_layer(self):
        if self.dropout_rate_entry.get() == '':
            self.dropout_rate_var.set(self.desired_rate)
        if self.num_neurons_entry.get() == '':
            self.num_neurons_var.set(self.desired_neurons)
        self.layer_type = self.desired_type
        self.function = self.function_var.get()
        self.bias = self.bias_var.get()
        self.desired_neurons = self.num_neurons_var.get()
        self.desired_rate = self.dropout_rate_var.get()
        self.dropout_rate = self.desired_rate
        self.set_neurons()
        if self.layer_type != 'Input' and self.layer_type != 'Output':
            self.color = LAYER_COLORS[LAYERS.index(self.layer_type)]
            self.set_color()
        self.settings.withdraw()

    # Runs when "Close" is clicked in a layer's settings
    # Disregards any and all changes made in the settings menu
    def close_layer(self):
        self.settings.withdraw()
        self.num_neurons_var.set(self.num_neurons)
        self.desired_neurons = self.num_neurons
        self.desired_type = self.layer_type
        self.layer_type_var.set(self.layer_type)

    def set_color(self):
        for neuron in self.layer:
            neuron.set_background(self.color)

    def switch_bias_dropdown(self):
        if self.bias_dropdown_flag:
            self.bias_dropdown.config(state = 'disable')
        else:
            self.bias_dropdown.config(state = 'enable')
        self.bias_dropdown_flag = not self.bias_dropdown_flag

    # Increase the number of neurons in the layer by one, bind it to settings on right click, reorient the layer
    def add_neuron(self):
        self.layer.append(Neuron(self.color, self.x, self.y_interval, 25))
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
        self.y_interval = CAN_HEIGHT / (self.num_neurons + 1)
        for i in range(self.num_neurons):
            canvas.coords(self.layer[i].get_tag(), coords(self.x, self.y_interval * (i + 1), 25))

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

    def increase_dropout_rate(self, rate = 0.05):
        self.desired_rate += rate
        if self.desired_rate >= 1.0:
            self.desired_rate = 1.0
        self.dropout_rate_var.set(round(self.desired_rate, 2))

    def decrease_dropout_rate(self, rate = 0.05):
        self.desired_rate -= rate
        if self.desired_rate <= 0.0:
            self.desired_rate = 0.0
        self.dropout_rate_var.set(round(self.desired_rate, 2))

    def arrange_settings(self, event = None):
        self.desired_type = self.layer_type_var.get()
        for widget in self.sett_frame.winfo_children():
            widget.grid_forget()

        self.sett_frame.columnconfigure(0, minsize = 150)
        self.sett_frame.columnconfigure(1, minsize = 150)

        self.layer_type_frame.grid(row = 0, column = 0, sticky = W)

        self.sett_frame.rowconfigure(1, minsize = 20)
        ttk.Separator(self.sett_frame, orient = HORIZONTAL).grid(row = 1, column = 0, padx = 7, columnspan = 2,
                                                                 sticky = EW)

        if self.desired_type == 'Activation':
            self.function_frame.grid(row = 2, column = 0, sticky = W)

        elif self.desired_type == 'Convolutional':
            pass
        elif self.desired_type == 'Dense':
            self.num_neurons_frame.grid(row = 2, column = 0, sticky = W)

            self.function_frame.grid(row = 2, column = 1, sticky = W)

            self.sett_frame.grid_rowconfigure(3, minsize = 20)

            self.bias_check_frame.grid(row = 4, column = 0, sticky = W)

            self.bias_initializer_frame.grid(row = 4, column = 1, sticky = W)

        elif self.desired_type == 'Dropout':
            self.dropout_rate_frame.grid(row = 2, column = 0, sticky = W)

        elif self.desired_type == 'Flatten':
            pass
        elif self.desired_type == 'Normalization':
            pass
        elif self.desired_type == 'Pooling':
            pass
        self.sett_frame.rowconfigure(100, weight = 1)
        ttk.Separator(self.sett_frame, orient = HORIZONTAL).grid(row = 101, column = 0, padx = 7, columnspan = 2,
                                                                 sticky = EW)
        self.apply_close_frame.grid(row = 102, column = 1, padx = 7, sticky = E)

    def get_x(self):
        return self.x

    def set_x(self, x):
        self.x = x

    def move_forward_x(self):
        for i in self.layer:
            i.move_x(50)
        self.x += 50

    def move_backward_x(self):
        for i in self.layer:
            i.move_x(-50)
        self.x -= 50

    def move_x_num(self, x):
        for i in self.layer:
            i.move_x(x)
        self.x += x

    # Code for the layer settings menu that appears on right clicking a neuron
    def open_settings(self, event = None):
        self.arrange_settings()
        self.settings.deiconify()
        self.settings.focus_force()
        self.num_neurons_var.set(self.desired_neurons)
        self.settings.mainloop()

    def erase_layer(self):
        for neuron in self.layer:
            canvas.delete(neuron.get_tag())


# The entire network itself, a collection of layers
class NeuralNetwork:
    # Default constructor for the network
    # Starts with an input layer with 2 nodes, a single hidden layer with 3 nodes, and an output  layer with two nodes
    def __init__(self):
        self.last_x = CAN_WIDTH / 4 * 3
        self.input = Layer(INPUT_COLOR)
        self.input.layer_type = "Input"
        self.input.layer_type_var.set(self.input.layer_type)
        self.input.layer_dropdown.config(state = DISABLED)
        self.input.set_neurons(2)
        self.output = Layer(OUTPUT_COLOR)
        self.input.next_layer = self.output
        self.output.layer_type = "Output"
        self.output.layer_type_var.set(self.output.layer_type)
        self.output.layer_dropdown.config(state = DISABLED)
        self.output_index = 1
        self.num_layers = 2
        self.hidden_desired = 1

        self.net_model = keras.Sequential()
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.x_train_button = Button(canvas, text = 'X_Train', command = lambda: self.prompt_data(self.x_train))
        self.x_train_button.pack(side = BOTTOM)
        self.y_train_button = Button(canvas, text = 'Y_Train', command = lambda: self.prompt_data(self.y_train))
        self.y_train_button.pack(side = BOTTOM)
        self.x_test_button = Button(canvas, text = 'X_Test', command = lambda: self.prompt_data(self.x_test))
        self.x_test_button.pack(side = BOTTOM)
        self.y_test_button = Button(canvas, text = 'Y_Test', command = lambda: self.prompt_data(self.y_test))
        self.y_test_button.pack(side = BOTTOM)
        self.run_button = Button(canvas, text = 'Run Network', bg = MAROON, command = self.run, state = DISABLED)
        self.run_button.pack(side = BOTTOM)
        Button(canvas, text = 'load', command = self.load_net).pack(side = TOP)
        Button(canvas, text = 'save', command = self.save_net).pack(side = TOP)

        self.orient_network()

    def orient_network(self):
        x_interval = CAN_WIDTH / (self.num_layers + 1)
        last_x = x_interval
        current = self.input
        while current is not None:
            current.x = last_x
            last_x += x_interval
            current.orient_neurons()
            current = current.next_layer

    def save_net(self):
        net_details = [self.x_train, self.y_train, self.x_test, self.y_test]
        layer_details = []
        current = self.input.next_layer
        while current != self.output:
            layer_details.append([current.layer_type])
            current = current.next_layer
        np.savez("saves.npz", net_details = net_details, layer_details = layer_details)

    def load_net(self):
        self.erase_hidden()
        self.input.next_layer = self.output
        self.output.prev_layer = self.output
        self.num_layers = 2
        try:
            saves = np.load("saves.npz", allow_pickle = True)
            net_details = saves['net_details']
            layer_details = saves['layer_details']
            current = self.input
            for i in range(len(layer_details)):
                new = Layer()
                # Somehow set new with all the necessary details
                store_next = current.next_layer
                current.next_layer = new
                new.prev_layer = current
                new.next_layer = store_next
                store_next.prev_layer = new
                self.num_layers += 1
                self.orient_network()
        except FileNotFoundError:
            pass

    # Increase the number of hidden layers by one
    def add_layer(self, index, new = None):
        new = Layer(ACTIVATION_COLOR)
        current = self.input
        print(current.layer_type)
        for pos in range(index - 1):
            current = current.next_layer
            print(current.layer_type)
        store_next = current.next_layer
        current.next_layer = new
        new.prev_layer = current
        new.next_layer = store_next
        store_next.prev_layer = new
        self.num_layers += 1
        self.orient_network()
        canvas.update()

    def prompt_data(self, data):
        root.filename = filedialog.askopenfilename(initialdir = os.path.dirname(__file__), title = "Select File",
                                                   filetypes = [('All valid files',
                                                                 '*.xls;*.xlsx;*.xlsm;*.xlsb;*.odf;*.csv'),
                                                                ('Excel files', '*.xls;*.xlsx;*.xlsm;*.xlsb;*.odf'),
                                                                ('CSV files', '*.csv')])
        try:
            try:
                temp_data = pd.read_csv(root.filename, sep = ',', header = None)
            except:
                temp_data = pd.read_excel(root.filename, usecols = [0], nrows = 10)

            if data is self.x_train:
                self.x_train = temp_data
            elif data is self.y_train:
                self.y_train = temp_data
            elif data is self.x_test:
                self.x_test = temp_data
            elif data is self.y_test:
                self.y_test = temp_data

            if self.x_train is not None and self.y_train is not None:
                if self.x_test is not None and self.y_test is not None:
                    self.run_button.config(state = NORMAL, bg = FOREST)
        except:
            print("Nope")

    def compile_network(self):
        current = self.input.next_layer
        while current != self.output:
            # Check for different layer types
            if current.layer_type == 'Activation':
                self.net_model.add(keras.layers.Activation(input_shape = (2,), activation = 'sigmoid'))

            elif current.layer_type == 'Dense':
                self.net_model.add(keras.layers.Dense(1, input_shape = (2,), activation = 'sigmoid'))

            elif current.layer_type == 'Dropout':
                self.net_model.add(keras.layers.Dropout(current.dropout_rate, input_shape = (2,)))

            elif current.layer_type == 'Flatten':
                self.net_model.add(keras.layers.Flatten())

            elif current.layer_type == 'Pooling':
                pass
            current = current.next_layer

        self.net_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        X, Y = make_classification(n_samples = 1000, n_features = 2, n_redundant = 0, n_informative = 2,
                                   random_state = 7, n_clusters_per_class = 1)
        self.net_model.fit(x = X, y = Y, verbose = 0, epochs = 50)
        print(self.net_model.summary())

    def run(self, event = None):
        self.compile_network()

        # Train the completed model on the predetermined training data
        pass

        # Predict the results for the predetermined inputs
        pass

    def erase_hidden(self):
        current = self.input.next_layer
        while current != self.output:
            current.erase_layer()
            current = current.next_layer


app = NeuralNetwork()
Button(canvas, text = 'add hidden layer', command = lambda: app.add_layer(app.num_layers - 1)).pack(side = BOTTOM)
root.mainloop()
