from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.colorchooser import *

color = 'black', 'black'
COORDS = 50, 50, 100, 100

class Neuron:
    # default constructor
    def __init__(self):
        self.node = canvas.create_oval(COORDS)
        canvas.tag_bind(self.node, '<Double-1>', self.circle_click)


    def set_background(self, event = None):
        color = askcolor()
        canvas.itemconfig(self.node, fill = color)
    

    def circle_click(self, event):
        print('hi')
        set_background(self)


def new_file():
    print("New File!")


def open_file():
    name = askopenfilename()
    print(name)


def about():
    print("This is a simple example of a menu")

def chooseColor():
    global color
    color = askcolor()
    print(color[1])

def circleClick(event):
    canvas.create_oval(200, 200, 300, 300, fill=color[1])

    chooseColor()

global root
root = Tk()
global menu 
menu= Menu(root)

root.config(menu=menu)
fileMenu = Menu(menu, tearoff=False)
menu.add_cascade(label="File", menu=fileMenu)
fileMenu.add_command(label="New", command=new_file)
fileMenu.add_command(label="Open...", command=open_file)
fileMenu.add_separator()
fileMenu.add_command(label="Exit", command=root.quit)

helpMenu = Menu(menu)
menu.add_cascade(label="Help", menu=helpMenu)
helpMenu.add_command(label="About...", command=about)

root.minsize(1200, 600) 

Button(text = 'hi ben', command = chooseColor).pack()

global canvas 
canvas = Canvas(root, height=500, width=500)
canvas.pack()
coords = 50, 50, 100, 100

a = Neuron()

mainloop()