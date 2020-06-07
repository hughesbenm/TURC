from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.colorchooser import *

color = 'black', 'black'

def new_file():
    print("New File!")


def open_file():
    name = askopenfilename()
    print(name)


def about():
    print("This is a simple example of a menu")

<<<<<<< HEAD

def choose_color():
    color = askcolor()


def circle_click(event = None, object = canvas.create_oval()):
    # color = askcolor()
    print(object)
    # change_color(self, color)


def change_color(self, color):
    canvas.itemconfig(self, fill = color[1])


=======
def chooseColor():
    global color
    color = askcolor()

def circleClick(event):
    canvas.create_oval(200, 200, 300, 300, fill=color[1])
    chooseColor()
>>>>>>> master

root = Tk()
menu = Menu(root)

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

Button(text = 'hi ben', command = choose_color).pack()

canvas = Canvas(root, height=500, width=500)
canvas.pack()
coords = 50, 50, 100, 100
<<<<<<< HEAD
circleOne = canvas.create_oval(coords, fill='red')
canvas.tag_bind(circleOne, '<Double-1>', circle_click(circleOne))
=======
circleOne = canvas.create_oval(coords, fill=color[1])
canvas.tag_bind(circleOne, '<Double-1>', circleClick)
>>>>>>> master

mainloop()