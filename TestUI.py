from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.colorchooser import *


def new_file():
    print("New File!")


def open_file():
    name = askopenfilename()
    print(name)


def about():
    print("This is a simple example of a menu")

def chooseColor():
    color = askcolor()

def circleClick(event):
    chooseColor()

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

Button(text = 'hi ben', command = chooseColor).pack()

canvas = Canvas(root, height=500, width=500)
canvas.pack()
coords = 50, 50, 100, 100
circleOne = canvas.create_oval(coords, fill='red')
canvas.tag_bind(circleOne, '<Double-1>', circleClick)

mainloop()