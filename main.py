from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.colorchooser import *

global root 
global menu 


root = Tk()
menu= Menu(root)

def new_file():
    print("New File!")


def open_file():
    name = askopenfilename()
    print(name)


def about():
    print("This is a simple example of a menu")

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

canvas = Canvas(root, height=500, width=500)
canvas.pack()


mainloop()