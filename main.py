from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.colorchooser import *

global root 
global menu 
global canvas

root = Tk()
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

canvas = Canvas(root, height=500, width=500)
canvas.pack()
coords = 50, 50, 100, 100

mainloop()