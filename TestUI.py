from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.colorchooser import *

root = Tk()

B = Button(root, text='test', fg='black')
B.pack()
B.place(bordermode = OUTSIDE, height = 100, width = 100, anchor = SE, relx = 1, rely = 1)

root.mainloop()