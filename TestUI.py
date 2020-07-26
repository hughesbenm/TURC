from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.colorchooser import *


class Test:
    def __init__(self):
        self.data1 = 10

    def test(self, data2):
        if self.data1 is data2:
            print(10)


test = Test()
test.test(test.data1)
