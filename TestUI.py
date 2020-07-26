from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.colorchooser import *
import numpy as np

ar = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
np.savez("Test", ar = ar, )
arr1 = np.load("Test.npz")
print(arr1['ar'])
