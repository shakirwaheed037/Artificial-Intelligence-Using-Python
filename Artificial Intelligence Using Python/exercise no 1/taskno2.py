# 2. Write a Python program using NumPy to: 
# • Create a NumPy array containing the marks [78, 85, 92, 67, 88]. 
# • Display the array. 
# • Convert the array into float data type. 
# • Display the converted array


import numpy as np

print("Hi , To convert the integer array into float type using Numpy")

marks = np.array([78, 85, 92, 67, 88])

print("old array ", marks)

print("new converted array ", marks.astype(float))