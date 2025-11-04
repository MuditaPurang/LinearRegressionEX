#!/usr/bin/env python
# coding: utf-8
 
 


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')


 
#   housing price prediction.  
 


# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

 


# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")
 


# m is the number of training examples
m = len(x_train)
print(f"Number of training examples is: {m}")


# ### Training example `x_i, y_i`
# 
#will use (x$^{(i)}$, y$^{(i)}$) to denote the $i^{th}$ training example. Since Python is zero indexed, (x$^{(0)}$, y$^{(0)}$) is (1.0, 300.0) and (x$^{(1)}$, y$^{(1)}$) is (2.0, 500.0). 
# 
# To access a value in a Numpy array, one indexes the array with the desired offset. For example the syntax to access location zero of `x_train` is `x_train[0]`.
 


i = 0 # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")


# ### Plotting the data

# You can plot these two points using the `scatter()` function in the `matplotlib` library, as shown in the cell below. 
# - The function arguments `marker` and `c` show the points as red crosses (the default is blue dots).
 


# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()


# ## Model function
 
# $$ f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{1}$$
 
# **Note: You can come back to this cell to adjust the model's w and b parameters**
 

w = 100
b = 100
print(f"w: {w}")
print(f"b: {b}")


# Now, let's compute the value of $f_{w,b}(x^{(i)})$ for your two data points. You can explicitly write this out for each data point as - 
# 
# for $x^{(0)}$, `f_wb = w * x[0] + b`
# 
# for $x^{(1)}$, `f_wb = w * x[1] + b`
 


def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb


# Now let's call the `compute_model_output` function and plot the output..

# In[ ]:


tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()


# As you can see, setting $w = 100$ and $b = 100$ does *not* result in a line that fits our data. 
# 
 


w = 200                         
b = 100    
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")


# # Congratulations!
#  you have learned:
#  - Linear regression builds a model which establishes a relationship between features and targets
#      - In the example above, the feature was house size and the target was house price
#      - for simple linear regression, the model has two parameters $w$ and $b$ whose values are 'fit' using *training data*.
#      - once a model's parameters have been determined, the model can be used to make predictions on novel data.
 