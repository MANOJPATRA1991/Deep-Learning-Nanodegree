import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.

# This implementation supports L of any dimensionality
def softmax(L): # Let L = [5, 6, 7]
    # View inputs as arrays with atleast two dimensions
    y = np.atleast_2d(L) # [[ 5, 6, 7 ]]
    
    # Find axis along which to do computation
    # axis = 0 means do computation row wise
    # axis = 1 means do computation column wise
    axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1) # axis = 1
    
    # Calculate maximum value in given list, i.e, max(L)
    y_max = np.max(y, axis = axis) # [7]
    
    # Step 1: Calculate exp(Ln - max(L)) which is equivalent to exp(Ln) / exp(max(L))
    # (i) Ln - max(L)
    # Expand dimensions of y_max along axis = 1 in our case and subtract from y
    y = y - np.expand_dims(y_max, axis) # [[5, 6, 7]] - [[7]] = [[-2,-1,0]]
    # (ii) exp(Ln - max(L))
    y = np.exp(y) # [[0.13533528 0.36787944 1.        ]]
    
    # Step 2: Calculate sum(exp(Ln - max(L))) which is equivalent to sum(exp(Ln)) / exp(max(L))
    y_sum = np.sum(y, axis = axis) # Add along axis = 1 in our case => [1.50321472]
    y_sum_expanded = np.expand_dims(y_sum, axis) # [[1.50321472]]
    
    # Step 3: Calculate exp(Ln) / sum(exp(Ln)) which can be obtained by dividing result of Step 1 by result of Step 2
    p = y / y_sum_expanded # [[0.09003057 0.24472847 0.66524096]]
    
    return p.flatten() # Return array of copy collapsed to one dimension
    
    pass
