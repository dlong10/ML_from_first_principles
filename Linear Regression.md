# **Linear Regression**
## 
**Linear Regression** is a supervised ML technique used to model the relationship between one or more input features and a continuous output variable.
It does by fitting a linear function to the training data, allowing it to predict numeric values (e.g. house prices, test scores, measurements).

The function for our linear model is: **f<sub>w,b</sub>(x) = w · x + b**

where: 
- **x** is a matrix of shape (m,); m examples and n features
- **w** is the weights of the model with shape (n,); n weights
- **b** is the bias with shape (); 1 value
- **·** is the mathematical notation for the dot product.

1) Let's **compute** the linear regression model, given initial values of w, b and dataset X

```python
def compute_model_output(x,w,b):
  """
  Args:
  x (ndarray): dataset with shape (m,n); m examples and n features
  w (1d-array): weights with shape (n,); n weights
  b (scalar): bias with shape (); 1 value
  
  Returns:
  f_wb (ndarray): output predictions with shape (m,); m predictions
  """

  m, n = x.shape
  f_wb = np.zeros(m)
  
  for i in range(m):
    f_wb[i] = np.dot(w, x[i]) + b
  
  return f_wb
```


Ideally, our model should minimise the error between our predictions, **f<sub>w,b</sub>(x)**, and actual values, **y**. 

We can define the total errors that our model makes using the **" Mean Squared Error" (MSE) cost function**

**J(w, b) = (1 / 2m) · ∑ (f<sub>w,b</sub>(x<sup>(i)</sup>) − y<sup>(i)</sup>)²**

where  J(w, b), our cost, is a function of our model parameters **w** and **b**.

2) Let's **compute** our cost function J(w, b)

```python
def compute_cost(x,y,w,b):
  """
  Args:
  x (ndarray): dataset with shape (m,n); m examples, n features
  y (1d-array): array of output values with shape (m,); m values
  w (ndarray): array of weights with shape (n,); n weights
  b (scalar): bias with shape (); 1 value
  
  Returns:
  J (float): Cost (i.e. model error) from using parameters (w, b) to predict y using x
  """

  m, n = x.shape
  J = 0 # Initialise cost
  f_wb = compute_model_output(x,w,b) # generate predictions for the model
  
  for i in range(m):
    error = (f_wb[i] - y[i]) ** 2
    J += error
  J = (1 / (2 * m)) * J
  
  return J
```

We've computed J(w, b), which highlights the model's prediction error. 

Now, we want to find values (w, b) that minimise J. 

This process is called Gradient Descent, where:

**w ← w − α · ∂J(w, b)/∂w**  
**b ← b − α · ∂J(w, b)/∂b**
