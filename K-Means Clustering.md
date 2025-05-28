# **K-Means Clustering**
##
**K-Means Clustering** is an unsupervised ML technique that divides unlabelled data into "clusters", based on shared characteristics. 

Examples of when this technique might be used include:
- Customer Segmentation
- Image Compression
- Document clustering (by topic)

The process of K-Means Clustering is:

1) **Randomly plot** 'K' cluster centroids (where K is an integer, denoting number of centroids)

```python
def random_initialisation_centroids(X,K):
  """
  Args:
  X (ndarray): training set of shape (m,n); m examples, n features
  K (int): desired number of centroids
  
  Returns:
  centroids: initial coordinates of centroids with shape (K, n); K centroids, n coordinates
  """

  randidx = np.random.permutation(X.shape[0]) # Randomly reorder the indices of examples

  # We will then take K random examples as the first coordinates for our centroids

  centroids = X[randidx[:K]

  return centroids
```


2) Once we have the initial coordinates for our centroids, we **assign** points to one of our K-centroids based on proximity


```python
def find_closest_centroids(X, centroids):
  """
  Args:
  X (ndarray): training set of shape (m,n); m examples, n features
  centroids (ndarray): coordinates of centroids with shape (K,n); K centroids, n coordinates
  
  Returns:
  idx (1d-array): list with each value storing an integer from 0 -> K-1, corresponding to the centroid closest to each example, with shape (m,); m examples
  """

  K = centroids.shape[0] # Set value of K

  # Create ([0,0,0,...]) of length m - one '0' for each of our m-examples.
  #These '0's will be replaced by an integer from 0 -> K - 1, corresponding to the centroid that each example of X is closest to.
  idx = np.zeros(X.shape[0], dtype = int)

  #Start by iterating through each of the examples
  for i in range(len(idx)):
    # Initialise a unique list for each example, that will store its distance from all centroids
    distance = []
    for j in range(K): # Iterate through all centroids
      # Calculate the distances and append this to the list
      distance.append(np.linalg.norm(X[i] - centroids[j])
    #For each example, we will return the index of the cluster that is the closest proximity (minimum distance)
    idx[i] = np.argmin(distance) 

  return idx
```


3) Once we have assigned examples to cluster centroids, **update** the position of the centroids based on the mean coordinates of the current assigned examples.

```python
def compute_centroids(X, idx, K):
  """
  Args:
  X (ndarray): training set of shape (m,n); m examples, n features
  idx (1d-array): an array of shape (m,) where each element is an integer from 0 to K-1, indicating the index of the closest centroid for each of the m examples.
  K (int): number of centroids
  
  Returns:
  centroids (ndarray): updated coordinates of centroids with shape (K, n); K centroids, n coordinates
  """

  # Start by iterating through each of the K centroids
  for j in range(K):
    # Extract the examples that have been assigned to centroid 'j'
    examples_assigned_to_centroid_j = X[idx == j]
    # From those examples, extract mean coordinates.
    # Important note: "axis = 0" means taking the mean across columns rather than rows!
    centroids[j] = np.mean(examples_assigned_to_centroid_j, axis = 0)

  return centroids
```

4) Once we have updated the position of the centroids, **iterate** over the process! Reassign examples to centroids, update position of centroids based on newly assigned examples...
```python
def runkMeans(X, initial_centroids, max_iters = 10):
  """
  Args:
  X (ndarray): training set of shape (m,n); m examples, n features
  initial_centroids (ndarray): initial coordinates of centroids with shape (K, n); K centroids, n features
  max_iters (int): How many iterations do you want to run K-means clustering for?
  
  Returns:
  centroids (ndarry): final coordinates of centroids with shape (K, n); K centroids, n features
  idx (1d-array): list of values from 0 to K-1, showing (final) corresponding centroid id for each example; shape(m,); m examples
  """

  m,n = X.shape #
  K = initial_centroids.shape[0]
  centroids = initial_centroids
  previous_centroids = centroids
  idx = np.zeros(m)
  
  #run K-means iteratively
  for i in range(max_iters):
    idx = find_closest_centroid(X, centroids)
    centroids = compute_centroids(X, idx, K)
  
  return centroids,idx
```

