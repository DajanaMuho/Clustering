from Cure import *
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_blobs

# Import the dataset
X, y = make_blobs(centers=5, random_state=1)
dataset = X
print('Shape of X: ', X.shape)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

# Set input params
num_cluster = 4
alpha = 0.1
num_representative_points = 5

# Run algorithm
start = time.process_time()
Y_pred = CURE(dataset, num_representative_points, alpha, num_cluster)
end = time.process_time()
print("The time of CURE algorithm is", end - start, "\n")
# Print the clusters
plt.scatter(X[:, 0], X[:, 1], c=Y_pred)
plt.show()
