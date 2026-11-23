
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.dot(a, b))         # dot product
print(np.linalg.norm(a))    # length

W = np.array([[1, 1, 1], [2, 0, -1]])
x = np.array([1.0, 0.5, -0.3])
print(W @ x)                # matrix transforms the input
