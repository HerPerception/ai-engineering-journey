import numpy as np
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j]
             for j in range(self.cols)]
            for i in range(self.rows)
        ])
    
    # Matrix multiplication multiplies each row in the first matrix or grid by each column in the second.
    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j]
                    for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])


    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("No inverse exists")
        return Matrix([
            [ self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det,  self.data[0][0] / det]
        ])


    # For each position, it multiplies the number in this matrix by the number in the same position of the other matrix. Both matrices need the same number of rows and columns.
    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j]
             for j in range(self.cols)]
            for i in range(self.rows)
        ])
    
# W @ x makes two values, one from each row of W. + bias adds an adjustment to each value. np.maximum(0, z) applies ReLU: it replaces any negative result with zero. The output is [[0.], [1.3]].
W = np.array([[1., -1., 0.],
              [2.,  0., 1.]])
x = np.array([[0.5],
              [0.8],
              [0.2]])
bias = np.array([[0.1],
                 [0.1]])

z = W @ x + bias
output = np.maximum(0, z)
print(output)

