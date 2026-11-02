class Vector:
    def __init__(self, components):
        self.components = list(components)

    def __add__(self, other):
        return Vector([
            a + b
            for a, b in zip(self.components, other.components)
        ])

    def __sub__(self, other):
        return Vector([
            a - b
            for a, b in zip(self.components, other.components)
        ])
    def dot(self, other):
        return sum(
            a * b
            for a, b in zip(self.components, other.components)
        )

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5
    
    def cosine_similarity(self, other):
        return self.dot(other) / (
            self.magnitude() * other.magnitude()
        )
    
    def project_onto(a, b):
        scale = a.dot(b) / b.dot(b)
        return Vector([scale * value for value in b.components])
    

import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))  # 32

W = np.array([[0.1, -0.2, 0.3],
              [0.4,  0.5, -0.1]])
x = np.array([1.0, 0.5, -0.3])
print(W @ x)  # matrix times vector