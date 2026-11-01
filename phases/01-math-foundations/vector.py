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