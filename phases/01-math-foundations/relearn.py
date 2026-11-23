class Vector:
    def __init__(self, components):
        self.components = list(components)

    def dot(self, other):
        return sum(
            a * b
            for a, b in zip(self.components, other.components)
        )
    
    def magnitude(self):
        return sum(x * x for x in self.components) ** 0.5

    def normalize(self):
        length = self.magnitude()
        return Vector([x / length for x in self.components])
    
    def cosine_similarity(self, other):
        return self.dot(other) / (
            self.magnitude() * other.magnitude()
        )

    def apply_matrix(matrix, vector):
        outputs = []
        for row in matrix:
            total = 0
            for position in range(len(vector)):
                total += row[position] * vector[position]
            outputs.append(total)
        return 
    

    def project_onto(vector, direction):
        factor = vector.dot(direction) / direction.dot(direction)
        return Vector([
            factor * value
            for value in direction.components
        ])
    
    def gram_schmidt(vectors):
        kept = []
        for vector in vectors:
            remainder = vector
            for old_direction in kept:
                remainder = remainder - remainder.project_onto(old_direction)
            if remainder.magnitude() < 1e-10:
                continue
            kept.append(remainder.normalize())
        return kept