def dot(first, second):
        return sum(
            a * b
            for a, b in zip(first, second)
        )

print(dot([1, 2, 3], [1, 2, 3]))