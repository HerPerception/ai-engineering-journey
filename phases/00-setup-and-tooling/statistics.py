import math

def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))

def mean(values):
    if not values:
        raise ValueError("mean needs at least one value")
    return sum(values) / len(values)

def std(values):
    average = mean(values)
    variance = sum((value - average) ** 2 for value in values) / len(values)
    return math.sqrt(variance)