def square_root(vector: list) -> int:
    square_total = 0
    for num in vector:
        square_total += num ** 2
    return square_total ** 0.5

def get_dot_product(vector1: list, vector2: list) ->  int:
    dot_product = 0
    for idx in range(len(vector1)):
        dot_product += vector1[idx] * vector2[idx]
    return dot_product


def project_onto(a, b):
    v = []
    for i in range(len(a)):
        val = a[i] * b[i]
        v.append(val)
    total = sum(v)
    total_val = 0
    for i in range(len(b)):
        total_val += b[i] * b[i]
        
    scale = total/total_val
    shadow = []
    for i in range(len(b)):
        val = b[i] * scale
        shadow.append(val)
    residual = []
    for i in range(len(a)):
        leftover = a[i] - shadow[i]
        residual.append(leftover)
    return shadow, residual

v1 = [1, 2, 3]
v2 = [4, 5, 6]
dot_product = get_dot_product(v1, v2)
root1 = square_root(v1)
root2 = square_root(v2)

cosine_similarity = dot_product/ (root1 * root2)
print(cosine_similarity)
print(project_onto(v1, v2))
