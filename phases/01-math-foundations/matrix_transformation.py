def scaling_2d(sx, sy):
    return [[sx, 0], [0, sy]]

def mat_vec_mul(matrix, point):
    return [
        sum(matrix[i][j] * point[j] for j in range(len(point)))
        for i in range(len(matrix))
    ]


def reflection_y():
    return [[-1, 0], [0, 1]]


#A shear slants a shape. An x-shear moves a point sideways by an amount based on how high it is:
def shearing_2d(kx, ky):
    return [[1, kx], [ky, 1]]

shear = shearing_2d(kx=0.5, ky=0)
new_point = mat_vec_mul(shear, [2, 3])


import math

def rotation_2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]

rotate_90 = rotation_2d(math.pi / 2)
new_point = mat_vec_mul(rotate_90, [1, 0])