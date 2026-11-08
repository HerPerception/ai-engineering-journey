# Scaling stretches or shrinks the x and y coordinates:
def scaling_2d(sx, sy):
    return [[sx, 0], [0, sy]]

# mat_vec_mul multiplies a matrix by a point (vector) to calculate the point's new coordinates after a transformation, like a rotation or scaling.
def mat_vec_mul(matrix, point):
    return [
        sum(matrix[i][j] * point[j] for j in range(len(point)))
        for i in range(len(matrix))
    ]

#  Reflection flips a point like a mirror.
def reflection_y():
    return [[-1, 0], [0, 1]]


#A shear slants a shape. An x-shear moves a point sideways by an amount based on how high it is:
def shearing_2d(kx, ky):
    return [[1, kx], [ky, 1]]

shear = shearing_2d(kx=0.5, ky=0)
new_point = mat_vec_mul(shear, [2, 3])


import math
# A rotation turns a point around the center without stretching it. Python measures angles in radians: math.pi / 2 means 90 degrees. The functions cos and sin help calculate the numbers the rotation matrix needs:
# WHAT IS COUNTERCLOCKWISE?
# It means moving in the opposite direction of the clock's hand. 
# • Clockwise moves from top to right, then down, then left (12 → 3 → 6 → 9 → 12).
# • Counterclockwise moves from top to left, then down, then right (12 → 9 → 6 → 3 → 12).

# WHY DOES THE PYTHON CODE USE COSINE AND SINE? 
# The grid above works only for 90 degrees. To rotate a point by any random angle (like 12 degrees or 45 degrees), the grid numbers must change dynamically.
# Trigonometric functions—\(\cos(\theta)\) and \(\sin(\theta)\)—are simply tools that calculate the exact grid numbers needed for whatever angle (\(\theta \)) you choose.
def rotation_2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]

rotate_90 = rotation_2d(math.pi / 2)
new_point = mat_vec_mul(rotate_90, [1, 0])


# Composition means applying more than one transformation in sequence. Try rotating [1, 0] by 90 degrees, then scaling x by 2 and y by 0.5:

# 1. Rotation moves [1, 0] to [0, 1].
# 2. Scaling then moves [0, 1] to [0, 0.5].

# If you scale first, [1, 0] becomes [2, 0]; rotating that gives [0, 2]. The results differ because the order changed.

# The code shows both orders:


from matrix import Matrix 
R = rotation_2d(math.pi / 2)
S = scaling_2d(2, 0.5)

# Creating an instance of the class 'Matrix' so I can import one of its method 'mat_mul':
mat = Matrix()
# mat_mul is a function that takes two matrices and returns their matrix product:
rotate_then_scale = mat.mat_mul(S, R)
scale_then_rotate = mat.mat_mul(R, S)

# In S @ R, the point gets R first, then S: the matrix closest to the point acts first. Swapping them changes the result.

# In S @ R, which happens first: rotation or scaling? rotation happens first.


