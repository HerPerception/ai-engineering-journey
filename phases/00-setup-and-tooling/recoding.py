# Task 1
import sys
print(f"The Python version is: {sys.version.split()[0]}")


# Task 2
a = [1, 2, 3]
# b = a would create a new variable that points to or references the same object. Any change made to one would affect the other. 
# A better option would be to copy the object being referenced by the first variable into the second variable explicitly.
b = list(a)
b.append(4)
print(a)


# Task 3
def dot(a, b):
    dot_product = 0
    for i in range(len(a)):
        dot_product += (a[i] * b[i])
    return dot_product

print(dot([1, 2, 3], [1, 2, 3]))


# Task 4
import math
def softmax(logits):
    largest_num =max(logits)
    exp_logits = [math.exp(x - largest_num) for x in logits]
    total = sum(exp_logits)
    # for idx in range(len(exp_logits)):
    #     exp_logits[idx] /= total
    probabilities = [exp_x/total for exp_x in exp_logits]
    return probabilities

print(softmax([1000, 1001, 1002]))


# Task 5
# w = weight, grad = gradient, lr = learning rate
def gd_step(w, grad, lr=0.1):
    return w-(lr*grad)

print(gd_step(2.0, 4.0, 0.1))
