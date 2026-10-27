
import random

order = list(range(len(items)))
random.Random(seed).shuffle(order)

n_test = round(len(items) * 0.2)
test = [items[i] for i in order[:n_test]]
train = [items[i] for i in order[n_test:]]