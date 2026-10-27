from dataclasses import dataclass
weights = [0.5, 1.2]
backup = weights.copy()
# backup = weights - ths would make backup point to the same 
# list as weight. It does not create a new list.
backup.append(3.0)
print(weights)

shape = (32, 784)
batch_size, feature_count = shape  # unpack the two values

config = {"learning_rate": 0.01, "epochs": 5}
tokens = {"cat", "dog", "cat"}  # repeats collapse to one "cat"

print(f"Batch size: {batch_size}, features: {feature_count}")
print(f"Vocabulary size: {len(tokens)}")
print(tokens)


scores = [0.91, 0.42, 0.77, 0.15]

high_scores = [score for score in scores if score > 0.6]
labels = [1 if score >= 0.5 else 0 for score in scores]

print(high_scores)
print(labels)


def add_bad(item, bucket=[]):
    bucket.append(item)
    return bucket

print(add_bad("a"))
print(add_bad("b"))

def add_item(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket

print(add_item("a"))
print(add_item("b"))



@dataclass(frozen=True)
class Record:
    features: tuple[float, ...]
    label: int

example = Record((1.5, 2.0), 1)
print(example)


squares = [n * n for n in range(3)]       # list: [0, 1, 4]
square_stream = (n * n for n in range(3)) # generator

print(list(square_stream))
print(list(square_stream))

with open("data.csv", encoding="utf-8") as file:
    text = file.read()

try:
    label = int("cat")
except ValueError:
    print("The label must be a whole number.")
