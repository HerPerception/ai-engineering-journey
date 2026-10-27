from collections import Counter, defaultdict, deque

labels = ["cat", "dog", "cat", "bird", "cat"]
counts = Counter(labels)

positions = defaultdict(list)
for index, label in enumerate(labels):
    positions[label].append(index)

recent_losses = deque([1.2, 1.1, 0.9], maxlen=3)
# Counter counts repeated labels. defaultdict(list) groups items 
# under a key without making you check whether the key already 
# exists. enumerate gives each item and its index. deque(maxlen=3) 
# holds only the three most recent values; adding a new one drops 
# the oldest.