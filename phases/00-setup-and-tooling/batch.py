def  batches(items, batch_size):
    if batch_size < 1:
        return
    iterator = iter(items)
    while True:
        batch = []
        for n in range(batch_size):
            item = next(iterator, "EMPTY")
            if item == "EMPTY":
                break
            else:
                batch.append(item)
        if batch != []:
            yield batch
        else:
            break

# The version provided by the AI tutor.
def batches(items, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    iterator = iter(items)

    while True:
        batch = []

        for _ in range(batch_size):
            try:
                batch.append(next(iterator))
            except StopIteration:
                break

        if not batch:
            break

        yield batch
