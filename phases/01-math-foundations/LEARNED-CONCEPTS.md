# Phase 1: Math Foundations — Learned Concepts

Plain-English notes from completed lessons will be added here. Missed questions and their corrections will also be recorded here.

## Concepts learned

### Lesson 1: Linear Algebra Intuition

- A **vector** is an ordered list of numbers. It can represent coordinates,
  direction, or an AI representation such as a word or image embedding. A
  vector's **magnitude** is its length; **normalizing** it changes its length
  to 1 while keeping its direction.
- The **dot product** multiplies matching entries and adds the results. It
  measures how aligned two vectors are, and is used in attention and similarity
  search. **Cosine similarity** divides the dot product by both vector lengths,
  focusing on direction: near 1 means similar direction, near 0 means
  perpendicular, and near -1 means opposite direction.
- A **matrix** is a grid of numbers that transforms a vector. Multiplying a
  matrix with *m* rows by a compatible vector produces *m* numbers. Neural
  network layers use a weight matrix to transform input features into output
  features. In Python, `@` means matrix multiplication; NumPy performs these
  operations with optimized code.
- **Projection** keeps the part of one vector that points along another:
  `proj_b(a) = (a · b / b · b) b`. The leftover, `a - proj_b(a)`, is
  perpendicular to `b`. For example, projecting `[3, 4]` onto `[1, 0]` gives
  `[3, 0]`; the residual is `[0, 4]`.
- Vectors are **linearly independent** when none can be made by combining the
  others. In `[1,0,0]`, `[0,1,0]`, `[2,1,0]`, the third is `2` times the first
  plus the second, so it adds no new direction. **Rank** counts independent
  columns/directions. A **basis** is a minimal independent set that spans the
  space. Redundant features can make model weights hard to determine reliably.
- **Gram-Schmidt** builds perpendicular unit vectors from independent input
  vectors: remove the components along directions already kept, then normalize
  what remains. A near-zero remainder means the input was dependent, so it is
  skipped. An **orthonormal** set has length 1 vectors that are pairwise
  perpendicular. QR decomposition and stable numerical solvers use this idea.
- **LoRA (Low-Rank Adaptation)** represents a large model weight update as the
  product of two smaller matrices. For a 4096-by-4096 update, two rank-16
  factors need about 131,000 values instead of about 16.8 million. It assumes
  the useful update lies in a much smaller set of directions, reducing the
  number of values that need training.
- Ran `phases/01-math-foundations/01-linear-algebra-intuition/code/vectors.py`.
  Its output showed a 90-degree matrix rotation, a perpendicular projection
  residual, rank 1 for a redundant 2-by-2 matrix, and Gram-Schmidt vectors with
  length 1 and pairwise dot products near zero (small floating-point rounding).

## Questions initially missed or corrected

- **Question:** How does LoRA use linear algebra to make fine-tuning a large
  language model more efficient?
  - **Initial answer:** It removes unused rows from the model's weight matrices.
  - **Correct answer:** It represents the weight update using two smaller
    matrices instead of updating the full large matrix.
  - **Simple explanation:** A 4096-by-4096 update has about 16.8 million
    values. Two rank-16 factors have about 131,000 values. LoRA trains this
    smaller update, assuming it can capture the useful change.

## Quiz results

- Lesson 1, Linear Algebra Intuition: **2/3 post (66.7%)**. Review LoRA's
  low-rank update and how two smaller matrices represent a large weight change.
