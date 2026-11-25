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

#### Lesson 1 revisit: beginner-friendly recap (2026-11-23)

- In 2D, `[x, y]` gives horizontal and vertical movement from the start:
  positive x is right, negative x is left; positive y is up, negative y is
  down. Vectors can describe points, movements, or directions.
- **Magnitude** is the straight-line length. For `[3, 4]`, the right-angle
  triangle has sides 3 and 4: `3*3 + 4*4 = 25`, and `sqrt(25) = 5`.
- **Normalization** makes a vector length 1 by dividing every part by the
  original length. `[3,4]` becomes `[0.6,0.8]`: both movements shrink by the
  same factor, so the direction stays the same.
- Add and subtract vectors position by position. Multiply every part by a
  single scalar to stretch, shrink, or reverse the direction.
- A **dot product** multiplies numbers at matching positions and adds those
  products, returning one number. For `[2,1]` and `[3,4]`, it is
  `2*3 + 1*4 = 10`. **Cosine similarity** adjusts for vector lengths so it
  compares direction; near 1 means similar direction.
- A **matrix** is a table of weights. Each row combines the input numbers into
  one output number. This is the basic operation in a neural-network layer.
- **Projection** keeps the part of one vector along a chosen direction. The
  leftover points at a right angle to that direction. Example: `[6,2]` onto
  right `[1,0]` is `[6,0]`.
- **Independent** vectors each add a new direction. `[2,0]` adds none beside
  `[1,0]`, because it is just twice as long. **Rank** counts the independent
  directions. A **basis** is a smallest independent set that can reach every
  point in the space; right and up form a basis for a flat 2D grid.
- **Gram-Schmidt** removes from each new vector the parts already covered by
  earlier directions, skips a vector if nothing remains, and makes each
  remaining vector length 1. The result has unit-length, perpendicular vectors.
- Revisited and ran the lesson's vector program. Explained the dot product as
  one number, solved magnitude and normalization exercises, identified
  redundant directions, and explained why Gram-Schmidt skips them.

#### Lesson 1: remaining concepts covered (2026-11-25)

- A matrix transforms a vector. Its columns show where the basic right and up
  directions move. The scaling matrix `[[2,0],[0,3]]` sends `[x,y]` to
  `[2*x,3*y]`; it doubles horizontal movement and triples vertical movement.
- **Matrix-matrix multiplication** chains transformations. Each result cell
  comes from multiplying a row of the left matrix by a column of the right,
  then adding. This differs from element-wise multiplication, which multiplies
  values in matching positions. The left matrix's column count must equal the
  right matrix's row count; e.g. `(2 by 3) @ (3 by 2)` produces `(2 by 2)`.
- A **transpose** swaps rows and columns without changing the values. It is
  useful when a calculation needs columns in the place of rows; neural-network
  backpropagation uses transposed weight matrices to send gradients backward.
- The **angle between vectors** can be found by first dividing their dot
  product by both lengths (cosine similarity), then using inverse cosine
  (`acos`) to get an angle. `math.degrees` converts radians to degrees. Cosine
  similarity requires nonzero vectors; implementations clamp tiny rounding
  errors to the valid range -1 through 1 before calling `acos`.
- Dot products also score how well two AI vectors match. Embeddings represent
  words, images, or users as number lists. In attention, a query vector
  represents what a token seeks and key vectors represent information other
  tokens offer; their dot products are relevance scores.
- Exact duplicate or scaled features add no new information. In regression,
  many weight combinations can then produce the same predictions. This is
  **multicollinearity**. Full column rank means the columns are independent;
  deficient rank means redundancy and possibly no unique solution. Nearly dependent
  columns make weights sensitive to small data changes. Regularization prefers
  smaller, steadier weights; SVD truncation can remove weak directions.
- Projection applications: linear regression finds the closest point on its
  fitted line or plane; PCA keeps directions with the most variation and
  projects data onto them. The leftover after projection is the part not
  explained by the chosen direction or surface.
- **QR decomposition** writes a matrix as `A = Q @ R`. `Q` contains
  orthonormal directions produced with Gram-Schmidt; `R` records the
  coefficients needed to rebuild the original matrix. QR helps solve systems
  and calculate eigenvalues stably. Whitening uses clean directions to remove
  feature correlations and balance their scales.
- NumPy provides `np.linalg.matrix_rank`, `np.linalg.qr`, and ready-made
  projection arithmetic. PyTorch tensors can track how a result changes with
  its inputs: `requires_grad=True` asks PyTorch to track the calculation, and
  `.backward()` fills in `.grad`. For `dot(x, [1,0,0])`, the gradient with
  respect to `x` is `[1,0,0]` because only the first input affects that result.
- **LoRA** represents a large weight update with two smaller matrices. For a
  4096-by-4096 update, the full update has about 16.8 million values, while
  4096-by-16 and 16-by-4096 factors together have about 131,000. The middle
  dimension 16 limits the update to a smaller set of directions; it does not
  delete rows from the original model weights.
- The lesson's Julia example performs the same vector, dot-product, and
  matrix-vector operations with Julia syntax; the math is unchanged.
- Lesson exercises also practice finding vector angles, scaling points,
  comparing word-like vectors, checking orthonormal vectors, building a
  rank-2 3-by-3 matrix, and projecting `[1,2,3]` onto `[1,1,1]`.

### Lesson 2: Vectors, Matrices & Operations

- A matrix is a table of numbers. Its **shape** is rows by columns; a 2-by-3
  matrix has two horizontal rows and three vertical columns.
- **Element-wise multiplication** multiplies values in matching positions.
  The matrices need the same shape. **Matrix multiplication** instead takes a
  row from the first matrix and a column from the second, multiplies their
  matching values, and adds them to get one result value. For shapes
  `(m by n) @ (n by p)`, the middle sizes must match and the result is
  `(m by p)`. Example:
  `[[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]`.
- Matrix addition and subtraction combine matching positions. Scalar
  multiplication multiplies every entry by one number. A **transpose** turns
  every old column into a new row; for example,
  `[[1, 2, 3], [4, 5, 6]]` becomes `[[1, 4], [2, 5], [3, 6]]`.
- The **determinant** is a number that tells whether a square matrix can be
  undone. For `[[a, b], [c, d]]`, it is `a*d - b*c`. A zero determinant means
  the transformation squashed information together, so there is no inverse.
  A nonzero determinant allows an **inverse** matrix that undoes the original;
  multiplying a matrix by its inverse gives the **identity matrix**, with ones
  on the diagonal and zero elsewhere.
- A neural network's dense layer first computes `W @ x`: each row of `W`
  creates one output by mixing the input values. It adds a **bias**, an
  adjustment value for each output, then applies **ReLU**, which replaces
  negative results with zero. With `W` shaped 2-by-3 and `x` containing 3
  values, the result has 2 values.
- **Broadcasting** lets a smaller bias be added to each row of a larger output
  table. For example, adding `[10, 20, 30]` to `[[1,2,3],[4,5,6]]` gives
  `[[11,22,33],[14,25,36]]`: each bias entry is added to the matching position
  in both rows.
- NumPy writes these operations as `A * B` for element-wise multiplication,
  `A @ B` for matrix multiplication, `.T` for transpose, and
  `np.linalg.det(A)` / `np.linalg.inv(A)` for determinant and inverse.
  `np.maximum(0, z)` applies ReLU. NumPy performs the inner arithmetic in
  optimized code; the programmer still chooses the operation and supplies
  compatible shapes.
- Ran `phases/01-math-foundations/02-vectors-matrices-operations/code/matrices.py`.
  The output showed element-wise multiplication `[[5,12],[21,32]]` differs from
  matrix multiplication `[[19,22],[43,50]]`, and multiplying a matrix by its
  inverse produced the identity. The neural-network example mapped a 3-value
  input through a 4-value hidden layer to 2 outputs.

### Lesson 3: Matrix Transformations

- A **transformation** is a rule that moves points. A 2D point is written
  `[x, y]`. A matrix can rotate, stretch, slant, or mirror every point in a
  picture. The columns of a transformation matrix show where the basic right
  and up directions move.
- **Rotation** turns points around the origin while preserving lengths and
  angles. In Python, `math.pi / 2` is a 90-degree angle; `sin` and `cos` supply
  the matrix values. A 90-degree counterclockwise rotation maps `[1, 0]` to
  `[0, 1]` and `[0, 1]` to `[-1, 0]`.
- **Scaling** changes horizontal and vertical coordinates separately. Scaling
  by `(2, 3)` maps `[1, 1]` to `[2, 3]`. **Shearing** slants a shape; an x-shear
  moves x by `kx * y`, leaving y unchanged when `ky=0`. **Reflection** mirrors
  across an axis; reflecting across the y-axis changes x's sign and keeps y.
- **Composition** applies several transformations in sequence. For a point
  `[1, 0]`, rotating 90 degrees then scaling by `(2, 0.5)` gives `[0, 0.5]`;
  scaling first then rotating gives `[0, 2]`. In `S @ R @ point`, the rightmost
  operation, `R`, acts first. Matrix multiplication order therefore matters.
- A **3D rotation** turns points around an axis. A rotation around z spins the
  x-y plane and keeps z fixed; around x it spins the y-z plane; around y it
  spins the x-z plane. In the program, rotating `[1, 0, 0]` around x leaves it
  where it is because the point lies on that axis.
- The **determinant** measures area scaling in 2D (volume in 3D). A value of 1
  preserves area, 6 multiplies it by six, -1 preserves area but flips
  orientation, and 0 crushes the space into a lower dimension. The script
  showed rotation and shear with determinant 1, scaling by `(2,3)` with 6,
  reflection with -1, and a collapsed matrix with 0.
- An **eigenvector** is a direction a matrix does not turn; it only stretches,
  shrinks, or flips that direction. Its **eigenvalue** is the multiplier. For
  `A = [[2,1],[1,2]]`, `[1,1]` becomes `[3,3]` (eigenvalue 3), while `[1,-1]`
  stays unchanged (eigenvalue 1). Eigenvectors may be normalized to length 1;
  reversing all signs describes the same direction.
- For a 2-by-2 matrix, the scratch eigenvalue function uses the diagonal sum
  (trace), determinant, and `trace**2 - 4*determinant` to calculate the two
  eigenvalues. NumPy's `np.linalg.eig(A)` does this in a library call and
  returns eigenvectors as columns. A pure 90-degree rotation has complex
  eigenvalues and no real direction that stays fixed.
- **Eigendecomposition** writes a matrix as `A = V @ D @ V^-1`: `V` stores
  eigenvector directions, `D` stores their stretch amounts on its diagonal,
  and `V^-1` changes coordinates into those directions. It means: switch to
  the special directions, stretch each one, then switch back.
- In **PCA**, eigenvectors of the covariance matrix give data directions and
  their eigenvalues give the amount of variation in each. PCA keeps directions
  with large eigenvalues first. In recurrent networks, repeatedly multiplying
  by eigenvalues with magnitude above 1 can make values/gradients grow
  explosively; magnitudes below 1 can make them shrink toward zero.
- Ran `phases/01-math-foundations/03-matrix-transformations/code/transformations.py`
  with the journey repository's `.venv` Python and NumPy 2.5.4. The
  from-scratch and NumPy sections agreed on eigenvalues 3 and 1, the matching
  directions, and the PCA rule to keep the direction with greatest variance.

## Questions initially missed or corrected

- **Question:** How does LoRA use linear algebra to make fine-tuning a large
  language model more efficient?
  - **Initial answer:** It removes unused rows from the model's weight matrices.
  - **Correct answer:** It represents the weight update using two smaller
    matrices instead of updating the full large matrix.
  - **Simple explanation:** A 4096-by-4096 update has about 16.8 million
    values. Two rank-16 factors have about 131,000 values. LoRA trains this
    smaller update, assuming it can capture the useful change.
- **Revisit:** The earlier LoRA misconception was reviewed. On the Phase 1,
  Lesson 1 restart, the learner correctly chose that LoRA represents a large
  weight change using two smaller matrices.
- **Build correction:** Initially transposed `[[1, 2, 3], [4, 5, 6]]` as
  `[[1, 2], [3, 4], [5, 6]]`. The correct result is
  `[[1, 4], [2, 5], [3, 6]]`: each original column becomes a new row. A second
  example was answered correctly as `[7, 9]`, the first row from the first
  column of `[[7, 8], [9, 10]]`.
- **Broadcasting check:** Correctly explained that each bias value is added
  to the matching position in every row of the output table.
- **Answer clarification:** The eigendecomposition quiz response was typed as
  C by mistake; the learner clarified they intended D. The intended answer
  matches the concept explained and recalled: `[1,1]` is stretched by 3, while
  `[1,-1]` is unchanged.

## Quiz results

- Lesson 1, Linear Algebra Intuition: **2/3 post (66.7%)**. Review LoRA's
  low-rank update and how two smaller matrices represent a large weight change.
- Lesson 2, Vectors, Matrices & Operations: **2/2 post (100%)**.
- Lesson 3, Matrix Transformations: **3/3 post (100%)**. The learner clarified
  that the final response was a typing slip and intended the correct answer.
- Lesson 1 revisit, Linear Algebra Intuition: **3/3 post (100%)**. The earlier
  LoRA review item was answered correctly on the restart.
- Lesson 1 coverage revisit: the omitted matrix operations, AI connections,
  NumPy/PyTorch uses, QR, and LoRA details were explained. No additional quiz
  was given.
