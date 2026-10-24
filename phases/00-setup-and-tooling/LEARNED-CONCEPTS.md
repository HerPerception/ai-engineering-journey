# Phase 0: Setup & Tooling — Learned Concepts

Plain-English notes from completed lessons. New concepts and missed questions will be added as lessons are completed.

## Concepts learned

### The environment has four layers

An AI development setup is built from the bottom up:

1. **System foundation:** the operating system, shell, Git, editor, and hardware drivers.
2. **Package manager:** installs software packages and manages their dependencies.
3. **Language runtime:** runs code, such as the Python interpreter.
4. **AI libraries:** provide tools such as NumPy and PyTorch.

Each layer depends on the layer below it. If Python is missing, the missing part is the language runtime. Install the system foundation first, then the package manager, runtime, and libraries.

### Package managers and virtual environments

`uv` installs Python packages and works out which package versions fit together. A virtual environment keeps one project's packages separate from another project's packages, so changing one project is less likely to break another.

### A dot product

A dot product multiplies matching values in two lists, then adds those products. For `[1, 2, 3]` dotted with itself: `1×1 + 2×2 + 3×3 = 14`.

A Python loop can show each multiply and addition. NumPy's `np.dot` performs those steps inside the library. In this workspace, Python was installed but NumPy was not, so the NumPy example stopped at the import step.

### Checking GPU access

`nvidia-smi` reports information about NVIDIA hardware and its driver. To check whether PyTorch itself can use CUDA, run `import torch; print(torch.cuda.is_available())`. On Apple Silicon, check `torch.backends.mps.is_available()` instead.

### Git saves a history of project changes

The usual path is **working directory → staging area → local repository → remote repository**:

- `git add` selects edits for the next save.
- `git commit` records the staged edits in the local history.
- `git push` sends committed work to a remote, such as GitHub.
- `git status` shows which edits are staged, unstaged, or untracked.

Git commits let you inspect when changes were recorded and restore earlier work. An edit that was not staged with `git add` is not included in the commit.

### Branches isolate experiments

`git checkout -b experiment/name` creates a branch and switches to it. Work committed on that branch does not change `main` until it is merged. Branches make it safer to try an idea without disturbing the main line of work.

### Ignore large model checkpoints

Model checkpoint files such as `.pt`, `.pth`, and `.safetensors` can be very large. A `.gitignore` can keep them out of ordinary Git history so the repository stays manageable.

### Lesson 3: GPU Setup & Cloud

#### Processors and parallel work

- A **CPU** (central processing unit) is the computer's general-purpose
  processor. It has a small number of powerful cores that handle many kinds of
  tasks well.
- A **GPU** (graphics processing unit) is a processor built to do many similar
  calculations at the same time. This is called **parallel work**. Neural
  networks repeatedly multiply large matrices, so that kind of work can suit
  a GPU. A GPU is not automatically faster for every task; small jobs and data
  movement can reduce or erase its advantage.
- **Matrix multiplication** combines rows and columns of numbers using
  multiply-and-add operations. It is one of the main calculations in neural
  networks. In Python, PyTorch uses `@` for matrix multiplication.
- A **tensor** is PyTorch's container for numbers. A tensor can live in CPU
  memory or GPU memory. PyTorch checks where the tensors live and selects the
  matching implementation: `a_cpu @ b_cpu` runs on the CPU, while
  `a_gpu @ b_gpu` runs on the GPU. The operation looks the same in code even
  though PyTorch uses different device code underneath.
- A **Tensor Core** is a specialized part of some NVIDIA GPUs designed to do
  matrix calculations used in AI efficiently. The exact speed benefit depends
  on the GPU, data type, and operation.

#### NVIDIA hardware, driver, CUDA, and PyTorch

- The **GPU driver** is system software that lets the operating system and
  programs communicate with an NVIDIA card. The card is the hardware; the
  driver is software that helps control it.
- **CUDA** stands for **Compute Unified Device Architecture**. It is NVIDIA's
  software platform for using NVIDIA GPUs for general calculations, including
  AI. It includes programming interfaces and runtime components used by
  libraries such as PyTorch to send work to an NVIDIA GPU. CUDA is not a GPU
  card, and it is not the driver itself; CUDA software and a compatible driver
  work together to make GPU computing available.
- The **CUDA version shown by `nvidia-smi`** is the newest CUDA version the
  installed driver says it supports; it does not necessarily mean that a
  separate CUDA Toolkit of that version is installed. PyTorch's
  `torch.version.cuda` reports the CUDA version used to build that PyTorch
  installation. They describe related parts of the setup, not the same thing.
- A **CUDA-enabled PyTorch build** is a PyTorch installation that includes
  support for CUDA. Even when `nvidia-smi` can see the card, PyTorch may not be
  able to use it if PyTorch is missing, is a CPU-only build, or the software
  setup is incompatible.
- `nvidia-smi` is NVIDIA's System Management Interface command. It reports
  NVIDIA GPU and driver information, such as the GPU model, memory use, and
  activity. It answers: “Can the NVIDIA system see this card, and what is it
  doing?” It does not answer whether a particular Python environment can use
  the card.
- `torch.cuda.is_available()` asks whether the current PyTorch installation
  can use CUDA. It answers: “Can this Python program use an NVIDIA GPU through
  PyTorch?” In Colab, it returned `True` for the Tesla T4.
- `torch.cuda.get_device_properties(0)` gives information about the first GPU.
  Its `total_memory` field was used for the fp16 estimate. **Compute
  capability** is a number identifying features of an NVIDIA GPU architecture
  that software can use.
- **MPS** is Apple's Metal Performance Shaders backend for using Apple Silicon
  GPUs from PyTorch. `torch.backends.mps.is_available()` checks that route;
  it is separate from CUDA, which is NVIDIA-specific.

#### Local, hosted, and cloud GPUs

- A **local GPU** is installed in your own computer. `nvidia-smi` and
  `torch.cuda.is_available()` can check different parts of a local NVIDIA
  setup.
- **Google Colab** runs notebook code on a hosted computer. Choosing
  **Runtime → Change runtime type → GPU** gave this notebook access to a Tesla
  T4, so no local GPU setup was needed. The hardware Colab assigns can vary.
- A **cloud GPU** is a GPU on a remote rented computer, often billed by the
  hour. Colab is a convenient hosted notebook; other cloud GPU services give
  more control but require setup and usually cost money.
- Most early lessons can run on a CPU. Use a GPU when a lesson or workload
  needs the extra parallel calculation capacity.

#### Benchmarking CPU and GPU work

- A **benchmark** measures how long a particular task takes. The lesson
  creates two 4,000-by-4,000 random matrices, multiplies them on the CPU, then
  copies them to the GPU and multiplies them there.
- `time.perf_counter()` is a precise stopwatch. The code records its value
  before and after an operation; subtracting the two gives elapsed seconds.
  `_ = a @ b` means “do the multiplication and discard the result.”
- CPU calculations finish before the Python call returns. GPU calculations
  are often **asynchronous**: Python places work in the GPU's queue and can
  continue before the GPU has completed it. The GPU is still working even
  though Python has returned from the `@` line.
- To measure GPU time correctly, first copy the matrices to the GPU and call
  `torch.cuda.synchronize()` **before starting the stopwatch**. This waits for
  the copies and any earlier queued GPU work to finish, keeping that earlier
  time out of the multiplication measurement. Then start the stopwatch, submit
  `a_gpu @ b_gpu`, and call `torch.cuda.synchronize()` **again before reading
  the stopwatch**. This second wait blocks Python until the multiplication is
  finished. Without it, the timer would mostly measure how quickly Python
  submitted the work, stop too early, and report a GPU time that is too short.
- The wait before starting the stopwatch matters too: copying a tensor to CUDA
  can itself be queued GPU work. Waiting before timing ensures that the copy
  finishes outside the measured section; waiting after the matrix operation
  ensures the measured section includes the full operation. CPU operations
  finish synchronously in this example, so they do not need this CUDA wait.
- In this benchmark, copying the matrices happens before the timed GPU
  multiplication. The measured GPU time therefore excludes the copy. A real
  program may also spend time moving data between CPU and GPU memory.
- **Speedup** is CPU time divided by GPU time. The first Colab run measured
  1.040 seconds on CPU and 0.138 seconds on GPU, or about 7.6x. The second
  measured 0.960 seconds on CPU and 0.046 seconds on GPU, or about 21.0x. The
  first GPU run can include startup work, so repeated runs may differ. These
  results apply to this matrix operation on this Tesla T4; they are not a
  promise that every AI workload will be 21 times faster.

#### VRAM, parameters, and fp16

- **VRAM** is memory on the GPU. It holds model values and temporary values
  needed during calculations. Its capacity limits how much work fits on the
  GPU. The Colab T4 reported 15.6 GB in PyTorch.
- A **model parameter** is a learned number inside a model. The number of
  parameters is one rough way to describe a model's size.
- **Floating point** is a way to store numbers that can include fractions.
  **fp16** means each number uses 16 bits (two bytes). It uses less memory
  than fp32, which uses 32 bits (four bytes), though the two formats store
  numbers with different precision and range.
- The lesson's fp16 rule of thumb divides GPU memory by two bytes per
  parameter. The T4's reported 15.6 GB gives about 7.8 billion parameters; a
  24 GB GPU gives about 12 billion. This is only an ideal estimate for model
  weights. A real workload also needs memory for intermediate calculations,
  inputs, and sometimes gradients and optimizer data, so the model that fits
  is smaller. During training, these extra needs can be substantial.
- For commands, run Python statements in a notebook's Python code cell. A
  shell command such as `nvidia-smi` can run in a terminal or in a notebook
  cell prefixed with `!`. Pasting Python code directly at a zsh prompt makes
  the shell try to parse it as shell syntax, which caused the reported parse
  error; it was not a problem with the Colab GPU.

## Questions initially missed or corrected

### Lesson 1: Dev Environment

- **Question:** How do you check whether PyTorch can access an NVIDIA GPU?
  - **Initial answer:** `nvidia-smi --query`.
  - **Correct answer:** `import torch; print(torch.cuda.is_available())`.
  - **Simple explanation:** `nvidia-smi` checks the hardware and driver. `torch.cuda.is_available()` checks PyTorch's access to CUDA.

- **Question:** Which command checks whether an NVIDIA GPU is detected and
  shows its status?
  - **Initial answer:** `torch.cuda.is_available()`.
  - **Correct answer:** `nvidia-smi`.
  - **Simple explanation:** `nvidia-smi` reports the NVIDIA card and driver;
    `torch.cuda.is_available()` checks whether PyTorch can use CUDA. The
    distinction was reviewed after the quiz answer.

## Quiz results

- Lesson 1, Dev Environment: **2/2 pre, 2/3 post (4/5 total)**. Revisit the PyTorch GPU access check.
- Lesson 2, Git & Collaboration: **2/2 pre, 3/3 post (5/5 total)**.
- Lesson 3, GPU Setup & Cloud: **2/3 post**. Ran the benchmark on a Colab
  Tesla T4 and measured 7.6x speedup on the first run and 21.0x on the second.
