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

### Lesson 3: GPU Setup & Cloud — in progress

- GPUs can accelerate neural network training by performing many matrix
  operations in parallel.
- VRAM is dedicated memory on the GPU, separate from system RAM. Although the
  name comes from video use, it also holds data for AI workloads and limits
  how much model data fits on the GPU.
- Google Colab is a convenient learning option because it runs notebooks on a
  hosted machine and avoids local GPU setup. A local NVIDIA GPU uses hardware
  already in your computer; a rented cloud GPU offers remote compute with
  more control and an hourly cost.
- In Colab, enable a GPU through **Runtime → Change runtime type → GPU**. The
  learner identified selecting a GPU runtime as the step that makes GPU
  compute available to the notebook.
- In the current AI journey workspace, `nvidia-smi` is unavailable and the
  project virtual environment does not have PyTorch installed. This means the
  workspace cannot verify local CUDA access; it does not establish whether a
  separate personal computer has a GPU.

## Questions initially missed or corrected

### Lesson 1: Dev Environment

- **Question:** How do you check whether PyTorch can access an NVIDIA GPU?
  - **Initial answer:** `nvidia-smi --query`.
  - **Correct answer:** `import torch; print(torch.cuda.is_available())`.
  - **Simple explanation:** `nvidia-smi` checks the hardware and driver. `torch.cuda.is_available()` checks PyTorch's access to CUDA.

## Quiz results

- Lesson 1, Dev Environment: **2/2 pre, 2/3 post (4/5 total)**. Revisit the PyTorch GPU access check.
- Lesson 2, Git & Collaboration: **2/2 pre, 3/3 post (5/5 total)**.
