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

### Lesson 4: APIs & Keys

#### What an API call is

- An **API** (application programming interface) is a way for one program to
  ask another program for a service. In this lesson, a Python or TypeScript
  program sends a prompt to an AI provider's server, and the server returns a
  model response.
- The program sending the request is the **client**; the provider's computer
  handling it is the **server**. The **endpoint** is the web address for the
  operation, such as Anthropic's Messages endpoint.
- An **HTTP request** carries the action and data to the server. The lesson
  uses `POST` to send a message request. The server sends an **HTTP response**
  containing a status code and usually a JSON body.
- **Headers** carry request details, such as authentication and the format of
  the body. The **request body** contains the actual data: the model ID, an
  output token limit, and the message. **JSON** is a text format for structured
  data, such as objects, lists, strings, and numbers.
- **HTTPS** protects data while it travels between the program and the server.
  An API key is not encryption; it is a secret used to identify an account and
  authorize requests. The provider can also use it to attribute usage to that
  account.

#### API keys and configuration

- An **API key** is a secret credential. Treat it like a password: do not put
  it in source code, screenshots, shared notebooks, chat, or a public repo. If
  it is exposed, revoke it and create a replacement.
- An **environment variable** is a named value supplied to a program when it
  starts. Python reads them through `os.environ`; Node.js reads them through
  `process.env`. `ANTHROPIC_API_KEY` holds the secret. `LLM_MODEL` is a
  separate, non-secret setting that selects a model.
- In Python, `os.environ.get("LLM_MODEL", "claude-sonnet-5")` means: use the
  environment's `LLM_MODEL` value when it is set; otherwise use
  `claude-sonnet-5`. In TypeScript, `process.env.LLM_MODEL ?? "claude-sonnet-5"`
  does the same when the left side is missing or null. A model ID must be one
  supported by the provider; check the [current Anthropic model list](https://platform.claude.com/docs/en/models/overview)
  before making a live request because available IDs can change.
- A **`.env` file** is a local plain-text way to hold environment variables.
  Add `.env` to `.gitignore` so Git does not include it in commits or pushes.
  `.gitignore` does not encrypt the file or remove it from the computer.
- Python's standard library includes `os`, so `import os` needs no install.
  The `anthropic` package is an external Python SDK and must be installed in
  the same Python environment before `import anthropic` works, for example
  with `pip install anthropic`. Installing the package does not create or set
  an API key; those are separate steps. The Python sample reads environment
  variables but does not load a `.env` file on its own.
- An **SDK** (software development kit) is a provider's helper library. The
  Python SDK's `anthropic.Anthropic()` client reads `ANTHROPIC_API_KEY` from
  the environment by default and handles request formatting and sending. Its
  response is a Python object, for example `response.content[0].text`.
  The lesson's TypeScript SDK example similarly uses
  `@anthropic-ai/sdk`; that package must be installed before importing it.

#### SDK requests and raw HTTP

- The SDK and raw HTTP send the same basic information. The SDK packages it
  for you; raw HTTP makes those steps visible so you can debug them yourself.
- In the raw Python example, `json.dumps(...)` turns a Python dictionary into
  JSON text, and `.encode()` turns the text into bytes for the HTTP request.
  `urllib.request.Request(...)` builds the request; `urlopen(...)` sends it;
  `json.loads(resp.read())` parses the JSON reply back into Python data.
- The request body has a `model`, `max_tokens`, and `messages` field.
  `model` selects the model. `max_tokens` caps how many output tokens the
  model may produce. A `messages` list describes who said what; in this
  example it contains the user's prompt.
- The SDK returns an object, so the example reads fields as attributes, such
  as `response.content[0].text` and `response.usage.input_tokens`. Raw HTTP
  returns JSON, which the example reads with dictionary keys such as
  `result["content"][0]["text"]` and `result["usage"]["output_tokens"]`.
- The course's raw HTTP example uses `x-api-key` for Anthropic authentication.
  Current Anthropic docs show `Authorization: Bearer <key>` and say the older
  `x-api-key` header is still supported. The SDK chooses the authentication
  details automatically. See [Anthropic authentication](https://platform.claude.com/docs/en/manage-claude/authentication).
- The `anthropic-version` header selects the API's request/response contract;
  it is different from the `model` field, which selects the AI model. The SDK
  supplies the API-version header for you.
- The lesson's TypeScript implementation uses Node's built-in `fetch` and a
  small `.env` loader rather than the Anthropic SDK. Its `MOCK=1` mode returns
  a sample response locally and does not send a network request. Running that
  mock returned the sample answer with 12 input tokens and 28 output tokens.

#### Tokens, streaming, and errors

- An **API token** in language-model usage is a small text unit, not always a
  whole word. Input and output token counts are reported separately. Providers
  may use those counts for billing and rate limits; a real request can use
  account quota or incur charges.
- **Streaming** sends parts of the response as the model generates them,
  rather than making the program wait for the complete response. The pieces
  are not necessarily whole words.
- A **rate limit** caps how many requests or tokens can be used in a period.
  Exceeding it commonly returns HTTP `429` (Too Many Requests). Wait before
  retrying; repeated immediate retries make the problem worse. A server may
  include a `retry-after` header saying how long to wait.
- Common status-code meanings: `400` means the request is malformed or has
  invalid fields; `401` means the key is missing or invalid; `403` means the
  key lacks permission; `429` means a rate or usage limit was reached; and
  `5xx` means a server-side problem. Fix a bad request or credential before
  retrying; for temporary rate or server errors, wait and retry with a delay.
- The shipped `prompt-api-troubleshooter` artifact is a small guide for
  diagnosing these common API errors.

### Lesson 5: Jupyter Notebooks

#### Notebook, cells, and kernel

- A **Jupyter notebook** is an interactive document saved as an `.ipynb` file.
  The file stores code cells, Markdown cells, their outputs, and notebook
  metadata. JupyterLab, Jupyter Notebook, VS Code's Jupyter extension, and
  Colab can all work with this notebook format.
- A **code cell** runs Python. A **Markdown cell** displays formatted writing,
  headings, tables, images, and math. Keeping explanation next to code and its
  result makes a notebook useful as a lab notebook.
- **Shift+Enter** runs the selected cell and moves to the next cell. In the
  classic Jupyter keyboard model, Enter edits a selected cell and Escape
  returns to command mode, where shortcuts such as A and B insert cells.
- The **kernel** is the Python process running behind the notebook. The
  notebook sends code to it and receives results. All code cells connected to
  the same kernel share its imports, variables, and objects in memory.
- Cells can be run in any order. A cell that uses a variable may work only
  because another cell created it earlier in the current kernel, even if that
  earlier cell appears later in the notebook or has since been deleted. Old
  output can also remain visible after code changes, so the output may no
  longer match the cell's current code.
- **Restart & Run All** starts with a fresh kernel and runs cells from top to
  bottom. If that fails, the notebook depended on hidden state or a particular
  run order. Do this before sharing a notebook to check that it is reproducible.

#### Timing and magic commands

- A **magic command** is a Jupyter command for controlling the notebook. A
  single-percent command such as `%timeit` applies to one line; a double-
  percent command such as `%%time` applies to an entire cell. They are not
  ordinary Python syntax.
- `%timeit` runs a small operation repeatedly and reports timing statistics.
  It automatically chooses how many loops to run. Use it to compare short
  operations, not for a task with external side effects such as an API call or
  a training job that should happen only once.
- `%%time` measures one whole cell. Its output may show **user CPU time** for
  work done by the Python program, **system CPU time** for work done by the
  operating system on its behalf, their **total**, and **wall time**, which is
  elapsed real time from start to finish. The Colab run reported 130 ms user,
  40 microseconds system, 130 ms total, and 129 ms wall time; the small
  difference comes from measurement and rounding.
- `%matplotlib inline` asks Jupyter to show Matplotlib plots in the notebook.
  Colab and some other notebook interfaces already display plots inline.
- A leading `!`, as in `!pip install package`, runs a shell command from a
  notebook cell. `%env NAME` inspects an environment variable. Do not display
  a secret such as an API key while inspecting environment variables.
- The list-versus-NumPy exercise measured 5.12 ms (± 815 µs) per Python-list
  loop and 67.6 µs (± 1.62 µs) per NumPy loop. Since 1 ms is 1,000 µs, the
  list version took about 5,120 µs and this NumPy operation was roughly 76x
  faster in that run. The ± value is the reported variation (standard
  deviation) across repeated runs. `%timeit` chose 100 loops for the slower
  list operation and 10,000 for the faster NumPy operation to measure each
  more reliably. This result describes only this operation and machine.

#### Tables, plots, and notebook output

- A **data frame** is a table of data in memory. In pandas, its columns name
  fields and each row contains the values for one item or observation. For
  example, a sales data frame might have `day` and `sales` columns.
- A **plot** is a chart made from data. A line plot can make a trend over time
  easier to see than reading a table of numbers; a histogram shows how values
  are distributed.
- Notebooks automatically display the last expression in a code cell. Leaving
  `df` as the final line can show a formatted HTML table, while `print(df)`
  displays plain text. Matplotlib output can appear below its cell, so the
  chart is next to the code that created it. Images can also be displayed with
  IPython's `display` helper.

#### Choosing a notebook or script

- Notebooks are good for exploring data, trying model ideas, visualizing
  results, and explaining an experiment. A `.py` script is better for reusable
  utilities, training pipelines, packages, scheduled jobs, and production code.
- The workflow is: explore and prototype in a notebook, move stable reusable
  code into a script, then import that script back into the notebook if needed.
  The short rule is **explore in notebooks; ship in scripts**. A reusable
  training pipeline should live in a script.
- Variables remain in the kernel between cells, so large arrays and models
  also remain in memory. This can cause a notebook's memory use to grow.
  Delete unneeded variables with `del name`, use `gc.collect()` when
  appropriate, or restart the kernel to clear all its in-memory objects.

#### Colab and the shipped helper

- Colab is a hosted notebook service based on Jupyter. The `.ipynb` notebook
  can be saved in Google Drive or downloaded, but its runtime is a temporary
  virtual machine that can be deleted. Files and packages installed on that
  machine are not included just by sharing the notebook; save important files
  separately. Google's [Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html)
  explains current storage and runtime behavior; exact resource limits can
  change.
- The shipped `prompt-notebook-helper` artifact is a guide for diagnosing
  notebook problems such as hidden execution order, kernel crashes, memory
  growth, missing plots, and package-install issues.

### Lesson 6: Python Environments

- A **package** is reusable software installed for Python, such as NumPy or
  PyTorch. Different projects can require different versions of the same
  package. Installing or upgrading a package for one project can otherwise
  interfere with another project.
- A **virtual environment** is a project-specific place for its Python
  interpreter and installed packages. It keeps one project's packages
  separate from other projects and from the system Python. On Linux or
  macOS, `python3 -m venv .venv` creates one, and
  `source .venv/bin/activate` activates it in the current terminal session.
  After activation, `which python` should show a path containing `.venv`.
- `venv` comes with Python. `uv` is a separate tool that can manage Python
  versions, environments, and packages. Conda also manages environments and
  can handle some non-Python libraries. A virtual environment isolates
  packages; it does not itself make code faster or install a GPU driver.
- **`pyproject.toml`** describes a Python project and its package needs. A
  version such as `numpy>=1.26` allows NumPy 1.26 or newer. A package may
  itself need other packages; those are **transitive dependencies**.
- A **lockfile** records the exact selected versions of packages, including
  transitive dependencies, so the environment can be recreated more
  consistently. `pyproject.toml` describes project requirements; the lockfile
  captures the specific versions selected to meet them. Commit both to Git
  when the project uses them.
- Keep the environment directory, commonly `.venv/`, out of Git by listing it
  in `.gitignore`. It can be large and is tied to the local machine. Recreate
  it from the project's package description and lockfile instead of copying
  it to another computer.
- Activate the intended environment before installing or running packages.
  `which python` shows which interpreter will run. `python -m pip install
  package` ties pip to that interpreter, reducing the chance of installing
  into a different Python by mistake.
- Conda tracks packages it installs. Pip does not update Conda's package
  records the same way, so mixing them can cause Conda's view of the
  environment to become inaccurate and make conflicts harder to resolve. If
  using Conda, prefer it for the environment's packages; if a package is only
  available through pip, install Conda packages first and pip packages after.
- For GPU troubleshooting, `nvidia-smi` reports the NVIDIA GPU and driver,
  while `torch.version.cuda` reports the CUDA version associated with the
  installed PyTorch build, and `torch.cuda.is_available()` checks whether
  this PyTorch installation can currently use CUDA. A driver/build
  incompatibility can prevent GPU use. The CUDA version displayed by
  `nvidia-smi` describes driver capability; it does not necessarily mean the
  same CUDA Toolkit is installed separately.
- Google Colab supplies a managed runtime, so a learner generally does not
  need to create a local virtual environment just to run a Colab notebook.
  Project environments are useful when installing packages on a personal
  computer or keeping separate local projects from interfering with one
  another.

### Lesson 7: Docker for AI

- **Docker** packages an application's software setup so it can run more
  consistently on different computers. A container shares the host computer's
  operating system kernel, unlike a full virtual machine, which includes its
  own guest operating system. Containers are therefore usually lighter to
  start, though they are not a guarantee that every machine behaves
  identically.
- A **Dockerfile** is the written recipe: instructions such as `FROM`, `ENV`,
  and `RUN` describe how to assemble the software. A Docker **image** is the
  packaged template built from that recipe. A **container** is a running,
  isolated instance of an image. In the learner's analogy, the Dockerfile is
  the written building instructions, the image is the prepared template, and
  the container is a running copy built from it.
- `FROM` chooses a **base image**, the starting point for the build. For
  example, a basic Python base image is suitable for a program that does not
  need CUDA. An NVIDIA CUDA base image is useful when the container needs
  CUDA software. Development (`devel`) images include tools for compiling
  code; runtime images omit many build tools and are usually smaller. The
  exact example image and pinned package versions in a course Dockerfile are
  examples, not universal current recommendations.
- Dockerfile `RUN` instructions execute during image building. Each build step
  contributes a layer, and Docker can reuse cached results when the step and
  its inputs have not changed. This is why unchanged earlier steps can make
  later image rebuilds faster. `ENV` sets environment values inside the
  image, and `WORKDIR` selects the working directory used by later commands.
- `docker build` creates an image from a Dockerfile, for example
  `docker build -t ai-dev -f Dockerfile .`; `-t` assigns it a name and `.`
  supplies the build context (the files made available to the build).
  `docker run` starts a container from an image. `--rm` removes that
  container after it exits, and `-it` allows interactive terminal use. The
  image can remain after a container is removed.
- A container has its own filesystem. Files kept only there can disappear
  when the container is removed. A **volume mount** connects a folder on the
  host computer to a path inside the container, for example
  `-v "$PWD":/workspace` for the project folder and
  `-v "$HOME/models":/models` for model files. The host files persist across
  container removal and image rebuilds, avoiding repeated downloads of large
  datasets and model weights.
- A **port mapping**, such as `-p 8888:8888`, makes a port inside a container
  reachable through a port on the host computer. It can expose a Jupyter
  server from the container to a browser on the host.
- A container does not acquire GPU hardware just because its image contains
  CUDA software. The host computer must have a GPU and a working driver. On
  supported Linux systems, the **NVIDIA Container Toolkit** allows Docker to
  expose the host's NVIDIA GPU to the container, commonly with `--gpus all`.
  The CUDA libraries can be in the image; the GPU driver is provided by the
  host. If the host has no usable GPU, the container can still run CPU work.
  This Docker setup is distinct from Colab, whose hosted runtime provides a
  GPU for notebook work.
- **Docker Compose** describes and manages multiple related containers in a
  YAML file. For example, an AI development service and a Qdrant vector
  database can be started together. Compose creates a shared network, so the
  AI service can reach Qdrant by its service name, such as
  `http://qdrant:6333`, without manually finding its changing IP address.
  Compose can also define each service's ports, volumes, and GPU reservation.
- AI projects commonly use different containers for development, training,
  and inference. A development container may include Jupyter and debugging
  tools; a training container can focus on the training script and
  dependencies; an inference container can focus on serving a model. This
  keeps each container's contents suited to its job.
- Docker containers do not replace version control or dependency files. Git
  stores source changes, while a Dockerfile describes how to build an image.
  Large models and datasets are usually stored outside the image and connected
  through mounts, so rebuilding the image does not require copying those
  files into every image layer.

### Lesson 8: Editor Setup

- A **code editor** is where you read and change source code. Extensions add
  features for particular tasks, such as Python support, notebooks, code
  formatting, error checking, Git history, or connecting to another computer.
  VS Code is one example; Cursor and Windsurf use many of the same settings
  and extensions. The lesson also describes a way to add Python features to
  Neovim.
- A **language server** is a helper program that analyzes code and answers
  questions from an editor, such as “what does this name refer to?” It can
  provide completions, type information, and warnings about likely mistakes.
  The **Language Server Protocol (LSP)** is the shared set of rules that lets
  different editors and language servers communicate. This lets an editor
  offer language features without implementing all Python analysis itself.
- **Pylance** is a Python language server extension for VS Code. It can show
  completions and inferred types, and its type checking can flag some
  incorrect argument types or other likely problems while editing. These
  warnings help find issues early but do not prove that code is free of bugs.
- An `extensions.json` file can list recommended VS Code extensions for a
  project, such as Python, Pylance, Jupyter, a formatter, Ruff, GitLens, and
  Remote SSH. VS Code can offer to install recommendations when the project
  opens. These are suggestions to adapt, not a requirement to install every
  extension.
- VS Code settings are stored as JSON key/value pairs. For example,
  `"python.analysis.typeCheckingMode": "basic"` enables basic Python type
  checking; `"editor.formatOnSave": true` runs a configured formatter when
  saving; and `"editor.rulers": [88, 120]` shows visual line-length guides.
  Rulers are guides, not enforced limits.
- A **formatter** rewrites code into a consistent layout. With
  `editor.formatOnSave` enabled and a Python formatter such as Black selected,
  saving a file applies that formatting automatically. Ruff can also check
  code for common problems and can be configured to run on save.
- Notebook setting `notebook.output.scrolling: true` keeps long cell output,
  such as thousands of training progress lines, in a scrollable area instead
  of letting it expand the notebook page indefinitely. Auto-save can help
  avoid running a stale file after forgetting to save edits.
- VS Code's **integrated terminal** is a command-line shell inside the editor.
  It can run Python scripts, install packages, or monitor a GPU, keeping the
  editor and development commands in one window. The integrated terminal is
  still a shell, so Python statements belong in a Python file or notebook,
  not directly at a shell prompt.
- **Remote SSH** connects VS Code to a remote computer. The editor window is
  local, while project files and commands can be on the remote machine. It is
  useful when a GPU server is elsewhere. It differs from Google Colab: Colab
  provides a hosted notebook runtime, while Remote SSH connects an editor to
  a computer over SSH. The learner currently uses Colab and does not need to
  set up Remote SSH for those notebook lessons.
- A productive editor workflow can combine code completions and warnings,
  automatic formatting on save, notebooks for experiments, an integrated
  terminal for commands, and Git tools for reviewing changes. Editor tools
  support the work but do not replace understanding or running the code.

### Lesson 9: Data Management

- A **dataset** is a collection of examples used by a program. For movie
  reviews, one example (often one row) might contain the review text and a
  label such as positive or negative. A **column** is one kind of information
  recorded for each example, such as text or label. To **inspect** a dataset
  is to look at basic details such as its column names and first example.
- The **Hugging Face Hub** is a place where people share datasets and models.
  The Python `datasets` library can retrieve datasets from it. For example,
  `from datasets import load_dataset` imports the library's `load_dataset`
  function; it is not part of Python's built-in functions. Calling it with a
  dataset name loads the requested data. `reviews[0]` selects the first
  example from the `reviews` dataset.
- A **cache** is a local saved copy of data already downloaded. The Hugging
  Face library can use the cache on a later load rather than downloading the
  same data again. Its default dataset cache is under
  `~/.cache/huggingface/datasets/`. Colab's local runtime storage is temporary,
  so the cache may disappear when the runtime is reset.
- **Streaming** asks the dataset library for examples as they are needed
  instead of downloading the complete dataset first. This is useful when a
  dataset is very large. The stream can be read one example at a time; code
  that saves every received example into its own list can still use growing
  memory, so streaming alone does not prevent that.
- Data **formats** are different ways to save the same information. **CSV**
  is readable table text and is convenient for people and spreadsheets.
  **JSON** is text that can represent nested information and is common for
  exchanging data between programs. **Parquet** is a compact binary format
  that stores data by column and is useful for large datasets and efficient
  column reads. **Apache Arrow** is a column-based format for working with
  data in memory; Hugging Face `datasets` uses it internally. Methods such as
  `to_csv`, `to_json`, and `to_parquet` save a dataset in those formats.
- Machine-learning data is commonly divided into three **splits**, each with
  a different job. **Training data** is used to teach the model. **Validation
  data** is checked during development to help decide how to improve or
  configure the model. **Test data** is held back until the final evaluation
  so the result estimates performance on examples not used during
  development. The names describe the intended use of each group.
- A random **seed** is a number used to make a random shuffle repeatable. If
  the same dataset and seed are used again, the data split can be reproduced,
  making experiment comparisons fairer. In the sample split helper, a seed of
  `42` is used for that purpose.
- The helper's `make_splits` function defaults to 80% training, 10%
  validation, and 10% test data. A detail in the lesson's separate IMDB code
  example: it loads only the original `train` split, sets aside 20% as test,
  then takes validation from the remaining 80%. That produces 70% train,
  10% validation, and 20% test of the loaded data, not 80/10/10. If a dataset
  already provides an official test split, keep it for the final test and
  create validation data from the training portion.
- Large datasets and model weights usually should not be committed directly
  into ordinary Git history. **`.gitignore`** tells Git to leave specified
  files out; it is simplest for personal or re-downloadable data, but does
  not store or version the data. **Git LFS** tracks large files through small
  pointers in Git while storing file contents separately. **DVC** puts small
  data-version pointer files in Git and stores the actual data in a remote
  location such as S3 or GCS, helping teams reproduce experiments with the
  same data version.
- The lesson's `data_utils.py` helper groups useful steps into functions:
  load and inspect a dataset, stream a few examples, convert formats and
  compare file sizes, make reproducible splits, download a model file,
  summarize the local dataset cache, and reload saved CSV/JSON/Parquet data.
  Its `fingerprint` function computes a SHA-256 digest from a sample of rows;
  this gives a compact way to notice changes in that sample, but it is not a
  guarantee that the entire dataset is identical.
- Main workflow: find data, load or stream examples, inspect them, make
  repeatable train/validation/test groups, save or convert them when useful,
  and keep large files managed outside ordinary Git history.

### Lesson 10: Terminal and Shell

- A **terminal** is the window for entering text commands. A **shell** is the
  program running inside that window that reads commands and asks the
  computer to perform them. Bash and zsh are common shells. The command-line
  examples here are for a Unix-like shell; a Windows shell may use different
  commands.
- `pwd` prints the current folder. `ls -la` lists its files, including hidden
  files. `cd path` changes the current folder, `~` means the user's home
  folder, and `..` means the parent (one level up) folder. Ctrl+R searches
  command history, Ctrl+C interrupts a running command, and Ctrl+L clears
  the terminal display.
- `grep` searches text for matching lines. In `grep "ERROR" train.log`, it
  prints lines containing `ERROR`. `tail -3` shows the last three lines of
  input. A **pipe** (`|`) sends one command's normal output into the next
  command as its input, so `grep "loss" train.log | tail -3` finds loss lines
  and then displays only the last three.
- Programs commonly write normal output to **stdout** and error messages to
  **stderr**. `>` sends stdout to a file and replaces that file's old content;
  `>>` sends stdout to the end of the file without replacing old content.
  `2>` sends stderr to a file. In
  `python train.py > train_full.log 2>&1`, stdout goes to `train_full.log`
  and `2>&1` sends stderr to the same destination. This records normal output
  and errors together.
- A long-running command can be placed in the background with `&`, but it
  usually stops if its terminal closes. `nohup command > run.log 2>&1 &`
  helps a process survive terminal closure and records its output, but gives
  no way to return to its live terminal session. **tmux** creates a terminal
  session that can be detached from and reattached to later, with live output
  and multiple panes. `tmux new -s training` creates a named session;
  Ctrl+B then D detaches; `tmux attach -t training` reattaches. tmux does not
  protect a process from the computer or hosted runtime shutting down. It is
  useful on a remote server that stays running; it cannot prevent Colab from
  ending its runtime.
- A **process** is a running program. A **PID** (process ID) is its unique
  number, used by system tools to inspect or signal that process. `jobs`
  lists jobs started from the current shell; `ps` lists processes. `kill` or
  other process-ending commands can stop work, so confirm which process will
  be affected before using them.
- `htop` displays computer processes and CPU/memory use. `nvidia-smi` reports
  NVIDIA GPU and driver status; `watch -n1 nvidia-smi` repeats that command
  once per second. `nvtop` shows GPU activity interactively. These show
  system/GPU status; checking whether PyTorch can use CUDA still requires
  `torch.cuda.is_available()` in the relevant Python environment.
- **SSH** (Secure Shell) securely connects to another computer so commands
  can run there. `ssh user@gpu-box` opens a remote shell. `scp` copies files,
  while `rsync` synchronizes files or directories and can avoid copying
  unchanged files. For example, `scp model.pt user@gpu-box:~/models/` copies
  one model file to a remote folder.
- SSH **port forwarding** securely connects a local port to a service port on
  the remote computer. `ssh -L 8888:localhost:8888 user@gpu-box` makes a
  remote service listening on port 8888 available through `localhost:8888`
  on the local machine, such as a remote Jupyter server. `localhost` refers
  to the computer at the end of that connection (here, the remote server).
- A shell **alias** is a short nickname for a command, such as `gpu` for
  `nvidia-smi`. Aliases can also run commands that change or stop things; for
  example, the lesson's `killtraining` alias stops matching Python training
  processes, so understand a shortcut before using it. Shell setup files
  such as `.bashrc` and `.zshrc` can load aliases when a shell starts.
- In a Colab notebook code cell, prefix a shell command with `!`, for example
  `!nvidia-smi`. This runs a shell command in the hosted runtime. The lesson's
  tmux and SSH patterns mainly apply when using a persistent computer or
  remote GPU server; a Colab runtime can still be shut down independently.
- A common AI terminal workflow is to run a training program, save its output
  to a log, filter that log with `grep`, monitor GPU activity with
  `nvidia-smi`, and use SSH/rsync to work with a remote machine and its files.

### Lesson 11: Linux for AI

- Linux organizes its files beneath one starting point called the **root
  directory**, written `/`. A path begins at that starting point, such as
  `/home/sam/project`. The **home directory** contains a user's files and is
  usually `/home/username`; `~` is its shortcut. Other common locations:
  `/tmp` holds temporary files and may be cleared on reboot; `/usr` contains
  installed programs and libraries; `/etc` holds system configuration;
  `/var/log` contains system logs; `/mnt` and `/media` are common mount
  points; `/proc` and `/sys` expose system and hardware information.
- Linux file commands: `pwd` shows the current directory; `ls -la` lists
  files, including hidden files; `cd` changes directory; `mkdir -p a/b/c`
  creates nested directories; `touch name` creates an empty file; `cp` copies;
  `mv` moves or renames; and `rm` deletes. `rm -rf folder/` permanently
  deletes that folder and its contents without a trash/recycle bin, so verify
  the path before running it.
- To read files, `cat` prints all contents, `head -20` prints the first 20
  lines, `tail -20` prints the last 20, `less` opens a scrollable view (press
  `q` to quit), and `tail -f log.txt` follows new lines as a log grows (stop
  with Ctrl+C). `grep` searches for matching text; `find . -name "*.py"`
  searches under the current directory for Python files.
- Linux files have an **owner** and permission bits for the owner, their
  group, and other users. `r` means read, `w` means write, and `x` means
  execute/run. `chmod +x train.sh` adds execute permission so it can be run
  as `./train.sh`. `chown user:group file` changes ownership and usually
  requires administrator permission. A “Permission denied” message can come
  from permissions, but check the file/path and the specific operation before
  changing permissions or using administrator access.
- Ubuntu uses **apt** to install system-level packages, such as `htop`,
  `tmux`, or build tools. Python packages such as NumPy are usually installed
  with pip/uv inside a Python environment. `sudo` temporarily runs one
  command with administrator (root) privileges. For example,
  `sudo apt update` refreshes Ubuntu's package list, then
  `sudo apt install -y htop` installs htop. Use `sudo` only for operations
  that need it; ordinary training scripts should generally run as your normal
  user.
- `whoami` prints the current user. A **process** is a running program;
  `ps aux` lists processes, `htop` gives an interactive process view, and
  `kill PID` asks the process with that ID to stop. `kill -9 PID` forces it to
  stop and should be a last resort. `nvidia-smi` reports NVIDIA GPU status.
  **systemd** manages background services, such as a server that should keep
  running. `systemctl status service` checks one; `start`, `stop`, and
  `restart` control it, and `enable` configures it to start at boot.
- Disk commands answer different questions: `df -h` shows free/used space on
  mounted disks, while `du -sh folder` shows the size of one folder.
  `du -h --max-depth=1 / | sort -hr | head -20` helps find large directories
  near the top of the filesystem. Models, datasets, and caches can fill a GPU
  server's disk. `pip cache purge` clears pip's download cache;
  `sudo apt clean` clears downloaded apt package files. Delete checkpoints
  only when you are sure they are no longer needed.
- `wget` and `curl` can download files or make web requests from a terminal.
  `scp` copies files between machines; `rsync -avz --progress` synchronizes
  files and directories, often sending only changes and resuming more
  conveniently after a connection interruption.
- **tmux** can keep a terminal session available after disconnecting from an
  SSH session. Start with `tmux new -s train`, detach with Ctrl+B then D,
  list sessions with `tmux ls`, and return with `tmux attach -t train`.
  Inside tmux, Ctrl+B then `%` or `"` splits panes; Ctrl+B then an arrow
  changes panes. The remote computer must remain on; tmux cannot keep work
  running if the machine is shut down.
- Linux and macOS use many similar commands but differ in some details.
  Ubuntu commonly uses `apt` while macOS commonly uses Homebrew (`brew`).
  macOS often uses zsh and `~/.zshrc`; Linux servers commonly use Bash and
  `~/.bashrc`. macOS's BSD `sed -i` expects an empty string after `-i`, as in
  `sed -i '' 's/a/b/' file`; GNU/Linux `sed -i` does not. Linux filenames
  are usually case-sensitive (`Model.py` and `model.py` are distinct). A
  remote server may have no graphical desktop, so commands such as `open`
  may not work there; read files with `less` or `cat` instead.
- **WSL2** gives Windows users a Linux environment without replacing Windows.
  From WSL, Windows files are generally available beneath `/mnt/c/Users/...`.
  The lesson notes that NVIDIA GPU access in WSL uses the Windows-side
  NVIDIA driver. Windows text files can use CRLF line endings, while Linux
  scripts expect LF; `dos2unix` can convert a script if those line endings
  cause problems.
- In Colab, notebook code runs on a hosted Linux runtime. Prefix a shell
  command with `!`, such as `!pwd` or `!df -h`, to run it from a notebook
  cell. Its filesystem and installed system packages belong to that runtime.

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
- Lesson 4, APIs & Keys: **3/3 post**. Correctly identified `.env`/`.gitignore`
  key storage, the lesson's `x-api-key` header, and waiting after a `429` rate limit.
- Lesson 5, Jupyter Notebooks: **3/3 post**. Ran two cells that shared a
  variable, compared Python-list and NumPy timings, used `%%time`, and chose a
  plot to see a data trend.
- Lesson 6, Python Environments: **2/2 pre, 3/3 post (5/5 total)**. Correctly
  explained package isolation, lockfiles, checking the active Python path,
  Conda/pip conflicts, and why `.venv/` belongs in `.gitignore`.
- Lesson 7, Docker for AI: **2/2 pre, 3/3 post (5/5 total)**. Correctly
  distinguished Docker images from containers and explained persistent
  volumes, host-provided GPU access, and Compose networking by service name.
- Lesson 8, Editor Setup: **2/2 pre, 3/3 post (5/5 total)**. Explained
  language-server help, notebook output scrolling, and the difference between
  Colab's hosted runtime and Remote SSH to a remote computer.
- Lesson 9, Data Management: **2/2 pre, 3/3 post (5/5 total)**. Worked through
  the new topic with a small movie-review example; correctly identified
  streaming, Parquet, CSV, and `.gitignore` uses. Needed a plain example of
  “inspect,” now recorded above.
- Lesson 10, Terminal and Shell: **2/2 pre, 3/3 post (5/5 total)**. Correctly
  described the terminal/shell distinction, pipe and redirection behavior,
  tmux detach/reattach, SSH file copying, and Colab's `!` shell prefix.
- Lesson 11, Linux for AI: **2/2 pre, 3/3 post (5/5 total)**. Correctly
  identified the home directory, execute permission, disk-usage command,
  and macOS/Linux `sed -i` difference.
