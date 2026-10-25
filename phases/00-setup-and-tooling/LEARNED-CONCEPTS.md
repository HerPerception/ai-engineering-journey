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
