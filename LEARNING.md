# My AI Engineering Path
<!-- Managed by the ai-engineering-from-scratch learning skills.
     Repo: https://github.com/rohitg00/ai-engineering-from-scratch -->
<!-- After each lesson, update that phase's LEARNED-CONCEPTS.md with plain-English concepts and missed questions with corrections. -->

## Mission
To understand AI tools I use, become an AI engineer, research. By the end, I want to solve real-world problems with AI engineering; I am not sure yet what specific thing I want to build.

## Placement
- Date: 2026-10-23
- Score: self-selected
- Entry point: Phase 0: Setup & Tooling
- Pace: ~5 hours/week

## Path
| Phase | Name | Status | Est. hours |
|-------|------|--------|------------|
| 0 | Setup & Tooling | Done | 16 |
| 1 | Math Foundations | Do | 23 |
| 2 | ML Fundamentals | Do | 26 |
| 3 | Deep Learning Core | Do | 15 |
| 4 | Computer Vision | Do | 27 |
| 5 | NLP — Foundations to Advanced | Do | 30 |
| 6 | Speech & Audio | Do | 18 |
| 7 | Transformers Deep Dive | Do | 14 |
| 8 | Generative AI | Do | 14 |
| 9 | Reinforcement Learning | Do | 13 |
| 10 | LLMs from Scratch | Do | 26 |
| 11 | LLM Engineering | Do | 19 |
| 12 | Multimodal AI | Do | 65 |
| 13 | Tools & Protocols | Do | 76 |
| 14 | Agent Engineering | Do | 55 |
| 15 | Autonomous Systems | Do | 20 |
| 16 | Multi-Agent & Swarms | Do | 28 |
| 17 | Infrastructure & Production | Do | 34 |
| 18 | Ethics, Safety & Alignment | Do | 31 |
| 19 | Capstone Projects | Do | 620 |

## Progress log
| Date | Lesson | Quiz | Note |
|------|--------|------|------|
| 2026-10-23 | 00/01 | 2/2 pre, 2/3 post (4/5 total) | Identified Python as the language runtime and correctly traced a dot product; missed how to verify PyTorch GPU access. |
| 2026-10-23 | 00/02 | 2/2 pre, 3/3 post (5/5 total) | Understood that Git commits preserve inspectable history and branches isolate experiments until merge. |
| 2026-10-24 | 00/03 | 2/3 post | Ran a CPU/GPU matrix benchmark in Colab on a Tesla T4 (7.6x first run, 21.0x second); review `nvidia-smi` versus `torch.cuda.is_available()`. |
| 2026-10-25 | 00/04 | 3/3 post | Understood that API keys identify and authorize requests, `.env` stays out of Git via `.gitignore`, and `429` means wait before retrying. |
| 2026-10-25 | 00/05 | 3/3 post | Used shared notebook state, compared Python and NumPy timings, and learned to use Restart & Run All to catch hidden cell-order dependencies. |
| 2026-10-25 | 00/06 | 2/2 pre, 3/3 post (5/5 total) | Explained how virtual environments isolate project packages, why lockfiles help recreate exact versions, and why `.venv/` should be ignored by Git. |
| 2026-10-25 | 00/07 | 2/2 pre, 3/3 post (5/5 total) | Distinguished Dockerfiles, images, and containers; explained persistent volume mounts, GPU access requirements, and Compose service networking. |
| 2026-10-26 | 00/08 | 2/2 pre, 3/3 post (5/5 total) | Explained how language servers help an editor understand code, why notebook output scrolling helps, and how Remote SSH differs from Colab. |
| 2026-10-26 | 00/09 | 2/2 pre, 3/3 post (5/5 total) | Understood datasets as examples, the roles of train/validation/test groups, and how caching, streaming, formats, and data-file tracking help manage them. |
| 2026-10-26 | 00/10 | 2/2 pre, 3/3 post (5/5 total) | Explained shell versus terminal, filtering and saving logs, keeping remote jobs accessible with tmux, and using SSH to connect, copy files, and forward ports. |
| 2026-10-26 | 00/11 | 2/2 pre, 3/3 post (5/5 total) | Explained Linux paths and file commands, the execute permission, `sudo` and `apt`, the difference between `df` and `du`, and macOS/Linux command differences. |
| 2026-10-26 | 00/12 | 2/2 pre, 3/3 post (5/5 total) | Explained how AI bugs can run silently, how to inspect tensors, profile slow steps, spot data leakage and overfitting, and investigate NaN loss. |
| 2026-10-27 | 00/13 | 2/2 pre, 3/3 post (5/5 total) | Practiced core Python patterns for data and AI: safe list handling, dataclasses, generators, repeatable splits, batches, and NumPy vectorization. |

## Review queue
- Phase 0, Lessons 1 and 3: Review how to check PyTorch CUDA access with `torch.cuda.is_available()`, distinct from checking NVIDIA hardware and its driver with `nvidia-smi`. On Apple Silicon, check `torch.backends.mps.is_available()`.
