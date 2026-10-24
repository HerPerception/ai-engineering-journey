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
| 0 | Setup & Tooling | Do | 16 |
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

## Review queue
- Phase 0, Lessons 1 and 3: Review how to check PyTorch CUDA access with `torch.cuda.is_available()`, distinct from checking NVIDIA hardware and its driver with `nvidia-smi`. On Apple Silicon, check `torch.backends.mps.is_available()`.
