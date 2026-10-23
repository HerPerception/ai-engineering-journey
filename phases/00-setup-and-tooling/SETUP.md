# HOW TO SET UP THE ENVIRONMENT AND THE TUTOR (CODEX)
- mkdir ai; cd ai

- git clone https://github.com/HerPerception/ai-engineering-journey.git

- git clone https://github.com/rohitg00/ai-engineering-from-scratch.git

- curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.7/install.sh | bash

- source ~/.zshrc

- nvm --version
- nvm install 24

- nvm alias default 24

- node --version

- npm --version

- npx --version

- cd ~/ai/ai-engineering-journey

- npx skills add rohitg00/ai-engineering-from-scratch

- npm install -g @openai/codex

- codex --version

- codex


# TO CREATE A VIRTUAL ENVIRONMENT USING UV
- curl -LsSf https://astral.sh/uv/install.sh | sh

- source $HOME/.local/bin/env

- uv python install 3.12

- uv venv
- source .venv/bin/activate  # or .venv\Scripts\activate on Windows

- uv pip install numpy matplotlib jupyter


# HOW TO SET UP RUST ON THE SCHOOL SYSTEM.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --no-modify-path

echo 'source "$HOME/.cargo/env"' >> ~/.zshrc

source ~/.zshrc

rustc --version
cargo --version

## TO SET UP A PROJECT FOLDER SO YOU CAN USE cargo run
cargo new my-ai-experiment
cd my-ai-experiment
cargo run


# Node.js with pnpm - For TypeScript lessons (agents, MCP servers, web apps)
curl -fsSL https://fnm.vercel.app/install | bash
fnm install 22
fnm use 22

npm install -g pnpm

node -e "console.log('Node', process.version)"


# Julia (Optional) - For math-heavy lessons where Julia shines
curl -fsSL https://install.julialang.org | sh

julia -e 'println("Julia ", VERSION)'