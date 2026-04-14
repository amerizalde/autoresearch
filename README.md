# AutoResearch

## What this project does
AutoResearch is a small experiment loop for model-improvement trials.

- `prepare.py` downloads TinyStories text data and builds a simple character-level tokenizer.
- `train.py` defines a lightweight GPT-style training loop and prints periodic validation loss.
- `agent.py` reads goals from `program.md`, asks an Ollama model to rewrite `train.py`, runs training, and keeps/reverts changes based on `val_loss`.

The intended goal (from `program.md`) is to lower validation loss while keeping runs within roughly 5 minutes.

## Project layout
- `agent.py`: orchestration loop (LLM call -> patch `train.py` -> run -> compare loss -> git keep/revert)
- `prepare.py`: data download + tokenizer creation + DataLoader preparation
- `train.py`: GPT-like model/training script used by the agent
- `train.py_sketch.py`: earlier sketch/pseudocode version of the training script
- `program.md`: user instructions for what the agent should optimize
- `plan.md`: setup/design plan for this environment
- `requirements.txt`: Python dependencies (`torch`, `requests`, `tqdm`, `numpy`)

## Requirements and setup
1. Use Python 3.10+ (PyTorch-compatible).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure Ollama is reachable from this machine, and update `OLLAMA_URL` / `MODEL_NAME` in `agent.py` if needed.
4. Run inside a git repository (`agent.py` expects `.git` and executes git add/commit/checkout).

## Quick start
From `autoresearch/`:

```bash
python prepare.py
python train.py
```

Optional autonomous loop:

```bash
python agent.py
```

## Workflow (prepare -> train)
1. `prepare.py`
- Downloads dataset text to `data/train.txt` (if missing).
- Builds tokenizer metadata in `data/tokenizer.bin`.
- Splits text into train/val and returns DataLoaders.

2. `train.py`
- Loads tokenizer metadata.
- Re-runs `prepare_data()` to get loaders.
- Trains/evaluates a GPT-style model and logs lines like `step X: val_loss ...`.

3. `agent.py` (optional)
- Reads `program.md` instructions.
- Sends full current `train.py` to Ollama and expects full updated code back.
- Runs `train.py`, parses `val_loss`, and keeps/reverts via git.
- Appends experiment metadata to `experiment_log.json`.

## Expected artifacts
- `data/train.txt`: downloaded training corpus.
- `data/tokenizer.bin`: serialized tokenizer maps and vocab size.
- `experiment_log.json`: run history from `agent.py` (created on first successful loop).
- Git commits from `agent.py` when a lower `val_loss` is found.

## Troubleshooting / known limitations
- `train.py` currently has a syntax error at model block construction (`for _ in in range(...)`), so it must be fixed before training can run.
- `prepare.py` contains placeholder/simplified tokenizer logic (character-level, not true BPE).
- `prepare.py` uses `os.path랑(...)` in `download_data()`; this appears to be a typo-like attribute and can fail at runtime.
- `agent.py` assumes:
  - a reachable Ollama endpoint (`OLLAMA_URL` defaults to `http://192.168.0.250:11434/api/generate`),
  - model name `llama3`,
  - and that the model returns the full valid `train.py` content.
- `agent.py` parses `val_loss` from stdout text format; output format changes can break result extraction.
