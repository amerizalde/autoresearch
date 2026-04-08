# Plan: AutoResearch Environment Setup

## Context
The goal is to create an automated AI research environment where a user provides high-level instructions in `program.md`, and an agent automatically modifies `train.py`, runs training, and evaluates the results to iteratively improve the model.

## Approach
I will propose a project structure and implementation details for the following components:
1.  **`train.py`**: A lightweight, minimal PyTorch-based training script (similar to nanoGPT) that outputs a final loss value.
2.  **`program.md`**: The user-facing instruction file.
3.  **`agent.py`**: The core orchestration script that:
    *   Reads `program.md`.
    *   Uses an LLM (via API) to propose changes to `train.py`.
    *   Applies changes to `train.py`.
    *   Executes `train.py` and captures the output.
    *   Compares the new loss against the best recorded loss.
    *   Keeps or reverts the changes based on the result.
    *   Logs each experiment (inputs, changes, results).
4.  **`requirements.txt`**: Necessary Python dependencies.

## Files to modify
- `train.py` (Initial version)
- `agent.py` (New)
- `program.md` (New, template)
- `requirements.txt` (New)
- `experiment_log.json` (New, for tracking)

## Reuse
- I will use standard PyTorch patterns for `train.py`.

## Steps
- [ ] Define the exact requirements for the LLM integration (OpenAI, Anthropic, or local).
- [ ] Draft the `train.py` script (minimalistic training loop).
- [ ] Draft the `agent.py` script logic.
- [ ] Create the project structure.

## Verification
- Manual verification of the `agent.py` loop:
  - Agent reads `program.md`.
  - Agent modifies `train.py` (e.g., changing a hyperparameter like learning rate).
  - Agent runs `train.py`.
  - Agent detects improvement or regression in loss.
  - Agent logs the result.
