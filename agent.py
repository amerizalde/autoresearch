import os
import subprocess
import json
import time
import requests
import sys

# --- CONFIGURATION ---
OLLAMA_URL = "http://192.168.0.250:11434/api/generate"
MODEL_NAME = "llama3" # or whatever model you use in Ollama
PROGRAM_FILE = "program.md"
TRAIN_SCRIPT = "train.py"
EXPERIMENT_LOG = "experiment_log.json"

def call_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return ""

def read_program():
    if not os.path.exists(PROGRAM_FILE):
        return "No instructions found in program.md."
    with open(PROGRAM_FILE, "r") as f:
        return f.read()

def apply_changes(changes_text):
    # This is a simplified way to apply changes. 
    # In a real scenario, the agent should produce a unified diff or a python script to apply changes.
    # For this demo, we'll assume the agent provides the FULL train.py content.
    with open(TRAIN_SCRIPT, "w") as f:
        f.write(changes_text)
    print("Applied changes to train.py")

def run_experiment():
    print("Running training script...")
    result = subprocess.run(["python", TRAIN_SCRIPT], capture_output=True, text=True)
    output = result.stdout
    error = result.stderr
    
    # Try to extract val_loss from output
    # Look for: "step XX: val_loss Y.YYYY"
    target_loss = None
    for line in output.split('\n'):
        if "val_loss" in line:
            try:
                parts = line.split("val_loss")
                target_loss = float(parts[1].strip().split(' ')[0])
                break
            except:
                continue
    
    return target_loss, output, error

def main_loop():
    best_loss = float('inf')
    
    # Initialize log
    if os.path.exists(EXPERIMENT_LOG):
        with open(EXPERIMENT_LOG, "r") as f:
            history = json.load(f)
    else:
        history = []

    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- Starting Experiment {iteration} ---")
        
        program_instructions = read_program()
        
        prompt = f"""
You are an AI Research Agent. Your goal is to improve the performance of the language model in `train.py`.
The current instruction is: {program_instructions}

I will provide you with the current content of `train.py`. 
You must output the ENTIRE updated content of `train.py`. 
Do not include any explanations, just the code.

Current `train.py`:
```python
{open(TRAIN_SCRIPT, 'r').read()}
```
"""
        
        print("Asking Ollama for improvements...")
        new_code = call_ollama(prompt)
        
        # Strip markdown code blocks if present
        if "```python" in new_code:
            new_code = new_code.split("```python")[1].split("```")[0].strip()
        elif "```" in new_code:
            new_code = new_code.split("```")[1].split("```")[0].strip()

        if not new_code:
            print("Agent failed to produce code. Retrying...")
            continue

        apply_changes(new_code)
        
        print("Running experiment...")
        loss, stdout, stderr = run_experiment()
        
        if loss is None:
            print("Failed to extract loss from training output.")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            # Revert changes? For simplicity, we just continue. In a real setup, we'd git revert.
            continue

        print(f"Experiment finished. Found val_loss: {loss}")

        is_improvement = loss < best_loss
        if is_improvement:
            print(f"SUCCESS! Improvement found: {best_loss:.4f} -> {loss:.4f}")
            best_loss = loss
            # Commit the improvement
            subprocess.run(["git", "add", TRAIN_SCRIPT])
            subprocess.run(["git", "commit", "-m", f"Experiment {iteration}: Improved loss to {loss:.4f}"])
        else:
            print(f"No improvement. {best_loss:.4f} -> {loss:.4f}")
            # Revert the changes
            subprocess.run(["git", "checkout", TRAIN_SCRIPT])

        # Log the experiment
        history.append({
            "iteration": iteration,
            "loss": loss,
            "timestamp": time.time(),
            "instructions": program_instructions
        })
        with open(EXPERIMENT_LOG, "w") as f:
            json.dump(history, f, indent=4)

        # Check for exit condition in program.md? 
        # For now, it just runs.

if __name__ == "__main__":
    # Ensure we are in a git repo
    if not os.path.exists(".git"):
        print("Error: This must be run inside a git repository to track experiments.")
        sys.exit(1)
        
    main_loop()
