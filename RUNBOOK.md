# Runbook — Assignment 2A (Group 172)

How to run this project end to end on a GPU machine (BITS Prayogshala lab, Google Colab, or any other CUDA box). Everything here has already been validated at small scale on a local CPU/MPS Mac; this runbook is for producing the real, full-fidelity submission.

## 0. What's in this folder

| File | Purpose |
|---|---|
| `Assignment2A_Group172.ipynb` | The deliverable notebook (Setup, Part A; Parts B/C still to come) |
| `train_adapter.py` | Reproduces 1A's two-stage pipeline (CPT, then QLoRA SFT) to regenerate the adapter |
| `instruction_dataset.jsonl` | 58 instruction/response pairs (50 from 1A + 8 supplementary, approved) |
| `domain_corpus/` | 13 cleaned source papers (same corpus as 1A) |
| `benchmark_10.json` | The fixed 10-query benchmark set used throughout Parts A/B/C |
| `requirements.txt` | Pins used for this local Mac (CPU/MPS only) — do **not** use this on a GPU machine |
| `requirements-gpu.txt` | Dependency set for a CUDA machine (labs, Colab) — use this instead |
| `results_partA.csv`, `*.csv` | Raw per-query results, produced by running the notebook |

## 1. Why you need to re-run training here

The notebook currently loads a **dev-scale placeholder adapter** (`./cpt_final_model`, `./final_qlora_adapter`) trained locally with a truncated corpus and 15 max steps, just to prove the pipeline works. It is not the real adapter — its generations are degenerate (repetitive loops) because it is severely undertrained. Before producing real submission numbers you must retrain at full scale on a GPU.

A real bug was found and fixed along the way: 1A's own `tokenize_sft` function computed the SFT loss over unmasked padding tokens, which destabilizes training. `train_adapter.py` already has the fix (labels are masked with `-100` at padding positions) — you don't need to do anything about this, just be aware the adapter you produce here will differ slightly from 1A's in the SFT loss curve shape, though the architecture, hyperparameters, and all other settings are an exact match to 1A's own executed run (cross-checked against 1A's own PDF report).

## 2. Set up the environment

### Option A — BITS Prayogshala lab (or any SSH'd CUDA Linux box)

```bash
git clone https://github.com/VijayYD/2025AE05316-LLM-2A.git
cd 2025AE05316-LLM-2A
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt
```

`requirements-gpu.txt` pins the whole ML stack to versions confirmed compatible with each other (`torch==2.5.1+cu121`, `transformers==4.46.3`, `peft==0.13.2`, `accelerate==0.34.2`, `bitsandbytes==0.43.3`) rather than just a floor version — leaving `transformers`/`peft` unpinned let pip grab a release requiring `torch>=2.6`, which doesn't ship a `cu121` build at all and breaks `peft`'s import chain (`NameError: name 'nn' is not defined`) when `transformers` disables its torch integration. `cu121` runs on any driver reporting CUDA 12.1 or newer — this matched the Prayogshala lab driver (CUDA 12.2) when this was last checked.

If `torch.cuda.is_available()` still comes back `False` after the install, check what CUDA version your driver actually supports:

```bash
nvidia-smi   # look at "CUDA Version:" in the top-right of the header
```

If it's older than 12.1, reinstall torch for a matching wheel (e.g. cu118 — a torch release that still ships a cu118 build, such as 2.5.1), and update the pin in `requirements-gpu.txt` to match:

```bash
pip uninstall -y torch
pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

(cu118, cu121, cu124, etc. — pick the index matching `nvidia-smi`'s reported CUDA version; check which torch releases still publish that build before picking a version.)

### Option B — Google Colab

1. In a Colab cell, clone straight from GitHub (simplest — no Drive mount needed):

```python
!git clone https://github.com/VijayYD/2025AE05316-LLM-2A.git
%cd 2025AE05316-LLM-2A
```

(Or, to persist outputs across Colab sessions, clone into Google Drive instead: mount Drive, then run the same `git clone` inside `/content/drive/MyDrive/`.)

2. Runtime → Change runtime type → GPU (T4 is enough).
3. Install dependencies:

```python
!pip install -q -r requirements-gpu.txt
```

4. Everything below (training, notebook execution) works the same way, just run each `!shell command` as its own Colab cell, and open `Assignment2A_Group172.ipynb` directly in Colab (or keep working in this same notebook's Jupyter interface via `jupyter notebook` if you prefer a local-style workflow through a Colab terminal).

## 3. Verify the environment (Step 0 of the notebook)

Run the notebook's Step 0 cell, or standalone:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
nvidia-smi
```

Confirm a CUDA GPU is visible before continuing — training and the Qwen2.5-7B-Instruct judge both need it to run in reasonable time.

## 4. Run the full-fidelity retrain

```bash
python train_adapter.py --mode full
```

This reproduces 1A exactly: 10 CPT epochs (900 steps on the full 399 packed sequences) then 8 SFT epochs on the full `instruction_dataset.jsonl`, matching 1A's own LoRA config (`r=16, lora_alpha=32, target_modules=["q_proj","v_proj"]`). On a T4-class GPU this should take well under an hour (1A's own SFT stage took 23 seconds on an RTX 2000 Ada for 80 steps; CPT took about 11 minutes for 900 steps).

It overwrites `./cpt_final_model` and `./final_qlora_adapter` in place — the notebook already points at these same paths, so no path changes are needed.

## 5. Point the judge at the real Qwen2.5-7B-Instruct

The notebook's A4 judge cell currently uses a tiny placeholder for local dev:

```python
JUDGE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # local dev placeholder; swap to "Qwen/Qwen2.5-7B-Instruct" on Prayogshala
```

Change this one line to:

```python
JUDGE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
```

If the GPU has less than ~16 GB free VRAM, load it in 4-bit instead — replace the judge model load cell's `AutoModelForCausalLM.from_pretrained(...)` call with:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
judge_model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL_ID, quantization_config=bnb_config, device_map="auto"
)
```

(`bitsandbytes` is already in `requirements-gpu.txt`.)

## 6. Run the notebook top to bottom

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 Assignment2A_Group172.ipynb
```

Or open it in Jupyter/Colab and use Kernel → Restart & Run All. Either way, every output must come from this real run — don't hand-edit any output cell.

`benchmark_10.json` is already fixed and reused as-is (the notebook only reads it, it isn't regenerated). This run regenerates `results_partA.csv` and the Part A results table with real numbers from the full-fidelity adapter and the real Qwen2.5-7B-Instruct judge.

## 7. Export to HTML and finish up

```bash
jupyter nbconvert --to html Assignment2A_Group172.ipynb
```

Then, per the assignment's ground rules:
- Scan the notebook for any leftover "marks" text (there shouldn't be any — already checked) and remove anything that slipped in.
- List every remaining `TODO (Vijay):` and `DRAFT -- review before submission:` cell so the human-judgement parts (spot-checks, verdicts) can be completed before submission.
- Check off the submission checklist (notebook + HTML, `benchmark_10.json`, `results_partA.csv`, and any other raw-results CSVs from Parts B/C once built).

## 8. Send results back

Once this has run on the labs machine, share: the executed `.ipynb`, the `.html` export, and the CSVs (`results_partA.csv` and any Part B/C equivalents). From there, remaining gaps (Parts B and C, which haven't been built yet) will continue against the real results instead of the dev-scale placeholder.
