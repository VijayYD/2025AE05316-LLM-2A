"""
Reproduces Assignment 1A's two-stage pipeline (CPT then QLoRA SFT) on
HuggingFaceTB/SmolLM2-360M, matching the real 1A submission's hyperparameters,
packing strategy, and chat-template format exactly (cross-checked cell by cell
against Group172_Assignment1A_submission/LLM_Assignment_1A_Group_172.ipynb).
The original 1A adapter weights were lost (ephemeral Colab storage), so this
script regenerates them from the same domain_corpus/ and instruction_dataset.jsonl.

SYSTEM_PROMPT and the "### Question:" / "### Answer:" chat format defined below
are also used, verbatim, by the 2A notebook's generator wrapper at inference
time -- the adapter is only ever trained and prompted in this one format.

Usage:
  python train_adapter.py --mode dev    # fast, truncated run to validate the pipeline (this Mac, CPU)
  python train_adapter.py --mode full   # exact 1A hyperparameters (4 CPT epochs, 8 SFT epochs) -- run on Prayogshala
"""
import argparse
import json
import math
import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset

MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
SEED = 42

# ---------------------------------------------------------------------------
# Chat format -- identical to the real 1A submission's SFTTrainer chat template.
# Shared verbatim with the 2A notebook's generate()/call_generator() wrapper.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a research assistant that answers questions about deep learning and "
    "large language model papers accurately and concisely."
)


def chat_prompt(instruction: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n### Question:\n{instruction}\n\n### Answer:\n"


# ---------------------------------------------------------------------------
# CPT config (matches the real 1A notebook's CONFIGURATION cell)
# ---------------------------------------------------------------------------
SEQ_LENGTH = 2048          # packed chunk length (1A: SEQ_LENGTH)
EVAL_FRACTION = 0.10       # per-document held-out span (1A: EVAL_FRACTION)
CPT_EFFECTIVE_BS = 8       # 1A: CPT_EFFECTIVE_BS
CPT_LR = 5e-5              # 1A: CPT_LR

# ---------------------------------------------------------------------------
# SFT config (matches the real 1A notebook's CONFIGURATION cell)
# ---------------------------------------------------------------------------
MIN_RESPONSE_WORDS = 30    # 1A: MIN_RESPONSE_WORDS (unused here -- dataset is pre-filtered)
SFT_MAX_LEN = 512          # 1A: SFT_MAX_LEN
SFT_LR = 2e-4              # 1A: SFT_LR
SFT_EFFECTIVE_BS = 4       # 1A: SFT_EFFECTIVE_BS
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGETS = ["q_proj", "v_proj"]

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["dev", "full"], required=True)
args = parser.parse_args()
DEV = args.mode == "dev"

CPT_EPOCHS = 1 if DEV else 4   # 1A: CPT_EPOCHS = 4
SFT_EPOCHS = 1 if DEV else 8   # 1A: SFT_EPOCHS = 8

torch.manual_seed(SEED)
random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
cc_major = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
USE_BF16 = torch.cuda.is_available() and cc_major >= 8   # Ampere/Ada/Hopper; matches 1A's own check
dtype = torch.bfloat16 if USE_BF16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
print(f"Mode: {args.mode} | Device: {device} | Dtype: {dtype}")

# ---------------------------------------------------------------------------
# Stage 0: load cleaned domain corpus (already cleaned in 1A; reused as-is)
# ---------------------------------------------------------------------------
corpus_dir = "domain_corpus"
cleaned_documents = {}
for fname in sorted(os.listdir(corpus_dir)):
    if fname.endswith(".txt"):
        with open(os.path.join(corpus_dir, fname), encoding="utf-8") as f:
            cleaned_documents[fname[:-4]] = f.read()
print(f"Loaded {len(cleaned_documents)} cleaned source documents")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
BOS_ID = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
EOS_ID = tokenizer.eos_token_id

# ---------------------------------------------------------------------------
# Per-document held-out span, then pack train/eval streams separately
# (matches 1A's Step 2 cell exactly, including its surplus-dropping logic).
# ---------------------------------------------------------------------------
rng = random.Random(SEED)
train_stream, eval_stream = [], []
for doc, text in cleaned_documents.items():
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    n = len(ids)
    span = max(1, int(round(EVAL_FRACTION * n)))
    start = rng.randint(0, n - span)
    eval_segment = ids[start:start + span]
    train_segments = [seg for seg in (ids[:start], ids[start + span:]) if seg]
    eval_stream += [BOS_ID] + eval_segment + [EOS_ID]
    for seg in train_segments:
        train_stream += [BOS_ID] + seg + [EOS_ID]


def pack(stream, length):
    n_chunks = len(stream) // length
    return [stream[i * length:(i + 1) * length] for i in range(n_chunks)]


train_chunks = pack(train_stream, SEQ_LENGTH)
n_surplus = len(train_chunks) % CPT_EFFECTIVE_BS
if n_surplus:
    surplus_idx = set(random.Random(SEED).sample(range(len(train_chunks)), n_surplus))
    train_chunks = [c for i, c in enumerate(train_chunks) if i not in surplus_idx]
eval_chunks = pack(eval_stream, SEQ_LENGTH)

if DEV:
    train_chunks = train_chunks[:8]
    eval_chunks = eval_chunks[:2]
    print(f"[dev mode] truncated to {len(train_chunks)} train / {len(eval_chunks)} eval packed sequences")

pd.DataFrame({"input_ids": train_chunks}).to_parquet("cpt_train_packed.parquet")
pd.DataFrame({"input_ids": eval_chunks}).to_parquet("cpt_eval_packed.parquet")
print(f"CPT train sequences: {len(train_chunks)} | eval sequences: {len(eval_chunks)} (seq_len={SEQ_LENGTH})")


class PackedCPTDataset(Dataset):
    def __init__(self, parquet_file):
        self.rows = pd.read_parquet(parquet_file)["input_ids"].tolist()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ids = torch.tensor(self.rows[idx], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


train_dataset = PackedCPTDataset("cpt_train_packed.parquet")
eval_dataset = PackedCPTDataset("cpt_eval_packed.parquet")

# ---------------------------------------------------------------------------
# Stage 1: Continual Pre-Training (CPT)
# ---------------------------------------------------------------------------
print("\n--- Loading base model for CPT ---")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
model.to(device)


class LossHistoryCallback(TrainerCallback):
    def __init__(self):
        self.losses = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])
            self.steps.append(state.global_step)


loss_cb = LossHistoryCallback()
model.gradient_checkpointing_enable()
model.train()

# CPT_MICRO_BS x grad_accum = CPT_EFFECTIVE_BS, as in 1A (which picks the split from free
# VRAM at start-up). Labs GPUs seen so far (A100-80GB) handle micro-batch 4 comfortably;
# this is a fixed, conservative choice rather than 1A's full OOM-retry/auto-halving logic.
CPT_MICRO_BS = 1 if DEV else 4
CPT_GRAD_ACCUM = max(1, CPT_EFFECTIVE_BS // CPT_MICRO_BS)
steps_per_epoch = math.ceil(len(train_dataset) / CPT_EFFECTIVE_BS)
total_cpt_steps = steps_per_epoch * CPT_EPOCHS

cpt_args = TrainingArguments(
    output_dir="./cpt_checkpoints",
    num_train_epochs=CPT_EPOCHS,
    per_device_train_batch_size=CPT_MICRO_BS,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=CPT_GRAD_ACCUM,
    learning_rate=CPT_LR,
    lr_scheduler_type="cosine",
    warmup_steps=max(1, round(0.05 * total_cpt_steps)),
    weight_decay=0.01,
    max_grad_norm=1.0,
    logging_steps=1,
    eval_strategy="no",
    save_strategy="no",
    fp16=(dtype == torch.float16),
    bf16=(dtype == torch.bfloat16),
    report_to="none",
    use_cpu=(device == "cpu"),
)

trainer = Trainer(model=model, args=cpt_args, train_dataset=train_dataset, callbacks=[loss_cb])
print(f"\nStarting Continual Pre-Training (CPT): {CPT_EPOCHS} epochs, {total_cpt_steps} steps, "
      f"micro-batch {CPT_MICRO_BS} x grad-accum {CPT_GRAD_ACCUM} = effective batch {CPT_EFFECTIVE_BS}")
trainer.train()

cpt_model_path = "./cpt_final_model"
trainer.model.save_pretrained(cpt_model_path)
tokenizer.save_pretrained(cpt_model_path)
print(f"CPT model saved to {cpt_model_path}")
if loss_cb.losses:
    print(f"CPT starting loss: {loss_cb.losses[0]:.4f} | final loss: {loss_cb.losses[-1]:.4f}")

with open("cpt_loss_history.json", "w") as f:
    json.dump({"steps": loss_cb.steps, "losses": loss_cb.losses}, f)

# ---------------------------------------------------------------------------
# Stage 2: QLoRA SFT on top of the CPT checkpoint (genuine 4-bit NF4, matching 1A)
# ---------------------------------------------------------------------------
print("\n--- Loading CPT checkpoint for QLoRA SFT ---")
with open("instruction_dataset.jsonl") as f:
    train_instructions = [json.loads(l) for l in f]
print(f"Instruction pairs: {len(train_instructions)}")

if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    qlora_model = AutoModelForCausalLM.from_pretrained(
        cpt_model_path, quantization_config=bnb_config, device_map={"": 0}, torch_dtype=dtype,
    )
    qlora_model = prepare_model_for_kbit_training(qlora_model, use_gradient_checkpointing=True)
else:
    # bitsandbytes 4-bit requires CUDA; dev mode on this CPU/MPS Mac falls back to full precision
    # to validate the pipeline logic only -- the real run (--mode full) is always on a CUDA GPU.
    print("[dev mode] no CUDA GPU -- loading in full precision instead of 4-bit (pipeline validation only)")
    qlora_model = AutoModelForCausalLM.from_pretrained(cpt_model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    qlora_model.to(device)

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGETS,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(qlora_model, peft_config)
peft_model.print_trainable_parameters()


def encode_pair(instruction, response):
    prompt_ids = tokenizer(chat_prompt(instruction), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(response, add_special_tokens=False)["input_ids"] + [EOS_ID]
    input_ids = (prompt_ids + answer_ids)[:SFT_MAX_LEN]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:SFT_MAX_LEN]
    return {"input_ids": input_ids, "labels": labels}


class ResponseOnlyCollator:
    # Pads input_ids with the pad token and labels with -100, so padding never contributes
    # to the loss -- matches 1A's ResponseOnlyCollator exactly.
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        L = max(len(f["input_ids"]) for f in features)
        ids = [f["input_ids"] + [self.pad_id] * (L - len(f["input_ids"])) for f in features]
        labs = [f["labels"] + [-100] * (L - len(f["labels"])) for f in features]
        mask = [[1] * len(f["input_ids"]) + [0] * (L - len(f["input_ids"])) for f in features]
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labs),
            "attention_mask": torch.tensor(mask),
        }


sft_train_ds = HFDataset.from_list([encode_pair(item["instruction"], item["response"]) for item in train_instructions])
collator = ResponseOnlyCollator(tokenizer.pad_token_id)

if DEV:
    sft_train_ds = sft_train_ds.select(range(min(8, len(sft_train_ds))))

SFT_MICRO_BS = 1 if DEV else 2
SFT_GRAD_ACCUM = max(1, SFT_EFFECTIVE_BS // SFT_MICRO_BS)
total_sft_steps = math.ceil(len(sft_train_ds) / SFT_EFFECTIVE_BS) * SFT_EPOCHS

sft_loss_cb = LossHistoryCallback()
sft_args = TrainingArguments(
    output_dir="./qlora_adapter_checkpoints",
    num_train_epochs=SFT_EPOCHS,
    per_device_train_batch_size=SFT_MICRO_BS,
    gradient_accumulation_steps=SFT_GRAD_ACCUM,
    learning_rate=SFT_LR,
    lr_scheduler_type="cosine",
    warmup_steps=max(1, round(0.10 * total_sft_steps)),
    weight_decay=0.0,
    max_grad_norm=0.3,
    optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
    logging_steps=1,
    save_strategy="no",
    fp16=(dtype == torch.float16),
    bf16=(dtype == torch.bfloat16),
    report_to="none",
    use_cpu=(device == "cpu"),
)

sft_trainer = Trainer(
    model=peft_model, args=sft_args, train_dataset=sft_train_ds,
    data_collator=collator, callbacks=[sft_loss_cb],
)
print(f"\nStarting QLoRA Fine-Tuning: {SFT_EPOCHS} epochs, {total_sft_steps} steps, "
      f"micro-batch {SFT_MICRO_BS} x grad-accum {SFT_GRAD_ACCUM} = effective batch {SFT_EFFECTIVE_BS}")
sft_trainer.train()

adapter_save_dir = "./final_qlora_adapter"
peft_model.save_pretrained(adapter_save_dir)
tokenizer.save_pretrained(adapter_save_dir)
print(f"QLoRA adapter saved to {adapter_save_dir}")
if sft_loss_cb.losses:
    print(f"SFT starting loss: {sft_loss_cb.losses[0]:.4f} | final loss: {sft_loss_cb.losses[-1]:.4f}")

with open("sft_loss_history.json", "w") as f:
    json.dump({"steps": sft_loss_cb.steps, "losses": sft_loss_cb.losses}, f)

print("\nDONE.")
