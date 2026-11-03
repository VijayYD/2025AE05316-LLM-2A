"""
Reproduces Assignment 1A's two-stage pipeline (CPT then QLoRA SFT) on
HuggingFaceTB/SmolLM2-360M, using the same hyperparameters as the 1A notebook.
The original 1A adapter weights were lost (ephemeral Colab storage), so this
script regenerates them from the same domain_corpus/ and instruction_dataset.jsonl.

Usage:
  python train_adapter.py --mode dev    # fast, truncated run to validate the pipeline (this Mac, CPU)
  python train_adapter.py --mode full   # exact 1A hyperparameters (10 CPT epochs, 8 SFT epochs) -- run on Prayogshala
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
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset as HFDataset

MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
SEQ_LENGTH = 1024

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["dev", "full"], required=True)
args = parser.parse_args()
DEV = args.mode == "dev"

torch.manual_seed(42)
random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
    torch.float16 if torch.cuda.is_available() else torch.float32
)
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

all_token_ids = []
for doc_name, text in cleaned_documents.items():
    tokens = [tokenizer.bos_token_id] + tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
    all_token_ids.extend(tokens)
total_tokens = len(all_token_ids)
print(f"Total tokens collected: {total_tokens}")

packed_chunks = []
for i in range(0, total_tokens - SEQ_LENGTH + 1, SEQ_LENGTH):
    packed_chunks.append(all_token_ids[i:i + SEQ_LENGTH])
print(f"Packed sequences: {len(packed_chunks)}")

if DEV:
    packed_chunks = packed_chunks[:40]
    print(f"[dev mode] truncated to {len(packed_chunks)} packed sequences")

df = pd.DataFrame({"input_ids": packed_chunks, "labels": packed_chunks})
parquet_path = "cpt_packed.parquet"
df.to_parquet(parquet_path)


class CPTDataset(Dataset):
    def __init__(self, parquet_file):
        self.df = pd.read_parquet(parquet_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


full_dataset = CPTDataset(parquet_path)
train_size = int(0.9 * len(full_dataset))
eval_size = len(full_dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42)
)
print(f"CPT Train Sequences: {len(train_dataset)}, CPT Validation Sequences: {len(eval_dataset)}")

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

cpt_args = TrainingArguments(
    output_dir="./cpt_checkpoints",
    num_train_epochs=1 if DEV else 10,
    max_steps=15 if DEV else -1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    warmup_steps=3,
    weight_decay=0.01,
    logging_steps=1,
    save_strategy="no",
    fp16=(dtype == torch.float16),
    bf16=(dtype == torch.bfloat16),
    report_to="none",
    use_cpu=(device == "cpu"),
)

trainer = Trainer(model=model, args=cpt_args, train_dataset=train_dataset, callbacks=[loss_cb])
print("\nStarting Continual Pre-Training (CPT)...")
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
# Stage 2: QLoRA SFT on top of the CPT checkpoint
# ---------------------------------------------------------------------------
print("\n--- Loading CPT checkpoint for QLoRA SFT ---")
with open("instruction_dataset.jsonl") as f:
    train_instructions = [json.loads(l) for l in f]
print(f"Instruction pairs: {len(train_instructions)}")

qlora_model = AutoModelForCausalLM.from_pretrained(cpt_model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
qlora_model.to(device)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(qlora_model, peft_config)
peft_model.print_trainable_parameters()

formatted_prompts = []
for item in train_instructions:
    full_prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}{tokenizer.eos_token}"
    formatted_prompts.append(full_prompt)

sft_hf_dataset = HFDataset.from_dict({"text": formatted_prompts})


def tokenize_sft(examples):
    tokens = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    tokens["labels"] = [
        [tid if am == 1 else -100 for tid, am in zip(ids, am_row)]
        for ids, am_row in zip(tokens["input_ids"], tokens["attention_mask"])
    ]
    return tokens


tokenized_sft_dataset = sft_hf_dataset.map(tokenize_sft, batched=True, remove_columns=["text"])

sft_loss_cb = LossHistoryCallback()
sft_args = TrainingArguments(
    output_dir="./qlora_adapter_checkpoints",
    num_train_epochs=1 if DEV else 8,
    max_steps=15 if DEV else -1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    warmup_steps=2,
    logging_steps=1,
    save_strategy="no",
    fp16=(dtype == torch.float16),
    bf16=(dtype == torch.bfloat16),
    report_to="none",
    use_cpu=(device == "cpu"),
)

sft_trainer = Trainer(model=peft_model, args=sft_args, train_dataset=tokenized_sft_dataset, callbacks=[sft_loss_cb])
print("\nStarting QLoRA Fine-Tuning...")
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
