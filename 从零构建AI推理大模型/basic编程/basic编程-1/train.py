import os
import json
import time
import math
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from config import Config
from model import GPTModel
from tokenizer import CharTokenizer


class BasicDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len, is_train=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        code = item["code"]

        prompt_ids = self.tokenizer.encode(prompt)
        code_ids = self.tokenizer.encode(code)

        input_ids = prompt_ids + [self.tokenizer.sep_id] + code_ids + [self.tokenizer.eos_id]

        labels = [-100] * len(prompt_ids) + [-100] + code_ids + [self.tokenizer.eos_id]

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]

        pad_len = self.max_seq_len - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
        labels = labels + [-100] * pad_len

        attention_mask = [1] * (self.max_seq_len - pad_len) + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Using device: {device}, GPUs: {n_gpus}")

    os.makedirs(config.model_dir, exist_ok=True)

    print("Loading dataset...")
    with open(config.data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    split_idx = int(len(dataset) * config.train_ratio)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    print("Training tokenizer...")
    all_texts = []
    for item in dataset:
        all_texts.append(item["prompt"])
        all_texts.append(item["code"])

    tokenizer = CharTokenizer()
    tokenizer.train(all_texts)
    tokenizer.save(config.tokenizer_path)

    actual_vocab_size = tokenizer.get_vocab_size
    config.vocab_size = actual_vocab_size
    print(f"Vocab size: {actual_vocab_size}")

    train_dataset = BasicDataset(train_data, tokenizer, config.max_seq_len, is_train=True)
    val_dataset = BasicDataset(val_data, tokenizer, config.max_seq_len, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    print("Building model...")
    model = GPTModel(config)
    model = model.to(device)

    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel with {n_gpus} GPUs")

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    total_steps = len(train_loader) * config.max_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)

    best_val_loss = float("inf")
    global_step = 0

    print(f"Starting training for {config.max_epochs} epochs, {total_steps} total steps")
    print(f"Batch size: {config.batch_size}, Warmup: {config.warmup_steps}")

    for epoch in range(config.max_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        start_time = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = model(input_ids, labels=labels)

            if n_gpus > 1:
                loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 100 == 0:
                avg_loss = total_loss / n_batches
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {global_step} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")

        avg_train_loss = total_loss / n_batches
        elapsed = time.time() - start_time

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                logits, loss = model(input_ids, labels=labels)
                if n_gpus > 1:
                    loss = loss.mean()
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        print(f"Epoch {epoch + 1}/{config.max_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Time: {elapsed:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model = model.module if n_gpus > 1 else model
            torch.save({
                "epoch": epoch,
                "model_state_dict": save_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "config": {
                    "vocab_size": config.vocab_size,
                    "d_model": config.d_model,
                    "n_heads": config.n_heads,
                    "n_layers": config.n_layers,
                    "d_ff": config.d_ff,
                    "max_seq_len": config.max_seq_len,
                    "dropout": config.dropout,
                },
            }, config.model_path)
            print(f"  -> Best model saved (val_loss: {avg_val_loss:.4f})")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {config.model_path}")


if __name__ == "__main__":
    train()
