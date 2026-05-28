import json
import os
import gc
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import Config
from tokenizer import CharTokenizer
from model import GPTModel


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
        reasoning = item["reasoning"]
        code = item["code"]

        prompt_ids = self.tokenizer.encode(prompt)
        reasoning_ids = self.tokenizer.encode(reasoning)
        code_ids = self.tokenizer.encode(code)

        input_ids = (
            prompt_ids
            + [self.tokenizer.sep_id]
            + reasoning_ids
            + [self.tokenizer.code_id]
            + code_ids
            + [self.tokenizer.eos_id]
        )

        sep_pos = len(prompt_ids)
        code_sep_pos = len(prompt_ids) + 1 + len(reasoning_ids)

        labels = [-100] * (sep_pos + 1) + input_ids[sep_pos + 1:]

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]

        pad_len = self.max_seq_len - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train():
    config = Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()

    print(f"Device: {device}, GPUs: {n_gpus}")

    with open(config.data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    split_idx = int(len(dataset) * config.train_ratio)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    all_texts = []
    for item in dataset:
        all_texts.append(item["prompt"])
        all_texts.append(item["reasoning"])
        all_texts.append(item["code"])

    tokenizer = CharTokenizer()
    tokenizer.train(all_texts)
    tokenizer.save(config.tokenizer_path)
    print(f"Vocab size: {tokenizer.get_vocab_size}")

    del all_texts
    gc.collect()

    train_dataset = BasicDataset(train_data, tokenizer, config.max_seq_len, is_train=True)
    val_dataset = BasicDataset(val_data, tokenizer, config.max_seq_len, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    model = GPTModel(config)

    if n_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    total_steps = len(train_loader) * config.max_epochs
    warmup_steps = config.warmup_steps

    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    print(f"Starting training for {config.max_epochs} epochs...")

    best_val_loss = float("inf")
    os.makedirs(config.model_dir, exist_ok=True)

    for epoch in range(config.max_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_idx
            lr = get_lr(step, warmup_steps, total_steps, config.learning_rate, config.learning_rate * 0.1)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = model(input_ids, labels=labels)
            if n_gpus > 1:
                loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / n_batches
                print(f"  Epoch {epoch+1}/{config.max_epochs} | Step {batch_idx+1}/{len(train_loader)} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")

        avg_train_loss = total_loss / n_batches
        epoch_time = time.time() - start_time

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

        print(f"Epoch {epoch+1}/{config.max_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model = model.module if n_gpus > 1 else model
            torch.save({
                "epoch": epoch,
                "model_state_dict": save_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "config": {k: v for k, v in vars(config).items() if not k.startswith("_")},
            }, config.model_path)
            print(f"  -> Best model saved (val_loss: {avg_val_loss:.4f})")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
