import os
import time
import math
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import MODEL_CONFIG, TRAIN_CONFIG, PATHS
from dataset import generate_dataset, split_and_save, create_dataloaders
from tokenizer import BPETokenizer, train_tokenizer_from_dataset
from model import GPTModel


def get_cosine_scheduler(optimizer, warmup_steps, max_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, grad_accum_steps=1):
    model.train()
    total_loss = 0
    total_steps = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=TRAIN_CONFIG["fp16"]):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"] / grad_accum_steps

        if loss.dim() > 0:
            loss = loss.mean()

        if TRAIN_CONFIG["fp16"]:
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        else:
            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        total_loss += loss.item() * grad_accum_steps
        total_steps += 1

        if (step + 1) % 50 == 0:
            avg_loss = total_loss / total_steps
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch} Step {step+1}/{len(train_loader)} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

    return total_loss / total_steps


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_steps = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=TRAIN_CONFIG["fp16"]):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        total_loss += outputs["loss"].mean().item()
        total_steps += 1

    avg_loss = total_loss / total_steps
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity


def save_model(model, tokenizer, path, optimizer=None, epoch=None, best_loss=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config,
        "epoch": epoch,
        "best_loss": best_loss,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("model_config", MODEL_CONFIG)
    model = GPTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint


def test_inference(model, tokenizer, device):
    model.eval()
    test_prompts = [
        "What's the weather in Beijing?",
        "List files in /home/user.",
        "How is the weather in Tokyo?",
        "Show me the contents of /var/log.",
        "Can you tell me the weather in London?",
        "What files are in /tmp?",
    ]

    print("\n" + "=" * 60)
    print("INFERENCE TEST")
    print("=" * 60)

    for prompt in test_prompts:
        from dataset import format_input
        input_text = format_input(prompt)
        input_ids = torch.tensor([tokenizer.encode(input_text)], dtype=torch.long).to(device)

        output_ids = model.generate(
            input_ids, tokenizer,
            max_new_tokens=64,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.2,
        )

        decoded = tokenizer.decode(output_ids)
        if "<eos>" in decoded:
            decoded = decoded.split("<eos>")[0]
        if "<bos>" in decoded:
            parts = decoded.split("<eos>")
            if len(parts) > 1:
                decoded = parts[1]
            else:
                decoded = decoded.replace("<bos>", "").strip()

        print(f"\n  Prompt:  {prompt}")
        print(f"  Output:  {decoded.strip()}")


def main():
    print("=" * 60)
    print("TRAINING FROM SCRATCH - Custom LLM")
    print("=" * 60)

    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n[Step 1] Generating dataset...")
    samples = generate_dataset()
    train_samples, val_samples = split_and_save(samples)

    print("\n[Step 2] Training tokenizer...")
    tokenizer = train_tokenizer_from_dataset()

    actual_vocab_size = len(tokenizer)
    MODEL_CONFIG["vocab_size"] = actual_vocab_size
    print(f"Updated vocab_size to {actual_vocab_size}")

    print("\n[Step 3] Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        tokenizer,
        MODEL_CONFIG["max_seq_len"],
        TRAIN_CONFIG["batch_size"],
        TRAIN_CONFIG["num_workers"],
    )

    print("\n[Step 4] Building model...")
    model = GPTModel(MODEL_CONFIG)

    if n_gpus > 1:
        print(f"Using DataParallel with {n_gpus} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    param_info = model.module.count_parameters() if hasattr(model, "module") else model.count_parameters()
    print(f"Total params: {param_info['total'] / 1e6:.2f}M")
    print(f"Trainable params: {param_info['trainable'] / 1e6:.2f}M")

    optimizer = AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        betas=(0.9, 0.95),
    )

    total_steps = len(train_loader) * TRAIN_CONFIG["epochs"]
    scheduler = get_cosine_scheduler(optimizer, TRAIN_CONFIG["warmup_steps"], total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=TRAIN_CONFIG["fp16"])

    best_val_loss = float("inf")
    os.makedirs(PATHS["model_dir"], exist_ok=True)
    os.makedirs(PATHS["log_dir"], exist_ok=True)

    train_log = []

    print(f"\n[Step 5] Training for {TRAIN_CONFIG['epochs']} epochs...")
    print(f"  Steps per epoch: {len(train_loader)}")
    print(f"  Total steps: {total_steps}")

    for epoch in range(1, TRAIN_CONFIG["epochs"] + 1):
        print(f"\n--- Epoch {epoch}/{TRAIN_CONFIG['epochs']} ---")
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch
        )

        val_loss, val_ppl = evaluate(model, val_loader, device)

        elapsed = time.time() - start_time
        print(f"\n  Epoch {epoch} Summary:")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Val Loss:   {val_loss:.4f}")
        print(f"    Val PPL:    {val_ppl:.2f}")
        print(f"    Time:       {elapsed:.1f}s")

        train_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "time": elapsed,
        })

        with open(os.path.join(PATHS["log_dir"], "training_log.json"), "w") as f:
            json.dump(train_log, f, indent=2)

        raw_model = model.module if hasattr(model, "module") else model

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(raw_model, tokenizer, PATHS["best_model"], optimizer, epoch, best_val_loss)
            print(f"  *** New best model! Val Loss: {best_val_loss:.4f} ***")

        if epoch % TRAIN_CONFIG["save_every"] == 0:
            ckpt_path = os.path.join(PATHS["model_dir"], f"checkpoint_epoch_{epoch}.pt")
            save_model(raw_model, tokenizer, ckpt_path, optimizer, epoch, val_loss)

        if epoch % TRAIN_CONFIG["eval_every"] == 0:
            test_inference(raw_model if hasattr(model, "module") else model, tokenizer, device)

        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)

    raw_model = model.module if hasattr(model, "module") else model
    test_inference(raw_model, tokenizer, device)

    return model, tokenizer


if __name__ == "__main__":
    main()
