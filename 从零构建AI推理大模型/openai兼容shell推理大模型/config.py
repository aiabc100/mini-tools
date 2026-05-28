import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CONFIG = {
    "vocab_size": 4096,
    "max_seq_len": 256,
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 2048,
    "dropout": 0.1,
    "pad_token_id": 0,
}

TRAIN_CONFIG = {
    "batch_size": 32,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "epochs": 20,
    "warmup_steps": 500,
    "max_grad_norm": 1.0,
    "save_every": 5,
    "eval_every": 1,
    "fp16": True,
    "num_workers": 2,
}

DATA_CONFIG = {
    "num_weather_samples": 3000,
    "num_directory_samples": 3000,
    "train_ratio": 0.9,
    "seed": 42,
}

PATHS = {
    "data_dir": os.path.join(BASE_DIR, "data"),
    "raw_data": os.path.join(BASE_DIR, "data", "raw_dataset.json"),
    "train_data": os.path.join(BASE_DIR, "data", "train.json"),
    "val_data": os.path.join(BASE_DIR, "data", "val.json"),
    "tokenizer_dir": os.path.join(BASE_DIR, "tokenizer"),
    "tokenizer_file": os.path.join(BASE_DIR, "tokenizer", "tokenizer.json"),
    "model_dir": os.path.join(BASE_DIR, "checkpoints"),
    "best_model": os.path.join(BASE_DIR, "checkpoints", "best_model.pt"),
    "log_dir": os.path.join(BASE_DIR, "logs"),
}

SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "model_name": "custom-llm",
    "max_new_tokens": 128,
    "temperature": 0.1,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
}

SPECIAL_TOKENS = {
    "pad": "<pad>",
    "bos": "<bos>",
    "eos": "<eos>",
    "unk": "<unk>",
}
