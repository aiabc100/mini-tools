import os


class Config:
    data_dir = "./data"
    data_path = os.path.join(data_dir, "dataset.json")
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    model_dir = "./checkpoints"
    model_path = os.path.join(model_dir, "model.pt")

    num_samples = 50000
    train_ratio = 0.95
    seed = 42

    vocab_size = 2048
    max_seq_len = 256

    d_model = 256
    n_heads = 8
    n_layers = 6
    d_ff = 1024
    dropout = 0.1

    batch_size = 64
    learning_rate = 3e-4
    weight_decay = 0.01
    max_epochs = 30
    warmup_steps = 500
    grad_clip = 1.0

    num_workers = 2
    num_gpus = 2

    inference_max_len = 256
    inference_temperature = 0.8
    inference_top_k = 50

    special_tokens = ["<pad>", "<eos>", "<sep>", "<unk>", "<code>"]
    pad_id = 0
    eos_id = 1
    sep_id = 2
    unk_id = 3
    code_id = 4
