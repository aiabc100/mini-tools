import os
import torch
from config import Config
from model import GPTModel
from tokenizer import CharTokenizer


def load_model_and_tokenizer(model_path=None, tokenizer_path=None):
    config = Config()
    model_path = model_path or config.model_path
    tokenizer_path = tokenizer_path or config.tokenizer_path

    tokenizer = CharTokenizer()
    tokenizer.load(tokenizer_path)

    checkpoint = torch.load(model_path, map_location="cpu")

    saved_config = checkpoint.get("config", {})
    if saved_config:
        config.vocab_size = saved_config.get("vocab_size", config.vocab_size)
        config.d_model = saved_config.get("d_model", config.d_model)
        config.n_heads = saved_config.get("n_heads", config.n_heads)
        config.n_layers = saved_config.get("n_layers", config.n_layers)
        config.d_ff = saved_config.get("d_ff", config.d_ff)
        config.max_seq_len = saved_config.get("max_seq_len", config.max_seq_len)
        config.dropout = saved_config.get("dropout", config.dropout)

    model = GPTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Device: {device}")

    return model, tokenizer, config, device


def generate_basic(model, tokenizer, prompt, device,
                   max_len=128, temperature=0.8, top_k=50):
    prompt_ids = tokenizer.encode(prompt)
    input_ids = prompt_ids + [tokenizer.sep_id]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    output_ids = model.generate(
        input_tensor,
        max_len=max_len,
        temperature=temperature,
        top_k=top_k,
    )

    generated = output_ids[0].tolist()

    sep_pos = None
    for i, id_ in enumerate(generated):
        if id_ == tokenizer.sep_id:
            sep_pos = i
            break

    if sep_pos is not None:
        code_ids = generated[sep_pos + 1:]
    else:
        code_ids = generated[len(prompt_ids):]

    code_text = tokenizer.decode(code_ids)

    return code_text


def interactive_test(model_path=None, tokenizer_path=None):
    model, tokenizer, config, device = load_model_and_tokenizer(model_path, tokenizer_path)

    test_prompts = [
        "Input 3 and 4, and print their sum",
        "Calculate the difference of 9 and 2",
        "Multiply 5 and 6 together and print the result",
        "Divide 8 by 4 and print the result",
        "Compute 7 plus 3",
        "What is 6 subtracted by 1?",
        "Find the product of 2 and 8",
        "What is 9 divided by 3?",
    ]

    print("\n" + "=" * 60)
    print("BASIC Code Generation Test")
    print("=" * 60)

    for prompt in test_prompts:
        code = generate_basic(
            model, tokenizer, prompt, device,
            max_len=config.inference_max_len,
            temperature=config.inference_temperature,
            top_k=config.inference_top_k,
        )
        print(f"\nPrompt: {prompt}")
        print(f"BASIC Code:\n{code}")
        print("-" * 40)

    print("\nInteractive mode (type 'quit' to exit):")
    while True:
        prompt = input("\nEnter prompt: ").strip()
        if prompt.lower() in ["quit", "exit", "q"]:
            break
        if not prompt:
            continue

        code = generate_basic(
            model, tokenizer, prompt, device,
            max_len=config.inference_max_len,
            temperature=config.inference_temperature,
            top_k=config.inference_top_k,
        )
        print(f"BASIC Code:\n{code}")


if __name__ == "__main__":
    interactive_test()
