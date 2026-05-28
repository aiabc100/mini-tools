import json
import torch
from config import Config
from tokenizer import CharTokenizer
from model import GPTModel


def load_model_and_tokenizer(model_path=None, tokenizer_path=None, device=None):
    config = Config
    if model_path is None:
        model_path = config.model_path
    if tokenizer_path is None:
        tokenizer_path = config.tokenizer_path
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CharTokenizer()
    tokenizer.load(tokenizer_path)

    model = GPTModel(config)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size}")
    print(f"Device: {device}")

    return model, tokenizer, device


def generate_basic(model, tokenizer, prompt, device,
                   max_len=256, temperature=0.8, top_k=50):
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
    code_sep_pos = None
    for i, id_ in enumerate(generated):
        if id_ == tokenizer.sep_id and sep_pos is None:
            sep_pos = i
        if id_ == tokenizer.code_id and sep_pos is not None and code_sep_pos is None:
            code_sep_pos = i

    if sep_pos is not None and code_sep_pos is not None:
        reasoning_ids = generated[sep_pos + 1: code_sep_pos]
        code_ids = generated[code_sep_pos + 1:]
    elif sep_pos is not None:
        reasoning_ids = []
        code_ids = generated[sep_pos + 1:]
    else:
        reasoning_ids = []
        code_ids = generated[len(prompt_ids):]

    reasoning_text = tokenizer.decode(reasoning_ids)
    code_text = tokenizer.decode(code_ids)

    print(f"[Prompt]    {prompt}")
    print(f"[Reasoning] {reasoning_text.strip()}")
    print(f"[BASIC Code]")
    for line in code_text.strip().split(chr(10)):
        print(f"  {line}")

    return reasoning_text, code_text


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=Config.inference_top_k)
    parser.add_argument("--max_len", type=int, default=Config.inference_max_len)
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer()
    generate_basic(
        model, tokenizer, args.prompt, device,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
