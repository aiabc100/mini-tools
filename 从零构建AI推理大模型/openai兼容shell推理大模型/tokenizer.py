import json
import os
import re
from collections import Counter

from config import MODEL_CONFIG, PATHS, SPECIAL_TOKENS


class BPETokenizer:
    def __init__(self, vocab_size=None):
        self.vocab_size = vocab_size or MODEL_CONFIG["vocab_size"]
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = SPECIAL_TOKENS
        self._initialized = False

    def _get_special_token_ids(self):
        return {
            "pad": self.token_to_id.get(self.special_tokens["pad"], 0),
            "bos": self.token_to_id.get(self.special_tokens["bos"], 1),
            "eos": self.token_to_id.get(self.special_tokens["eos"], 2),
            "unk": self.token_to_id.get(self.special_tokens["unk"], 3),
        }

    @property
    def pad_token_id(self):
        return self._get_special_token_ids()["pad"]

    @property
    def bos_token_id(self):
        return self._get_special_token_ids()["bos"]

    @property
    def eos_token_id(self):
        return self._get_special_token_ids()["eos"]

    def _text_to_bytes(self, text):
        return list(text.encode("utf-8"))

    def _bytes_to_text(self, byte_list):
        return bytes(byte_list).decode("utf-8", errors="replace")

    def _get_pairs(self, token_list):
        pairs = Counter()
        for i in range(len(token_list) - 1):
            pairs[(token_list[i], token_list[i + 1])] += 1
        return pairs

    def _merge_pair(self, token_list, pair, new_token):
        merged = []
        i = 0
        while i < len(token_list):
            if i < len(token_list) - 1 and token_list[i] == pair[0] and token_list[i + 1] == pair[1]:
                merged.append(new_token)
                i += 2
            else:
                merged.append(token_list[i])
                i += 1
        return merged

    def train(self, texts, verbose=True):
        special_list = [
            self.special_tokens["pad"],
            self.special_tokens["bos"],
            self.special_tokens["eos"],
            self.special_tokens["unk"],
        ]

        byte_tokens = [f"<0x{b:02X}>" for b in range(256)]

        self.token_to_id = {}
        self.id_to_token = {}
        idx = 0
        for st in special_list:
            self.token_to_id[st] = idx
            self.id_to_token[idx] = st
            idx += 1
        for bt in byte_tokens:
            self.token_to_id[bt] = idx
            self.id_to_token[idx] = bt
            idx += 1

        tokenized_corpus = []
        for text in texts:
            byte_seq = self._text_to_bytes(text)
            token_seq = [f"<0x{b:02X}>" for b in byte_seq]
            tokenized_corpus.append(token_seq)

        num_merges = self.vocab_size - idx
        if verbose:
            print(f"Base vocab: {idx} tokens, performing {num_merges} merges...")

        for merge_i in range(num_merges):
            pair_counts = Counter()
            for token_seq in tokenized_corpus:
                for i in range(len(token_seq) - 1):
                    pair_counts[(token_seq[i], token_seq[i + 1])] += 1

            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)

            self.token_to_id[new_token] = idx
            self.id_to_token[idx] = new_token
            idx += 1

            for j in range(len(tokenized_corpus)):
                tokenized_corpus[j] = self._merge_pair(
                    tokenized_corpus[j], best_pair, new_token
                )

            if verbose and (merge_i + 1) % 200 == 0:
                print(f"  Merge {merge_i + 1}/{num_merges}: {best_pair} -> {new_token} (freq: {pair_counts[best_pair]})")

        self._initialized = True
        if verbose:
            print(f"Tokenizer trained: {len(self.token_to_id)} tokens")

    def encode(self, text):
        if not self._initialized:
            raise RuntimeError("Tokenizer not trained or loaded")

        if text in self.token_to_id:
            return [self.token_to_id[text]]

        byte_seq = self._text_to_bytes(text)
        tokens = [f"<0x{b:02X}>" for b in byte_seq]

        for pair in self.merges:
            new_token = pair[0] + pair[1]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        ids = []
        for t in tokens:
            if t in self.token_to_id:
                ids.append(self.token_to_id[t])
            else:
                ids.append(self.token_to_id.get(self.special_tokens["unk"], 3))
        return ids

    def decode(self, ids):
        if not self._initialized:
            raise RuntimeError("Tokenizer not trained or loaded")

        tokens = []
        for id_ in ids:
            if id_ in self.id_to_token:
                token = self.id_to_token[id_]
                if token in [self.special_tokens["pad"], self.special_tokens["bos"],
                             self.special_tokens["eos"], self.special_tokens["unk"]]:
                    continue
                tokens.append(token)
            else:
                continue

        byte_list = []
        for token in tokens:
            if token.startswith("<0x") and token.endswith(">"):
                try:
                    byte_val = int(token[3:-1], 16)
                    byte_list.append(byte_val)
                except ValueError:
                    pass
            else:
                sub_bytes = []
                i = 0
                while i < len(token):
                    if token[i:i+4].startswith("<0x") and i + 5 < len(token) and token[i+4] == ">":
                        try:
                            byte_val = int(token[i+3:i+4], 16)
                            sub_bytes.append(byte_val)
                            i += 5
                        except ValueError:
                            i += 1
                    else:
                        sub_bytes.extend(token[i].encode("utf-8"))
                        i += 1
                byte_list.extend(sub_bytes)

        try:
            return bytes(byte_list).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def save(self, path=None):
        path = path or PATHS["tokenizer_file"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "vocab_size": len(self.token_to_id),
            "merges": self.merges,
            "token_to_id": self.token_to_id,
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {path}")

    def load(self, path=None):
        path = path or PATHS["tokenizer_file"]
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.merges = [tuple(m) for m in data["merges"]]
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        if "special_tokens" in data:
            self.special_tokens = data["special_tokens"]
        self._initialized = True
        print(f"Tokenizer loaded: {len(self.token_to_id)} tokens from {path}")

    def __len__(self):
        return len(self.token_to_id)


def train_tokenizer_from_dataset():
    from dataset import generate_dataset, format_training_pair

    print("Generating dataset for tokenizer training...")
    samples = generate_dataset()
    texts = [format_training_pair(s) for s in samples]

    print(f"Training BPE tokenizer on {len(texts)} samples...")
    tokenizer = BPETokenizer()
    tokenizer.train(texts, verbose=True)
    tokenizer.save()

    test_text = "What's the weather in Beijing?"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\nTest encode/decode:")
    print(f"  Input:  {test_text}")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: {decoded}")

    return tokenizer


if __name__ == "__main__":
    train_tokenizer_from_dataset()
