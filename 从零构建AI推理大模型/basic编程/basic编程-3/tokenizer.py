import json
import os
from config import Config


class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.special_tokens = Config.special_tokens
        self.pad_id = Config.pad_id
        self.eos_id = Config.eos_id
        self.sep_id = Config.sep_id
        self.unk_id = Config.unk_id
        self.code_id = Config.code_id

        for i, token in enumerate(self.special_tokens):
            self.char_to_id[token] = i
            self.id_to_char[i] = token

    def train(self, texts):
        chars = set()
        for text in texts:
            chars.update(set(text))
        chars = sorted(chars)

        idx = len(self.special_tokens)
        for char in chars:
            if char not in self.char_to_id:
                self.char_to_id[char] = idx
                self.id_to_char[idx] = char
                idx += 1

    def encode(self, text):
        ids = []
        for char in text:
            if char in self.char_to_id:
                ids.append(self.char_to_id[char])
            else:
                ids.append(self.unk_id)
        return ids

    def decode(self, ids):
        chars = []
        for id_ in ids:
            if id_ in self.id_to_char:
                token = self.id_to_char[id_]
                if token == "<eos>":
                    break
                if token == "<pad>":
                    continue
                if token in self.special_tokens:
                    chars.append(" ")
                    continue
                chars.append(token)
            else:
                chars.append("?")
        return "".join(chars)

    def save(self, path):
        data = {
            "char_to_id": self.char_to_id,
            "special_tokens": self.special_tokens,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char_to_id = {k: int(v) for k, v in data["char_to_id"].items()}
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
        self.special_tokens = data["special_tokens"]

    @property
    def get_vocab_size(self):
        return len(self.char_to_id)
