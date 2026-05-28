import json
import os
from collections import Counter
from config import Config


class BPETokenizer:
    def __init__(self):
        self.merges = []
        self.vocab = {}
        self.id_to_token = {}
        self.special_tokens = Config.special_tokens
        self.pad_id = Config.pad_id
        self.eos_id = Config.eos_id
        self.sep_id = Config.sep_id
        self.unk_id = Config.unk_id

    def _get_pairs(self, word):
        pairs = set()
        prev = word[0]
        for char in word[1:]:
            pairs.add((prev, char))
            prev = char
        return pairs

    def _build_vocab_from_merges(self, base_vocab):
        vocab = {}
        for token, idx in base_vocab.items():
            vocab[token] = idx
        idx = len(vocab)
        for pair in self.merges:
            new_token = pair[0] + pair[1]
            if new_token not in vocab:
                vocab[new_token] = idx
                idx += 1
        return vocab

    def train(self, texts, vocab_size):
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i

        word_freqs = Counter()
        for text in texts:
            words = text.strip().split()
            for word in words:
                word_tuple = tuple(" " + word if i > 0 else word for i, _ in enumerate([word]))
                word_tuple = tuple(word)
                chars = " ".join(list(word)) + " </w>"
                word_freqs[chars] += 1

        word_splits = {}
        for word, freq in word_freqs.items():
            word_splits[word] = list(word.split())

        base_vocab = dict(self.vocab)
        char_vocab = set()
        for word in word_splits:
            for char in word_splits[word]:
                char_vocab.add(char)
        for char in sorted(char_vocab):
            if char not in base_vocab:
                base_vocab[char] = len(base_vocab)

        num_merges = vocab_size - len(base_vocab)
        self.merges = []

        for _ in range(max(num_merges, 0)):
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                splits = word_splits[word]
                if len(splits) < 2:
                    continue
                for i in range(len(splits) - 1):
                    pair_freqs[(splits[i], splits[i + 1])] += freq

            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)
            self.merges.append(best_pair)

            for word in word_splits:
                splits = word_splits[word]
                new_splits = []
                i = 0
                while i < len(splits):
                    if i < len(splits) - 1 and splits[i] == best_pair[0] and splits[i + 1] == best_pair[1]:
                        new_splits.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_splits.append(splits[i])
                        i += 1
                word_splits[word] = new_splits

        self.vocab = self._build_vocab_from_merges(base_vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        tokens = []
        words = text.strip().split()
        for word in words:
            word_tokens = list(word)
            word_tokens.append("</w>")

            for pair in self.merges:
                new_tokens = []
                i = 0
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                        new_tokens.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens

            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.unk_id)

        return tokens

    def decode(self, ids):
        tokens = []
        for id_ in ids:
            if id_ in self.id_to_token:
                token = self.id_to_token[id_]
                if token in self.special_tokens:
                    if token == "<eos>":
                        break
                    elif token == "<sep>":
                        tokens.append(" ")
                    elif token == "<pad>":
                        continue
                    continue
                tokens.append(token)
            else:
                tokens.append("<unk>")

        text = "".join(tokens)
        text = text.replace("</w>", " ")
        text = text.replace("  ", " ")
        return text.strip()

    def save(self, path):
        data = {
            "merges": self.merges,
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.merges = [tuple(pair) for pair in data["merges"]]
        self.vocab = data["vocab"]
        self.id_to_token = {int(v): k for k, v in self.vocab.items()}
        self.special_tokens = data["special_tokens"]

    @property
    def get_vocab_size(self):
        return len(self.vocab)


class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.special_tokens = Config.special_tokens
        self.pad_id = Config.pad_id
        self.eos_id = Config.eos_id
        self.sep_id = Config.sep_id
        self.unk_id = Config.unk_id

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
