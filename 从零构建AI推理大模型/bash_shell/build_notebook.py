import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source if isinstance(source, list) else [source]}

cells = []

cells.append(md([
    "# Building a Bash Shell Command Inference Model from Scratch\n",
    "\n",
    "This notebook trains a **from-scratch GPT model** that infers bash shell commands (`ls`, `cd`, `pwd`) from English natural language descriptions.\n",
    "\n",
    "**Platform**: Kaggle T4 GPU (16GB VRAM)\n",
    "\n",
    "**Training Pipeline**:\n",
    "1. Generate 5000 training samples\n",
    "2. Train a BPE tokenizer from scratch\n",
    "3. Build a Mini GPT (Decoder-Only Transformer)\n",
    "4. **Phase 1: SFT** (Supervised Fine-Tuning) — cross-entropy on instruction-response pairs\n",
    "5. **Phase 2: GRPO** (Group Relative Policy Optimization) — RL with reward function\n",
    "6. Save model & run inference"
]))

cells.append(md("## Step 0: Environment Setup"))

cells.append(code("""import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import random
import os
import math
import time
import copy
from collections import Counter
from pathlib import Path

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
print(f"Device: {DEVICE}")"""))

cells.append(md("## Step 1: Generate Training Dataset (5000 samples)\n\nWe generate instruction-response pairs covering basic usage of `ls`, `cd`, and `pwd`."))

cells.append(code("""def generate_ls_data():
    data = []
    templates = [
        ("List files in the current directory", "ls"),
        ("Show me the files in this directory", "ls"),
        ("What files are in the current folder", "ls"),
        ("Display the contents of the current directory", "ls"),
        ("Show all files here", "ls"),
        ("List the items in this folder", "ls"),
        ("What is in the current directory", "ls"),
        ("Show directory contents", "ls"),
        ("List what is in this directory", "ls"),
        ("Display files in the working directory", "ls"),
        ("Show me what files exist here", "ls"),
        ("List all entries in the current directory", "ls"),
        ("What files and folders are here", "ls"),
        ("Print the directory listing", "ls"),
        ("Show the contents of this folder", "ls"),
        ("List everything in the current directory", "ls"),
        ("What is inside this directory", "ls"),
        ("Give me a listing of files", "ls"),
        ("Show files and directories in current path", "ls"),
        ("List the files here", "ls"),
    ]
    data.extend(templates)

    ls_l_templates = [
        ("List files in long format", "ls -l"),
        ("Show detailed file listing", "ls -l"),
        ("Display files with permissions and sizes", "ls -l"),
        ("List files with details", "ls -l"),
        ("Show file details including permissions", "ls -l"),
        ("Long listing of files", "ls -l"),
        ("List files in detailed format", "ls -l"),
        ("Show files with ownership and permissions", "ls -l"),
        ("Display a detailed directory listing", "ls -l"),
        ("List files showing size and date", "ls -l"),
        ("Show me the long listing format", "ls -l"),
        ("Display files with metadata", "ls -l"),
        ("List files with full information", "ls -l"),
        ("Show file attributes", "ls -l"),
        ("Give me a verbose listing of files", "ls -l"),
    ]
    data.extend(ls_l_templates)

    ls_a_templates = [
        ("List all files including hidden ones", "ls -a"),
        ("Show hidden files too", "ls -a"),
        ("Display all files including dotfiles", "ls -a"),
        ("List files including hidden files", "ls -a"),
        ("Show all entries including hidden", "ls -a"),
        ("Display hidden and visible files", "ls -a"),
        ("List everything including dot files", "ls -a"),
        ("Show me all files even the hidden ones", "ls -a"),
        ("Include hidden files in the listing", "ls -a"),
        ("List all entries including those starting with a dot", "ls -a"),
        ("Show all files in the directory including hidden", "ls -a"),
        ("Display all files even hidden ones", "ls -a"),
        ("List all including concealed files", "ls -a"),
        ("Show visible and hidden files", "ls -a"),
        ("Reveal all files including dotfiles", "ls -a"),
    ]
    data.extend(ls_a_templates)

    ls_la_templates = [
        ("List all files in long format including hidden", "ls -la"),
        ("Show detailed listing of all files including hidden", "ls -la"),
        ("Display all files with details and hidden files", "ls -la"),
        ("Long listing of all files including dotfiles", "ls -la"),
        ("List all files with permissions including hidden", "ls -la"),
        ("Show detailed view of all files including hidden ones", "ls -la"),
        ("Full detailed listing with hidden files", "ls -la"),
        ("Display all entries in long format", "ls -la"),
        ("List everything with full details", "ls -la"),
        ("Show all files with ownership and hidden ones", "ls -la"),
        ("Give me a complete detailed listing", "ls -la"),
        ("Verbose listing of all files", "ls -la"),
        ("Show all files with metadata including hidden", "ls -la"),
        ("List all entries with full attributes", "ls -la"),
        ("Display complete file information including hidden", "ls -la"),
    ]
    data.extend(ls_la_templates)

    ls_path_templates = [
        ("List files in the home directory", "ls ~"),
        ("Show files in my home folder", "ls ~"),
        ("Display contents of the home directory", "ls ~"),
        ("List what is in the home directory", "ls ~"),
        ("Show me the home directory contents", "ls ~"),
        ("List files in the root directory", "ls /"),
        ("Show files in the root folder", "ls /"),
        ("Display contents of the root directory", "ls /"),
        ("List what is in the root directory", "ls /"),
        ("Show me the root directory contents", "ls /"),
    ]
    data.extend(ls_path_templates)

    ls_path_variations = []
    paths = ["/tmp", "/var", "/etc", "/usr", "/home", "/opt", "/var/log", "/usr/local"]
    path_descs = [
        "List files in the {p} directory",
        "Show files in {p}",
        "Display contents of {p}",
        "What files are in {p}",
        "Show me what is in {p}",
    ]
    for p in paths:
        for desc in path_descs:
            ls_path_variations.append((desc.format(p=p), f"ls {p}"))
    data.extend(ls_path_variations)

    return data


def generate_cd_data():
    data = []
    cd_home_templates = [
        ("Change to the home directory", "cd ~"),
        ("Go to my home directory", "cd ~"),
        ("Switch to the home folder", "cd ~"),
        ("Navigate to home", "cd ~"),
        ("Go home", "cd ~"),
        ("Return to home directory", "cd ~"),
        ("Change directory to home", "cd ~"),
        ("Move to the home directory", "cd ~"),
        ("Take me to my home folder", "cd ~"),
        ("Switch to home directory", "cd ~"),
        ("Go back to my home directory", "cd ~"),
        ("Navigate to the home folder", "cd ~"),
        ("Change to home", "cd ~"),
        ("Open the home directory", "cd ~"),
        ("Enter the home directory", "cd ~"),
    ]
    data.extend(cd_home_templates)

    cd_up_templates = [
        ("Go up one directory", "cd .."),
        ("Move to the parent directory", "cd .."),
        ("Change to the parent folder", "cd .."),
        ("Navigate up one level", "cd .."),
        ("Go to the previous directory", "cd .."),
        ("Move up one level", "cd .."),
        ("Change to the directory above", "cd .."),
        ("Go back one directory level", "cd .."),
        ("Navigate to the parent folder", "cd .."),
        ("Switch to the parent directory", "cd .."),
        ("Ascend one directory level", "cd .."),
        ("Step up to the parent directory", "cd .."),
        ("Go up a level in the directory tree", "cd .."),
        ("Move back one directory", "cd .."),
        ("Change to the upper directory", "cd .."),
    ]
    data.extend(cd_up_templates)

    cd_prev_templates = [
        ("Go back to the previous directory I was in", "cd -"),
        ("Return to the last directory", "cd -"),
        ("Switch back to the previous directory", "cd -"),
        ("Go to the directory I came from", "cd -"),
        ("Toggle back to the last directory", "cd -"),
        ("Return to previous working directory", "cd -"),
        ("Switch to the previous directory", "cd -"),
        ("Go back to where I was before", "cd -"),
        ("Change back to the last directory", "cd -"),
        ("Navigate to the previous directory", "cd -"),
    ]
    data.extend(cd_prev_templates)

    cd_root_templates = [
        ("Change to the root directory", "cd /"),
        ("Go to the root directory", "cd /"),
        ("Navigate to root", "cd /"),
        ("Switch to the root folder", "cd /"),
        ("Move to the root directory", "cd /"),
        ("Change directory to root", "cd /"),
        ("Go to the top level directory", "cd /"),
        ("Navigate to the root of the filesystem", "cd /"),
        ("Open the root directory", "cd /"),
        ("Enter the root directory", "cd /"),
    ]
    data.extend(cd_root_templates)

    cd_path_variations = []
    paths = ["/tmp", "/var", "/etc", "/usr", "/home", "/opt", "/var/log", "/usr/local", "/home/user", "/home/user/projects"]
    path_descs = [
        "Change to the {p} directory",
        "Go to {p}",
        "Navigate to {p}",
        "Switch to the {p} folder",
        "Move to {p}",
        "Open {p}",
        "Enter the {p} directory",
        "Change directory to {p}",
    ]
    for p in paths:
        for desc in path_descs:
            cd_path_variations.append((desc.format(p=p), f"cd {p}"))
    data.extend(cd_path_variations)

    return data


def generate_pwd_data():
    data = []
    pwd_templates = [
        ("What is my current working directory", "pwd"),
        ("Show the current directory", "pwd"),
        ("Print working directory", "pwd"),
        ("Where am I in the filesystem", "pwd"),
        ("What directory am I in", "pwd"),
        ("Display the current path", "pwd"),
        ("Show me my current location", "pwd"),
        ("What is the current working directory", "pwd"),
        ("Tell me the current directory path", "pwd"),
        ("Print the current directory", "pwd"),
        ("Show current working directory", "pwd"),
        ("Where am I right now", "pwd"),
        ("What is my present working directory", "pwd"),
        ("Display current directory path", "pwd"),
        ("Get the current working directory", "pwd"),
        ("Output the current directory name", "pwd"),
        ("What path am I currently at", "pwd"),
        ("Show the present working directory", "pwd"),
        ("Which directory am I currently in", "pwd"),
        ("Give me the current directory", "pwd"),
        ("What folder am I in", "pwd"),
        ("Show my current location in the file system", "pwd"),
        ("Display the full path of the current directory", "pwd"),
        ("Tell me where I am in the directory tree", "pwd"),
        ("What is the path of the current directory", "pwd"),
    ]
    data.extend(pwd_templates)
    return data


def augment_data(data, target_count=5000):
    augmented = list(data)
    base_count = len(augmented)

    paraphrase_words = {
        "show": ["display", "reveal", "present", "output", "expose"],
        "list": ["enumerate", "catalog", "itemize", "outline", "index"],
        "files": ["documents", "items", "entries", "objects", "contents"],
        "directory": ["folder", "path", "location", "directory", "dir"],
        "change": ["switch", "move", "navigate", "go", "enter"],
        "current": ["present", "active", "working", "current"],
        "all": ["every", "each", "all", "complete", "entire"],
        "hidden": ["concealed", "invisible", "dot", "secret", "hidden"],
        "detailed": ["verbose", "full", "complete", "comprehensive", "thorough"],
        "parent": ["upper", "above", "superior", "parent", "preceding"],
        "previous": ["last", "prior", "former", "earlier", "previous"],
        "home": ["home", "personal", "user", "home"],
        "root": ["root", "top", "base", "root", "top-level"],
    }

    random.seed(SEED)
    while len(augmented) < target_count:
        instruction, output = random.choice(data)
        words = instruction.split()
        new_words = []
        changed = False
        for w in words:
            w_lower = w.lower()
            if w_lower in paraphrase_words:
                syns = paraphrase_words[w_lower]
                syn = random.choice(syns)
                if syn != w_lower:
                    changed = True
                if w[0].isupper():
                    new_words.append(syn.capitalize())
                else:
                    new_words.append(syn)
            else:
                new_words.append(w)
        if changed:
            new_instruction = " ".join(new_words)
            augmented.append((new_instruction, output))

    random.shuffle(augmented)
    return augmented[:target_count]


ls_data = generate_ls_data()
cd_data = generate_cd_data()
pwd_data = generate_pwd_data()

all_data = ls_data + cd_data + pwd_data
print(f"Base data count: {len(all_data)}")
print(f"  ls: {len(ls_data)}, cd: {len(cd_data)}, pwd: {len(pwd_data)}")

dataset = augment_data(all_data, target_count=5000)
print(f"Augmented dataset count: {len(dataset)}")

ls_count = sum(1 for _, o in dataset if o.startswith("ls"))
cd_count = sum(1 for _, o in dataset if o.startswith("cd"))
pwd_count = sum(1 for _, o in dataset if o.startswith("pwd"))
print(f"  ls: {ls_count}, cd: {cd_count}, pwd: {pwd_count}")

print("\\nSample data:")
for i in range(5):
    print(f"  {dataset[i][0]} -> {dataset[i][1]}")"""))

cells.append(code("""PROMPT_TEMPLATE = "### Instruction:\\n{instruction}\\n\\n### Response:\\n{output}"
PROMPT_PREFIX = "### Instruction:\\n{instruction}\\n\\n### Response:\\n"
EOS_TOKEN = ""

formatted_data = []
for instruction, output in dataset:
    text = PROMPT_TEMPLATE.format(instruction=instruction, output=output) + EOS_TOKEN
    formatted_data.append(text)

print(f"Total formatted samples: {len(formatted_data)}")
print(f"\\nSample formatted text:\\n{formatted_data[0]}")
print(f"\\nSample formatted text:\\n{formatted_data[100]}")"""))

cells.append(md("## Step 2: Train BPE Tokenizer from Scratch"))

cells.append(code("""class BPETokenizer:
    def __init__(self, vocab_size=2048):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}

    def _get_pairs(self, word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _merge_vocab(self, pair):
        new_vocab = {}
        bigram = pair[0] + pair[1]
        for word, freq in self.word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            new_vocab[new_word] = freq
        self.word_freqs = new_vocab

    def train(self, texts):
        self.word_freqs = Counter()
        for text in texts:
            words = text.split()
            for word in words:
                word_tuple = tuple(word)
                self.word_freqs[word_tuple] += 1

        base_vocab = set()
        for word_tuple in self.word_freqs:
            for char in word_tuple:
                base_vocab.add(char)

        special_tokens = ["", "###", "Instruction:", "Response:"]
        for st in special_tokens:
            base_vocab.add(st)

        self.vocab = {i: ch for i, ch in enumerate(sorted(base_vocab))}
        num_merges = self.vocab_size - len(self.vocab)

        print(f"Base vocabulary size: {len(self.vocab)}")
        print(f"Number of merges to perform: {num_merges}")

        for i in range(num_merges):
            pairs = Counter()
            for word, freq in self.word_freqs.items():
                word_pairs = self._get_pairs(word)
                for p in word_pairs:
                    pairs[p] += freq

            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            new_token = best[0] + best[1]
            self.vocab[len(self.vocab)] = new_token
            self._merge_vocab(best)

            if (i + 1) % 200 == 0:
                print(f"  Merge {i + 1}/{num_merges}: {best} -> {new_token} (freq={pairs[best]})")

        self.token_to_id = {token: idx for idx, token in self.vocab.items()}
        self.id_to_token = {idx: token for idx, token in self.vocab.items()}
        print(f"Final vocabulary size: {len(self.vocab)}")

    def encode(self, text):
        tokens = []
        words = text.split()
        for word in words:
            word_tokens = list(word)
            for merge_idx, (a, b) in enumerate(self.merges):
                i = 0
                new_tokens = []
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and word_tokens[i] == a and word_tokens[i + 1] == b:
                        new_tokens.append(a + b)
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens
            for t in word_tokens:
                if t in self.token_to_id:
                    tokens.append(self.token_to_id[t])
                else:
                    for ch in t:
                        if ch in self.token_to_id:
                            tokens.append(self.token_to_id[ch])
        return tokens

    def decode(self, ids):
        tokens = [self.id_to_token.get(i, "") for i in ids]
        text = " ".join(tokens)
        text = text.replace(" ", "")
        return text

    def save(self, path):
        data = {
            "vocab_size": self.vocab_size,
            "merges": self.merges,
            "vocab": {str(k): v for k, v in self.vocab.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer.vocab = {int(k): v for k, v in data["vocab"].items()}
        tokenizer.token_to_id = {v: int(k) for k, v in tokenizer.vocab.items()}
        tokenizer.id_to_token = {int(k): v for k, v in tokenizer.vocab.items()}
        return tokenizer


VOCAB_SIZE = 2048
tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
tokenizer.train(formatted_data)

tokenizer.save("/kaggle/working/tokenizer.json")
print("Tokenizer saved.")"""))

cells.append(code("""test_text = "### Instruction:\\nList files in the current directory\\n\\n### Response:\\nls"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
print(f"Original:  {test_text}")
print(f"Encoded:   {encoded}")
print(f"Decoded:   {decoded}")
print(f"Roundtrip match: {test_text == decoded}")"""))

cells.append(md("## Step 3: Build Mini GPT Model"))

cells.append(code("""class GPTConfig:
    def __init__(
        self,
        vocab_size=2048,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


config = GPTConfig(
    vocab_size=len(tokenizer.vocab),
    block_size=256,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
)

model = GPT(config).to(DEVICE)
print(f"Model created on {DEVICE}")"""))

cells.append(md("## Step 4: Prepare Dataset & DataLoader"))

cells.append(code("""class BashCommandDataset(Dataset):
    def __init__(self, formatted_texts, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []

        for text in formatted_texts:
            tokens = tokenizer.encode(text)
            tokens = tokens[:block_size + 1]
            if len(tokens) > 1:
                self.examples.append(tokens)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    pad_id = 0
    xs_padded = []
    ys_padded = []
    masks = []
    for x, y in zip(xs, ys):
        pad_len = max_len - x.size(0)
        xs_padded.append(F.pad(x, (0, pad_len), value=pad_id))
        ys_padded.append(F.pad(y, (0, pad_len), value=-100))
        mask = torch.cat([torch.ones(x.size(0)), torch.zeros(pad_len)])
        masks.append(mask)
    return torch.stack(xs_padded), torch.stack(ys_padded), torch.stack(masks)


train_size = int(0.95 * len(formatted_data))
train_texts = formatted_data[:train_size]
val_texts = formatted_data[train_size:]

train_dataset = BashCommandDataset(train_texts, tokenizer, config.block_size)
val_dataset = BashCommandDataset(val_texts, tokenizer, config.block_size)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")"""))

# ============================================================
# Phase 1: SFT
# ============================================================
cells.append(md([
    "---\n",
    "\n",
    "## Phase 1: SFT (Supervised Fine-Tuning)\n",
    "\n",
    "SFT trains the model using standard cross-entropy loss on instruction-response pairs. This teaches the model the basic mapping from natural language to bash commands.\n",
    "\n",
    "**Key characteristics**:\n",
    "- Loss computed only on the **response** portion (masked instruction tokens with `-100`)\n",
    "- Teacher forcing: model sees the ground-truth previous token\n",
    "- AdamW optimizer with cosine LR schedule"
]))

cells.append(code("""SFT_EPOCHS = 60
SFT_LR = 3e-4
MAX_GRAD_NORM = 1.0

sft_optimizer = torch.optim.AdamW(model.parameters(), lr=SFT_LR, weight_decay=0.1, betas=(0.9, 0.95))
sft_total_steps = len(train_loader) * SFT_EPOCHS
sft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(sft_optimizer, T_max=sft_total_steps, eta_min=1e-5)

response_marker_ids = tokenizer.encode("### Response:")
print(f"Response marker token ids: {response_marker_ids}")


def find_response_start(token_ids):
    marker = response_marker_ids
    marker_len = len(marker)
    for i in range(len(token_ids) - marker_len + 1):
        if token_ids[i:i + marker_len] == marker:
            return i + marker_len
    return 0


def sft_train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for xb, yb, mask in loader:
        xb, yb = xb.to(device), yb.to(device)

        masked_yb = yb.clone()
        for b in range(xb.size(0)):
            token_ids = xb[b].tolist()
            resp_start = find_response_start(token_ids)
            for t in range(resp_start):
                masked_yb[b, t] = -100

        logits, loss = model(xb, masked_yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def sft_evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    for xb, yb, mask in loader:
        xb, yb = xb.to(device), yb.to(device)
        masked_yb = yb.clone()
        for b in range(xb.size(0)):
            token_ids = xb[b].tolist()
            resp_start = find_response_start(token_ids)
            for t in range(resp_start):
                masked_yb[b, t] = -100
        logits, loss = model(xb, masked_yb)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


best_sft_val_loss = float("inf")
sft_train_losses = []
sft_val_losses = []

print(f"Starting SFT training for {SFT_EPOCHS} epochs...")
print(f"Total steps: {sft_total_steps}")

sft_start_time = time.time()
for epoch in range(SFT_EPOCHS):
    train_loss = sft_train_epoch(model, train_loader, sft_optimizer, sft_scheduler, DEVICE)
    val_loss = sft_evaluate(model, val_loader, DEVICE)
    sft_train_losses.append(train_loss)
    sft_val_losses.append(val_loss)

    if val_loss < best_sft_val_loss:
        best_sft_val_loss = val_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": config.vocab_size,
                "block_size": config.block_size,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_embd": config.n_embd,
                "dropout": config.dropout,
            },
            "epoch": epoch,
            "val_loss": val_loss,
            "phase": "sft",
        }, "/kaggle/working/sft_best_model.pt")

    if (epoch + 1) % 5 == 0 or epoch == 0:
        elapsed = time.time() - sft_start_time
        print(
            f"[SFT] Epoch {epoch + 1:3d}/{SFT_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Best Val: {best_sft_val_loss:.4f} | "
            f"Time: {elapsed:.0f}s"
        )

print(f"\\nSFT training complete! Best validation loss: {best_sft_val_loss:.4f}")"""))

cells.append(code("""import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(sft_train_losses, label="SFT Train Loss")
plt.plot(sft_val_losses, label="SFT Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Phase 1: SFT Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("/kaggle/working/sft_loss_curve.png", dpi=100)
plt.show()"""))

cells.append(code("""def generate_command(model, tokenizer, instruction, max_new_tokens=50, temperature=0.1, top_k=10):
    prompt = PROMPT_PREFIX.format(instruction=instruction)
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    eos_id = tokenizer.token_to_id.get("", None)
    output_ids = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    generated_ids = output_ids[0].tolist()
    if eos_id is not None and eos_id in generated_ids:
        generated_ids = generated_ids[: generated_ids.index(eos_id)]
    response_ids = generated_ids[len(input_ids):]
    response = tokenizer.decode(response_ids).strip()
    return response


print("=" * 60)
print("SFT Model Quick Test")
print("=" * 60)
test_instructions = [
    "List files in the current directory",
    "Show hidden files too",
    "List all files in long format including hidden",
    "Change to the home directory",
    "Go up one directory",
    "What is my current working directory",
]
for inst in test_instructions:
    cmd = generate_command(model, tokenizer, inst)
    print(f"  Q: {inst}")
    print(f"  A: {cmd}")
    print()"""))

# ============================================================
# Phase 2: GRPO
# ============================================================
cells.append(md([
    "---\n",
    "\n",
    "## Phase 2: GRPO (Group Relative Policy Optimization)\n",
    "\n",
    "GRPO is a reinforcement learning method that improves model outputs by:\n",
    "\n",
    "1. **Sampling**: For each instruction, generate a **group** of G candidate responses\n",
    "2. **Rewarding**: Score each response with a reward function\n",
    "3. **Advantage**: Compute advantages using **group-relative** normalization: $A_i = \\\\frac{r_i - \\\\mu_G}{\\\\sigma_G + \\\\epsilon}$\n",
    "4. **Policy Update**: Use a clipped surrogate objective (similar to PPO) with KL penalty against the reference (SFT) model\n",
    "\n",
    "**GRPO Loss**:\n",
    "$$L_{GRPO} = -\\\\min\\\\left(\\\\rho_i A_i, \\\\, \\\\text{clip}(\\\\rho_i, 1-\\\\epsilon, 1+\\\\epsilon) A_i\\\\right) + \\\\beta \\\\cdot D_{KL}(\\\\pi_\\\\theta \\\\| \\\\pi_{ref})$$\n",
    "\n",
    "where $\\\\rho_i = \\\\frac{\\\\pi_\\\\theta(a_i|s_i)}{\\\\pi_{ref}(a_i|s_i)}$ is the importance ratio.\n",
    "\n",
    "**Reward Function** for bash commands:\n",
    "- Exact match with ground truth: **+1.0**\n",
    "- Correct command name (ls/cd/pwd): **+0.3**\n",
    "- Correct flags/options: **+0.3**\n",
    "- Correct path argument: **+0.2**\n",
    "- Invalid format penalty: **-0.5**"
]))

cells.append(code("""class BashRewardFunction:
    def __init__(self):
        self.valid_commands = {"ls", "cd", "pwd"}

    def _parse_command(self, text):
        text = text.strip()
        if "\\n" in text:
            text = text.split("\\n")[0].strip()
        parts = text.split()
        if not parts:
            return None, [], None
        cmd = parts[0]
        flags = []
        path = None
        for p in parts[1:]:
            if p.startswith("-"):
                flags.append(p)
            else:
                path = p
        return cmd, flags, path

    def compute_reward(self, generated: str, ground_truth: str) -> float:
        gen_cmd, gen_flags, gen_path = self._parse_command(generated)
        gt_cmd, gt_flags, gt_path = self._parse_command(ground_truth)

        if generated.strip() == ground_truth.strip():
            return 1.0

        reward = 0.0

        if gen_cmd is None or gen_cmd not in self.valid_commands:
            return -0.5

        if gen_cmd == gt_cmd:
            reward += 0.3
        else:
            return -0.3

        if gen_flags == gt_flags:
            reward += 0.3
        elif len(gen_flags) > 0 and len(gt_flags) > 0:
            gen_flag_set = set("".join(gen_flags))
            gt_flag_set = set("".join(gt_flags))
            overlap = gen_flag_set & gt_flag_set
            total = gt_flag_set
            if len(total) > 0:
                reward += 0.3 * (len(overlap) / len(total))

        if gt_path is not None:
            if gen_path == gt_path:
                reward += 0.2
            elif gen_path is not None:
                reward += 0.05
        elif gt_path is None and gen_path is None:
            reward += 0.2

        return reward


reward_fn = BashRewardFunction()

test_cases = [
    ("ls", "ls"),
    ("ls -l", "ls -l"),
    ("ls -a", "ls -a"),
    ("ls -la", "ls -la"),
    ("ls", "ls -l"),
    ("ls -l", "ls -la"),
    ("ls /tmp", "ls /tmp"),
    ("ls /var", "ls /tmp"),
    ("cd ~", "cd ~"),
    ("cd ..", "cd .."),
    ("cd /tmp", "cd /tmp"),
    ("pwd", "pwd"),
    ("rm -rf /", "pwd"),
    ("echo hello", "ls"),
]

print("Reward Function Tests:")
print(f"{'Generated':<20} {'Ground Truth':<20} {'Reward':>8}")
print("-" * 50)
for gen, gt in test_cases:
    r = reward_fn.compute_reward(gen, gt)
    print(f"{gen:<20} {gt:<20} {r:>8.2f}")"""))

cells.append(code("""sft_ckpt = torch.load("/kaggle/working/sft_best_model.pt", map_location=DEVICE)
model.load_state_dict(sft_ckpt["model_state_dict"])
print("Loaded best SFT model as policy model.")

ref_model = GPT(config).to(DEVICE)
ref_model.load_state_dict(sft_ckpt["model_state_dict"])
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False
print("Reference model (frozen SFT) created.")"""))

cells.append(code("""GRPO_GROUP_SIZE = 4
GRPO_EPOCHS = 20
GRPO_LR = 1e-5
GRPO_CLIP_EPSILON = 0.2
GRPO_KL_COEFF = 0.05
GRPO_MAX_NEW_TOKENS = 20
GRPO_TEMPERATURE = 0.7
GRPO_SAMPLES_PER_EPOCH = 500

grpo_optimizer = torch.optim.AdamW(model.parameters(), lr=GRPO_LR, weight_decay=0.01)

grpo_instructions = [inst for inst, _ in dataset]
grpo_ground_truths = [out for _, out in dataset]

eos_id = tokenizer.token_to_id.get("", None)


@torch.no_grad()
def generate_group(model, tokenizer, instruction, group_size, temperature, max_new_tokens):
    prompt = PROMPT_PREFIX.format(instruction=instruction)
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    input_tensor = input_tensor.repeat(group_size, 1)

    output_ids = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=50,
    )

    responses = []
    full_sequences = []
    for i in range(group_size):
        gen_ids = output_ids[i].tolist()
        if eos_id is not None and eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id)]
        resp_ids = gen_ids[len(input_ids):]
        resp_text = tokenizer.decode(resp_ids).strip()
        responses.append(resp_text)
        full_sequences.append(gen_ids)

    return responses, full_sequences, input_ids


def compute_sequence_log_probs(model, input_ids, response_ids):
    full_ids = input_ids + response_ids
    full_tensor = torch.tensor([full_ids], dtype=torch.long, device=DEVICE)
    prompt_len = len(input_ids)
    resp_len = len(response_ids)

    if resp_len == 0:
        return torch.tensor(0.0, device=DEVICE)

    x = full_tensor[:, :-1]
    y = full_tensor[:, 1:]

    B, T = x.size()
    pos = torch.arange(0, T, dtype=torch.long, device=DEVICE).unsqueeze(0)
    tok_emb = model.token_embedding(x)
    pos_emb = model.position_embedding(pos)
    h = model.drop(tok_emb + pos_emb)
    for block in model.blocks:
        h = block(h)
    h = model.ln_f(h)
    logits = model.lm_head(h)
    logits = torch.clamp(logits, min=-1e4, max=1e4)

    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1)

    resp_log_probs = target_log_probs[0, prompt_len - 1:]
    result = resp_log_probs.sum()
    if torch.isnan(result) or torch.isinf(result):
        return logits.sum() * 0.0
    return result


print(f"Starting GRPO training for {GRPO_EPOCHS} epochs...")
print(f"  Group size: {GRPO_GROUP_SIZE}")
print(f"  Samples per epoch: {GRPO_SAMPLES_PER_EPOCH}")
print(f"  Clip epsilon: {GRPO_CLIP_EPSILON}")
print(f"  KL coefficient: {GRPO_KL_COEFF}")

grpo_metrics = {"loss": [], "reward_mean": [], "kl_mean": []}
grpo_start_time = time.time()

for epoch in range(GRPO_EPOCHS):
    model.train()
    epoch_loss = 0.0
    epoch_reward = 0.0
    epoch_kl = 0.0
    n_groups = 0

    sample_indices = random.sample(range(len(grpo_instructions)), GRPO_SAMPLES_PER_EPOCH)

    for idx in sample_indices:
        instruction = grpo_instructions[idx]
        ground_truth = grpo_ground_truths[idx]

        responses, full_seqs, prompt_ids = generate_group(
            model, tokenizer, instruction,
            GRPO_GROUP_SIZE, GRPO_TEMPERATURE, GRPO_MAX_NEW_TOKENS
        )

        rewards = []
        for resp in responses:
            r = reward_fn.compute_reward(resp, ground_truth)
            rewards.append(r)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        reward_mean = rewards_tensor.mean()
        reward_std = rewards_tensor.std() + 1e-8
        advantages = (rewards_tensor - reward_mean) / reward_std

        group_losses = []
        group_kl = 0.0

        for g in range(GRPO_GROUP_SIZE):
            resp_ids = full_seqs[g][len(prompt_ids):]
            if len(resp_ids) == 0:
                continue

            old_log_prob = compute_sequence_log_probs(ref_model, prompt_ids, resp_ids).detach()
            new_log_prob = compute_sequence_log_probs(model, prompt_ids, resp_ids)

            log_ratio = new_log_prob - old_log_prob
            log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
            kl_div = log_ratio
            group_kl += kl_div.item()

            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1 - GRPO_CLIP_EPSILON, 1 + GRPO_CLIP_EPSILON)

            adv = advantages[g]
            surr1 = ratio * adv
            surr2 = clipped_ratio * adv
            policy_loss = -torch.min(surr1, surr2)

            group_losses.append(policy_loss + GRPO_KL_COEFF * kl_div)

        if len(group_losses) == 0:
            continue
        group_loss = torch.stack(group_losses).mean()

        grpo_optimizer.zero_grad()
        group_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        grpo_optimizer.step()

        epoch_loss += group_loss.item()
        epoch_reward += reward_mean.item()
        epoch_kl += group_kl / GRPO_GROUP_SIZE
        n_groups += 1

    avg_loss = epoch_loss / n_groups
    avg_reward = epoch_reward / n_groups
    avg_kl = epoch_kl / n_groups

    grpo_metrics["loss"].append(avg_loss)
    grpo_metrics["reward_mean"].append(avg_reward)
    grpo_metrics["kl_mean"].append(avg_kl)

    if (epoch + 1) % 2 == 0 or epoch == 0:
        elapsed = time.time() - grpo_start_time
        print(
            f"[GRPO] Epoch {epoch + 1:3d}/{GRPO_EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Avg Reward: {avg_reward:.3f} | "
            f"Avg KL: {avg_kl:.4f} | "
            f"Time: {elapsed:.0f}s"
        )

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "dropout": config.dropout,
        },
        "epoch": epoch,
        "grpo_metrics": grpo_metrics,
        "phase": "grpo",
    }, "/kaggle/working/grpo_latest_model.pt")

print(f"\\nGRPO training complete!")"""))

cells.append(code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(grpo_metrics["loss"])
axes[0].set_title("GRPO Policy Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True)

axes[1].plot(grpo_metrics["reward_mean"])
axes[1].set_title("GRPO Average Reward")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Reward")
axes[1].grid(True)

axes[2].plot(grpo_metrics["kl_mean"])
axes[2].set_title("GRPO Average KL Divergence")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("KL")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("/kaggle/working/grpo_metrics.png", dpi=100)
plt.show()"""))

# ============================================================
# Final Inference
# ============================================================
cells.append(md("## Step 5: Final Inference (SFT + GRPO Model)"))

cells.append(code("""print("=" * 60)
print("FINAL MODEL (SFT + GRPO) INFERENCE RESULTS")
print("=" * 60)

test_instructions = [
    "List files in the current directory",
    "Show me all files including hidden ones",
    "List all files in long format including hidden",
    "Show detailed file listing",
    "What files are in the home directory",
    "List files in /tmp",
    "Change to the home directory",
    "Go up one directory",
    "Go back to the previous directory I was in",
    "Navigate to the root directory",
    "Switch to /tmp",
    "Go to /var/log",
    "What is my current working directory",
    "Print working directory",
    "Where am I in the filesystem",
    "Show the current directory",
]

for instruction in test_instructions:
    command = generate_command(model, tokenizer, instruction)
    print(f"  Q: {instruction}")
    print(f"  A: {command}")
    print()"""))

# ============================================================
# Save
# ============================================================
cells.append(md("## Step 6: Save Final Model & Artifacts"))

cells.append(code("""torch.save({
    "model_state_dict": model.state_dict(),
    "config": {
        "vocab_size": config.vocab_size,
        "block_size": config.block_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "dropout": config.dropout,
    },
    "grpo_metrics": grpo_metrics,
}, "/kaggle/working/final_model_sft_grpo.pt")

with open("/kaggle/working/train_data.json", "w") as f:
    json.dump([{"instruction": i, "output": o} for i, o in dataset], f, indent=2)

print("All artifacts saved to /kaggle/working/:")
for f in os.listdir("/kaggle/working/"):
    fpath = os.path.join("/kaggle/working/", f)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath)
        print(f"  {f} ({size / 1024:.1f} KB)")"""))

# ============================================================
# Load & Infer standalone
# ============================================================
cells.append(md("## Step 7: Load Model & Run Inference (Standalone)\n\nThis cell demonstrates how to load the saved model and run inference independently."))

cells.append(code("""def load_model_and_infer(model_path, tokenizer_path, instruction, temperature=0.1, top_k=10):
    tok = BPETokenizer.load(tokenizer_path)
    ckpt = torch.load(model_path, map_location=DEVICE)
    cfg = GPTConfig(**ckpt["config"])
    mdl = GPT(cfg).to(DEVICE)
    mdl.load_state_dict(ckpt["model_state_dict"])
    mdl.eval()
    return generate_command(mdl, tok, instruction, temperature=temperature, top_k=top_k)


result = load_model_and_infer(
    "/kaggle/working/final_model_sft_grpo.pt",
    "/kaggle/working/tokenizer.json",
    "List all files including hidden ones in long format",
)
print(f"Instruction: List all files including hidden ones in long format")
print(f"Command: {result}")"""))

# ============================================================
# SFT vs GRPO comparison
# ============================================================
cells.append(md("## Step 8: SFT vs SFT+GRPO Comparison"))

cells.append(code("""sft_only_model = GPT(config).to(DEVICE)
sft_only_model.load_state_dict(sft_ckpt["model_state_dict"])
sft_only_model.eval()

comparison_instructions = [
    "List files in the current directory",
    "Show hidden files",
    "List all files in long format including hidden",
    "Change to the home directory",
    "Go up one directory",
    "Navigate to /var/log",
    "What is my current working directory",
    "Print working directory",
]

print("=" * 70)
print(f"{'Instruction':<45} {'SFT':<12} {'SFT+GRPO':<12}")
print("=" * 70)
for inst in comparison_instructions:
    sft_cmd = generate_command(sft_only_model, tokenizer, inst)
    grpo_cmd = generate_command(model, tokenizer, inst)
    print(f"{inst:<45} {sft_cmd:<12} {grpo_cmd:<12}")
print("=" * 70)"""))

# ============================================================
# Build notebook
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

output_path = r"c:\Users\aiabc\Desktop\从零构建AI推理大模型\bash_shell\bash_shell_inference.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook written to {output_path}")
print(f"Total cells: {len(cells)}")
