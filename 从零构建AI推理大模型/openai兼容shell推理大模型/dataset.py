import json
import os
import random
import re
from torch.utils.data import Dataset, DataLoader
import torch

from config import DATA_CONFIG, PATHS, SPECIAL_TOKENS


CITIES = [
    "Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Chengdu",
    "Hangzhou", "Wuhan", "Xi'an", "Nanjing", "Chongqing",
    "Tianjin", "Suzhou", "Qingdao", "Dalian", "Xiamen",
    "New York", "London", "Tokyo", "Paris", "Berlin",
    "Sydney", "Toronto", "Moscow", "Dubai", "Singapore",
    "Seoul", "Mumbai", "Bangkok", "Cairo", "Rome",
    "Madrid", "Amsterdam", "Vienna", "Prague", "Budapest",
    "Lisbon", "Stockholm", "Oslo", "Helsinki", "Copenhagen",
    "Dublin", "Zurich", "Geneva", "Brussels", "Warsaw",
    "Istanbul", "Jakarta", "Manila", "Kuala Lumpur", "Hanoi",
]

WEATHER_TEMPLATES = [
    "What's the weather in {city}?",
    "What is the weather like in {city}?",
    "How is the weather in {city}?",
    "Tell me the weather in {city}.",
    "Show me the weather for {city}.",
    "Get the weather for {city}.",
    "Check weather in {city}.",
    "Weather in {city}.",
    "What's the weather like in {city} today?",
    "What's the current weather in {city}?",
    "Can you tell me the weather in {city}?",
    "Could you check the weather in {city}?",
    "I want to know the weather in {city}.",
    "Please show me the weather in {city}.",
    "Give me the weather update for {city}.",
    "Is it raining in {city}?",
    "Is it sunny in {city}?",
    "How cold is it in {city}?",
    "How hot is it in {city}?",
    "What temperature is it in {city}?",
    "What's the forecast for {city}?",
    "Weather forecast for {city}.",
    "Current conditions in {city}.",
    "What's it like outside in {city}?",
    "Do I need an umbrella in {city}?",
    "Should I wear a jacket in {city}?",
    "Is it going to rain in {city}?",
    "Will it snow in {city}?",
    "How windy is it in {city}?",
    "What's the humidity in {city}?",
]

DIRECTORY_PATHS = [
    "/home/user", "/home/user/documents", "/home/user/downloads",
    "/home/user/pictures", "/home/user/desktop", "/home/user/music",
    "/home/user/videos", "/var/log", "/var/www", "/etc",
    "/tmp", "/opt", "/usr/local", "/root", "/home/admin",
    "/data", "/data/projects", "/data/models", "/data/logs",
    "/srv/app", "/srv/www", "/workspace", "/workspace/src",
    "C:\\Users\\Admin", "C:\\Users\\Admin\\Desktop",
    "C:\\Users\\Admin\\Documents", "C:\\Users\\Admin\\Downloads",
    "C:\\Program Files", "C:\\Projects",
]

DIRECTORY_TEMPLATES = [
    "List files in {path}.",
    "Show me the files in {path}.",
    "What files are in {path}?",
    "List directory {path}.",
    "Show directory contents of {path}.",
    "List all files in {path}.",
    "What's in {path}?",
    "Display the contents of {path}.",
    "Show what's inside {path}.",
    "List the files under {path}.",
    "Can you list the files in {path}?",
    "Could you show me the contents of {path}?",
    "I want to see the files in {path}.",
    "Please list the directory {path}.",
    "What directories are in {path}?",
    "Show me the folder contents of {path}.",
    "List all items in {path}.",
    "Browse {path}.",
    "Open directory {path}.",
    "Explore {path}.",
    "What is inside the folder {path}?",
    "Show the file listing for {path}.",
    "Display files and folders in {path}.",
    "Get a directory listing for {path}.",
    "What can I find in {path}?",
    "List the contents of the folder {path}.",
    "Show me everything in {path}.",
    "Enumerate files in {path}.",
    "Read the directory {path}.",
    "What does {path} contain?",
]


def generate_weather_samples(n):
    samples = []
    for _ in range(n):
        city = random.choice(CITIES)
        template = random.choice(WEATHER_TEMPLATES)
        prompt = template.format(city=city)
        output = json.dumps({
            "name": "get_weather",
            "arguments": {"city": city}
        }, separators=(",", ":"))
        samples.append({
            "prompt": prompt,
            "output": output,
            "task": "weather"
        })
    return samples


def generate_directory_samples(n):
    samples = []
    for _ in range(n):
        path = random.choice(DIRECTORY_PATHS)
        template = random.choice(DIRECTORY_TEMPLATES)
        prompt = template.format(path=path)
        output = json.dumps({
            "name": "list_directory",
            "arguments": {"path": path}
        }, separators=(",", ":"))
        samples.append({
            "prompt": prompt,
            "output": output,
            "task": "directory"
        })
    return samples


def generate_dataset():
    random.seed(DATA_CONFIG["seed"])
    weather = generate_weather_samples(DATA_CONFIG["num_weather_samples"])
    directory = generate_directory_samples(DATA_CONFIG["num_directory_samples"])
    all_samples = weather + directory
    random.shuffle(all_samples)
    return all_samples


def save_dataset(samples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_and_save(samples):
    random.seed(DATA_CONFIG["seed"])
    random.shuffle(samples)
    split_idx = int(len(samples) * DATA_CONFIG["train_ratio"])
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    save_dataset(samples, PATHS["raw_data"])
    save_dataset(train_samples, PATHS["train_data"])
    save_dataset(val_samples, PATHS["val_data"])
    print(f"Total: {len(samples)}, Train: {len(train_samples)}, Val: {len(val_samples)}")
    return train_samples, val_samples


def format_input(prompt):
    return f"{SPECIAL_TOKENS['bos']} {prompt} {SPECIAL_TOKENS['eos']}"


def format_output(output):
    return f"{output}{SPECIAL_TOKENS['eos']}"


def format_training_pair(sample):
    src = format_input(sample["prompt"])
    tgt = format_output(sample["output"])
    full = f"{src} {tgt}"
    return full


class FunctionCallDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        full_text = format_training_pair(sample)
        token_ids = self.tokenizer.encode(full_text)
        token_ids = token_ids[:self.max_len]

        prompt_text = format_input(sample["prompt"])
        prompt_ids = self.tokenizer.encode(prompt_text)
        prompt_len = min(len(prompt_ids), self.max_len)

        pad_len = self.max_len - len(token_ids)
        input_ids = token_ids + [0] * pad_len
        attention_mask = [1] * len(token_ids) + [0] * pad_len

        labels = [-100] * prompt_len + token_ids[prompt_len:]
        labels = labels[:self.max_len]
        pad_len_label = self.max_len - len(labels)
        labels = labels + [-100] * pad_len_label

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def create_dataloaders(tokenizer, max_len, batch_size, num_workers=2):
    train_samples = load_dataset(PATHS["train_data"])
    val_samples = load_dataset(PATHS["val_data"])

    train_dataset = FunctionCallDataset(train_samples, tokenizer, max_len)
    val_dataset = FunctionCallDataset(val_samples, tokenizer, max_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
