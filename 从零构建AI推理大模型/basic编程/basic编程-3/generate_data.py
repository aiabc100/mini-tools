import json
import os
import random
import argparse
from config import Config


ADD_TEMPLATES = [
    "Input {a} and {b}, and print their sum",
    "Calculate the sum of {a} and {b}",
    "Add {a} and {b} together and print the result",
    "Write a program to add {a} and {b}",
    "Find the sum of {a} and {b}",
    "Compute {a} plus {b}",
    "What is {a} added to {b}?",
    "Print the result of adding {a} and {b}",
    "Add together {a} and {b} and display it",
    "Show the sum when you add {a} and {b}",
    "Calculate {a} + {b}",
    "Find the total of {a} and {b}",
    "Sum up {a} and {b} and print",
    "Get the addition result of {a} and {b}",
    "What do you get when you add {a} and {b}?",
]

SUB_TEMPLATES = [
    "Input {a} and {b}, and print their difference",
    "Calculate the difference of {a} and {b}",
    "Subtract {b} from {a} and print the result",
    "Write a program to subtract {b} from {a}",
    "Find the difference between {a} and {b}",
    "Compute {a} minus {b}",
    "What is {a} subtracted by {b}?",
    "Print the result of subtracting {b} from {a}",
    "Show the difference when you subtract {b} from {a}",
    "Calculate {a} - {b}",
    "Subtract {b} from {a} and display the answer",
    "What is the result of {a} minus {b}?",
    "Find {a} take away {b}",
    "Get the subtraction result of {a} and {b}",
    "What do you get when you subtract {b} from {a}?",
]

MUL_TEMPLATES = [
    "Input {a} and {b}, and print their product",
    "Calculate the product of {a} and {b}",
    "Multiply {a} and {b} together and print the result",
    "Write a program to multiply {a} and {b}",
    "Find the product of {a} and {b}",
    "Compute {a} times {b}",
    "What is {a} multiplied by {b}?",
    "Print the result of multiplying {a} and {b}",
    "Show the product when you multiply {a} and {b}",
    "Calculate {a} * {b}",
    "Multiply {a} by {b} and display the answer",
    "What is the result of {a} times {b}?",
    "Find the multiplication of {a} and {b}",
    "Get the product of {a} and {b}",
    "What do you get when you multiply {a} and {b}?",
]

DIV_TEMPLATES = [
    "Input {a} and {b}, and print their quotient",
    "Calculate the quotient of {a} and {b}",
    "Divide {a} by {b} and print the result",
    "Write a program to divide {a} by {b}",
    "Find the quotient of {a} divided by {b}",
    "Compute {a} divided by {b}",
    "What is {a} divided by {b}?",
    "Print the result of dividing {a} by {b}",
    "Show the quotient when you divide {a} by {b}",
    "Calculate {a} / {b}",
    "Divide {a} by {b} and display the answer",
    "What is the result of {a} divided by {b}?",
    "Find the division of {a} by {b}",
    "Get the quotient of {a} and {b}",
    "What do you get when you divide {a} by {b}?",
]


ADD_REASONING = [
    "The user wants to add {a} and {b}. I need to create a BASIC program that assigns {a} to A and {b} to B, then prints their sum. I will use assignment statements to set A={a} and B={b}, then use PRINT with the + operator to display the result.",
    "This is an addition problem with {a} and {b}. The program should assign A={a} and B={b}, then output their sum using PRINT A + B.",
    "To add {a} and {b}, I need a BASIC program with assignment statements A={a} and B={b}, and a PRINT statement that adds them with the + operator.",
    "The task is to compute the sum of {a} and {b}. I will write a BASIC program that assigns A={a} and B={b}, then prints A + B.",
    "For adding {a} and {b}, the program needs to assign A={a} and B={b}, then display their sum using PRINT A + B.",
]

SUB_REASONING = [
    "The user wants to subtract {b} from {a}. I need to create a BASIC program that assigns {a} to A and {b} to B, then prints their difference. I will use assignment statements to set A={a} and B={b}, then use PRINT with the - operator to display the result.",
    "This is a subtraction problem with {a} and {b}. The program should assign A={a} and B={b}, then output their difference using PRINT A - B.",
    "To subtract {b} from {a}, I need a BASIC program with assignment statements A={a} and B={b}, and a PRINT statement that subtracts them with the - operator.",
    "The task is to compute the difference of {a} and {b}. I will write a BASIC program that assigns A={a} and B={b}, then prints A - B.",
    "For subtracting {b} from {a}, the program needs to assign A={a} and B={b}, then display their difference using PRINT A - B.",
]

MUL_REASONING = [
    "The user wants to multiply {a} and {b}. I need to create a BASIC program that assigns {a} to A and {b} to B, then prints their product. I will use assignment statements to set A={a} and B={b}, then use PRINT with the * operator to display the result.",
    "This is a multiplication problem with {a} and {b}. The program should assign A={a} and B={b}, then output their product using PRINT A * B.",
    "To multiply {a} and {b}, I need a BASIC program with assignment statements A={a} and B={b}, and a PRINT statement that multiplies them with the * operator.",
    "The task is to compute the product of {a} and {b}. I will write a BASIC program that assigns A={a} and B={b}, then prints A * B.",
    "For multiplying {a} and {b}, the program needs to assign A={a} and B={b}, then display their product using PRINT A * B.",
]

DIV_REASONING = [
    "The user wants to divide {a} by {b}. I need to create a BASIC program that assigns {a} to A and {b} to B, then prints their quotient. I will use assignment statements to set A={a} and B={b}, then use PRINT with the / operator to display the result.",
    "This is a division problem with {a} and {b}. The program should assign A={a} and B={b}, then output their quotient using PRINT A / B.",
    "To divide {a} by {b}, I need a BASIC program with assignment statements A={a} and B={b}, and a PRINT statement that divides them with the / operator.",
    "The task is to compute the quotient of {a} divided by {b}. I will write a BASIC program that assigns A={a} and B={b}, then prints A / B.",
    "For dividing {a} by {b}, the program needs to assign A={a} and B={b}, then display their quotient using PRINT A / B.",
]


def generate_basic_add(a, b):
    return f"10 A = {a}\n20 B = {b}\n30 PRINT A + B"


def generate_basic_sub(a, b):
    return f"10 A = {a}\n20 B = {b}\n30 PRINT A - B"


def generate_basic_mul(a, b):
    return f"10 A = {a}\n20 B = {b}\n30 PRINT A * B"


def generate_basic_div(a, b):
    return f"10 A = {a}\n20 B = {b}\n30 PRINT A / B"


def generate_dataset(num_samples, seed=42):
    random.seed(seed)
    dataset = []
    ops = [
        ("add", ADD_TEMPLATES, ADD_REASONING, generate_basic_add),
        ("sub", SUB_TEMPLATES, SUB_REASONING, generate_basic_sub),
        ("mul", MUL_TEMPLATES, MUL_REASONING, generate_basic_mul),
        ("div", DIV_TEMPLATES, DIV_REASONING, generate_basic_div),
    ]

    per_op = num_samples // 4

    for op_name, templates, reasoning_templates, gen_func in ops:
        for _ in range(per_op):
            a = random.randint(0, 9)
            b = random.randint(0, 9)
            if op_name == "div" and b == 0:
                b = random.randint(1, 9)

            template = random.choice(templates)
            prompt = template.format(a=a, b=b)

            reasoning_template = random.choice(reasoning_templates)
            reasoning = reasoning_template.format(a=a, b=b)

            basic_code = gen_func(a, b)

            dataset.append({
                "prompt": prompt,
                "reasoning": reasoning,
                "code": basic_code,
                "operation": op_name,
                "a": a,
                "b": b,
            })

    random.shuffle(dataset)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=Config.num_samples)
    parser.add_argument("--seed", type=int, default=Config.seed)
    args = parser.parse_args()

    os.makedirs(Config.data_dir, exist_ok=True)

    dataset = generate_dataset(args.num_samples, args.seed)

    with open(Config.data_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(dataset)} samples, saved to {Config.data_path}")

    op_counts = {}
    for item in dataset:
        op = item["operation"]
        op_counts[op] = op_counts.get(op, 0) + 1
    for op, count in op_counts.items():
        print(f"  {op}: {count}")

    print("\nSample data:")
    for i in range(2):
        print(f"\n  Prompt:    {dataset[i]['prompt']}")
        print(f"  Reasoning: {dataset[i]['reasoning']}")
        print(f"  Code:      {repr(dataset[i]['code'])}")


if __name__ == "__main__":
    main()
