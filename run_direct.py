"""
E1 - Baseline "Direct": mo hinh viet code truc tiep (1 lan goi, khong co Analyst/Tester).
Dung cung mo hinh va cung dinh dang output voi run_humaneval.py de cham bang evaluate.sh.
"""
import argparse
import json
import re

import tqdm
from datasets import load_dataset

from core.config import ModelConfig
from core.backend import call_llm
from core.utils import find_method_name

SYSTEM_PROMPT = "You are a Python developer. Complete the given function."

USER_TEMPLATE = """Complete the following Python function. Return the complete function \
(including the signature and any needed imports) in a single ```python code block.

```python
{prompt}
```"""


def extract_code(text):
    """Lay code trong khoi ```python ... ``` (neu co)."""
    if not text:
        return ""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        for b in blocks:
            if "def " in b:
                return b.strip()
        return blocks[0].strip()
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="HumanEval: Direct baseline")
    parser.add_argument("--output_path", type=str, default="out_E1_direct.jsonl")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    config = ModelConfig(max_tokens=2048, temperature=0.0)
    print(f"Model: {config.model}")

    dataset = load_dataset("openai/openai_humaneval", split="test")
    if args.max_samples:
        dataset = list(dataset)[: args.max_samples]

    n_ok = 0
    with open(args.output_path, "w") as f:
        for task in tqdm.tqdm(dataset):
            task_id = task["task_id"]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(prompt=task["prompt"])},
            ]
            try:
                reply = call_llm(messages, config)[0]
            except Exception as e:
                print(f"  {task_id}: API error: {e}")
                reply = ""

            code = extract_code(reply)
            before_func = task["prompt"][: task["prompt"].rfind("def ")]
            solution = {
                "task_id": task_id,
                "prompt": before_func + "\n",
                "test": task["test"],
                "entry_point": find_method_name(code) or task["entry_point"],
                "completion": code,
            }
            f.write(json.dumps(solution) + "\n")
            f.flush()
            if code:
                n_ok += 1

    print(f"\nDone. {n_ok}/{len(dataset)} tasks have code. Output: {args.output_path}")


if __name__ == "__main__":
    main()
