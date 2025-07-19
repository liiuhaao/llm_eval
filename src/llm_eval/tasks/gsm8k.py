import re

import datasets

from llm_eval.task_factory import TaskFactory


@TaskFactory.register_task("GSM8K")
class GSM8K:
    def load():
        train_dataset = datasets.load_dataset("openai/gsm8k", "main", split="train")
        test_dataset = datasets.load_dataset("openai/gsm8k", "main", split="test")
        dataset = datasets.concatenate_datasets([train_dataset, test_dataset])
        return dataset

    def build(sample):
        system_prompt = 'You are a helpful assistant that answers math questions. Think step by step and then finish your answer with "answer is (X)" where X is the numerical value.'

        question = sample["question"]
        input = f"Question: {question}\n\n"

        user_prompt = input

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        answer = sample["answer"]
        truth_match = re.search(r"#### (-?\d+(?:\.\d+)?)", answer)
        truth = truth_match.group(1) if truth_match else None

        return {
            "input": question,
            "messages": messages,
            "truth": truth,
            "category": "gsm8k",
        }

    def extract(text):
        def extract_answer(text):
            pattern = r"answer is $([\d\.\-]+)$"
            match = re.search(pattern, text)
            if match:
                return match.group(1)
            else:
                return extract_again(text)

        def extract_again(text):
            match = re.search(r".*[aA]nswer:\s*([\d\.\-]+)", text)
            if match:
                return match.group(1)
            else:
                return extract_final(text)

        def extract_final(text):
            pattern = r"[\+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?=\D|$)"
            matches = re.findall(pattern, text)
            if matches:
                return matches[-1]
            else:
                return None

        result = extract_answer(text)
        return result
