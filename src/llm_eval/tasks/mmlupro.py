import re

import datasets

from llm_eval.task_factory import TaskFactory


@TaskFactory.register_task("MMLU-Pro")
class MMLUPro:
    def load():
        dataset = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        return dataset

    def build(sample):
        system_prompt = 'The following are multiple choice questions (with answers). Think step by step and then finish your answer with "answer is (X)" where X is the correct letter choice.'

        question = sample["question"]
        options = sample["options"]
        truth = sample["answer"]
        category = f"mmlupro_{sample['category']}"

        input = f"{question}\n\n"
        for i, option in enumerate(options):
            if option == "N/A":
                continue
            input += f"{chr(65 + i)}. {option}\n"

        user_prompt = input

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return {
            "input": sample["question"],
            "messages": messages,
            "truth": truth,
            "category": category,
        }

    def extract(text):
        def extract_answer(text):
            pattern = r"answer is \(?([A-J])\)?"
            match = re.search(pattern, text)
            if match:
                return match.group(1)
            else:
                return extract_again(text)

        def extract_again(text):
            match = re.search(r".*[aA]nswer:\s*([A-J])", text)
            if match:
                return match.group(1)
            else:
                return extract_final(text)

        def extract_final(text):
            pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(0)
            else:
                return None

        return extract_answer(text)
