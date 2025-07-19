import re

import datasets

from llm_eval.task_factory import TaskFactory


@TaskFactory.register_task("AIME")
class AIME:
    def load():
        dataset = datasets.load_dataset("HuggingFaceH4/aime_2024", split="train")
        return dataset

    def build(sample):
        system_prompt = 'You are a helpful assistant. Think step by step and then finish your answer with "answer is X" where X is the numerical value of the answer.'

        question = sample["question"]
        truth = sample["answer_number"]

        user_prompt = question

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return {
            "input": question,
            "messages": messages,
            "truth": truth,
            "category": "aime",
        }

    def extract(text):
        matches = re.findall(r"[aA]nswer is\s*((?:(?!answer is)[\s\S])*)", text, flags=re.IGNORECASE | re.DOTALL)
        if not matches:
            return None

        last_match = matches[-1].strip()

        number_match = re.search(r"-?\d+(?:\.\d+)?", last_match)
        if number_match:
            return number_match.group(0)

        return None
