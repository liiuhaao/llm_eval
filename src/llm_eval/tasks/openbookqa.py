import re

import datasets

from llm_eval.task_factory import TaskFactory


@TaskFactory.register_task("OpenBookQA")
class OpenBookQA:
    def load():
        train_dataset = datasets.load_dataset("allenai/openbookqa", "main", split="train", trust_remote_code=True)
        val_dataset = datasets.load_dataset("allenai/openbookqa", "main", split="validation", trust_remote_code=True)
        test_dataset = datasets.load_dataset("allenai/openbookqa", "main", split="test", trust_remote_code=True)
        dataset = datasets.concatenate_datasets([train_dataset, val_dataset, test_dataset])
        return dataset

    def build(sample):
        import random

        system_prompt = (
            'You are a helpful assistant that answers multiple-choice science questions. Think step by step and then finish your answer with "answer is (X)" where X is the correct letter choice.'
        )

        question = sample["question_stem"]
        choices = sample["choices"]
        truth = sample["answerKey"]

        input = f"Question: {question}\n\n"
        for t, l in zip(choices["text"], choices["label"]):
            l = l.strip()
            t = t.strip()
            input += f"{l}. {t}\n"
        user_prompt = input

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return {
            "input": question,
            "messages": messages,
            "truth": truth,
            "category": "openbookqa",
        }

    def extract(text):
        def extract_answer(text):
            pattern = r"answer is \(?([A-D])\)?"
            match = re.search(pattern, text)
            if match:
                return match.group(1)
            else:
                return extract_again(text)

        def extract_again(text):
            match = re.search(r".*[aA]nswer:\s*([A-D])", text)
            if match:
                return match.group(1)
            else:
                return extract_final(text)

        def extract_final(text):
            pattern = r"\b[A-D]\b(?!.*\b[A-D]\b)"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(0)
            else:
                return None

        return extract_answer(text)
