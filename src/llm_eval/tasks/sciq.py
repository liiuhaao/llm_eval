import re

import datasets

from llm_eval.task_factory import TaskFactory


@TaskFactory.register_task("SciQ")
class SciQ:
    def load():
        train_dataset = datasets.load_dataset("allenai/sciq", split="train", trust_remote_code=True)
        val_dataset = datasets.load_dataset("allenai/sciq", split="validation", trust_remote_code=True)
        test_dataset = datasets.load_dataset("allenai/sciq", split="test", trust_remote_code=True)
        dataset = datasets.concatenate_datasets([train_dataset, val_dataset, test_dataset])
        return dataset

    def build(sample):
        import random

        system_prompt = (
            'You are a helpful assistant that answers multiple-choice science questions. Think step by step and then finish your answer with "answer is (X)" where X is the correct letter choice.'
        )

        question = sample["question"]
        choices = [sample["distractor1"], sample["distractor2"], sample["distractor3"], sample["correct_answer"]]
        truth_answer = sample["correct_answer"]
        random.shuffle(choices)
        truth = chr(ord("A") + choices.index(truth_answer))

        input = f"Question: {question}\n\n"
        for i, option in enumerate(choices):
            l = chr(ord("A") + i)
            t = option.strip()
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
            "category": "sciq",
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
