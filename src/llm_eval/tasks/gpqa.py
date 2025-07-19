import re

import datasets

from llm_eval.task_factory import TaskFactory


@TaskFactory.register_task("GPQA")
class GPQA:
    def load():
        dataset = datasets.load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        return dataset

    def build(sample):
        import random

        system_prompt = 'You are a helpful assistant. Think step by step and then finish your answer with "answer is (X)" where X is the correct letter choice.'

        question = sample["Question"]
        options = [sample["Correct Answer"], sample["Incorrect Answer 1"], sample["Incorrect Answer 2"], sample["Incorrect Answer 3"]]
        truth_answer = sample["Correct Answer"]
        random.shuffle(options)

        truth = chr(ord("A") + options.index(truth_answer))
        category = "gpqamain_" + sample["Subdomain"]

        input = f"{question}\n"
        for i, option in enumerate(options):
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
            "category": category,
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
