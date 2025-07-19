import re

import datasets

from llm_eval.task_factory import TaskFactory


@TaskFactory.register_task("Winogrande")
class Winogrande:
    def load():
        train_dataset = datasets.load_dataset("allenai/winogrande", "winogrande_debiased", split="train", trust_remote_code=True)
        val_dataset = datasets.load_dataset("allenai/winogrande", "winogrande_debiased", split="validation", trust_remote_code=True)
        dataset = datasets.concatenate_datasets([train_dataset, val_dataset])
        return dataset

    def build(sample):
        system_prompt = 'Given a sentence with an ambiguous pronoun and two options, determine which noun the pronoun refers to. Think step by step and then finish your answer with "answer is (X)" where X is the correct letter choice.'

        question = sample["sentence"]

        input = f"Question: {question}\n"
        input += f"A. {sample['option1']}\n"
        input += f"B. {sample['option2']}\n"
        truth = "A" if sample["answer"] == "1" else "B"
        user_prompt = input

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return {
            "input": question,
            "messages": messages,
            "truth": truth,
            "category": "winogrande",
        }

    def extract(text):
        def extract_answer(text):
            pattern = r"answer is \(?([A-B])\)?"
            match = re.search(pattern, text)
            if match:
                return match.group(1)
            else:
                return extract_again(text)

        def extract_again(text):
            match = re.search(r".*[aA]nswer:\s*([A-B])", text)
            if match:
                return match.group(1)
            else:
                return extract_final(text)

        def extract_final(text):
            pattern = r"\b[A-B]\b(?!.*\b[A-B]\b)"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(0)
            else:
                return None

        return extract_answer(text)
