import re

import datasets

from llm_eval.task_factory import TaskFactory


@TaskFactory.register_task("MedMCQA")
class MedMCQA:
    def load():
        dataset = datasets.load_dataset("openlifescienceai/medmcqa", split="validation")
        return dataset

    def build(sample):
        system_prompt = 'You are a medical assistant that answers medical questions. Think step by step and then finish your answer with "answer is (X)" where X is the correct letter choice.'

        question = sample["question"]
        options = [sample["opa"], sample["opb"], sample["opc"], sample["opd"]]
        truth = chr(65 + int(sample["cop"]))
        category = f"medmcqa_{sample['subject_name']}"

        input = f"{question}\n\n"
        for i, option in enumerate(options):
            input += f"{chr(65 + i)}. {option}\n"

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
