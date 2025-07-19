from curses.ascii import isalpha
import re

import datasets

from llm_eval.task_factory import TaskFactory


@TaskFactory.register_task("MathQA")
class MathQA:
    def load():
        dataset = datasets.load_dataset("allenai/math_qa", split="test", trust_remote_code=True)
        return dataset

    def build(sample):
        system_prompt = 'You are a math problem solver. Think step by step and then finish your answer with "answer is (X)" where X is the correct letter choice.'

        question = sample["Problem"]
        options = sample["options"]
        truth = sample["correct"]
        category = f"mathqa_{sample['category']}"

        input = f"{question}\n{options}"
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
            pattern = r"answer is \(?([a-e])\)?"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
            else:
                return extract_again(text)

        def extract_again(text):
            match = re.search(r".*[aA]nswer:\s*([a-e])", text, re.IGNORECASE)
            if match:
                return match.group(1)
            else:
                return extract_final(text)

        def extract_final(text):
            matches = re.findall(r"\b([a-e])\b", text, re.IGNORECASE)
            if matches:
                return matches[-1].lower()
            else:
                return None

        answer = extract_answer(text)
        return answer.lower() if answer and answer.isalpha() else answer
