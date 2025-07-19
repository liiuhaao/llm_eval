import re

import datasets

from llm_eval.task_factory import TaskFactory


@TaskFactory.register_task("SocialIQA")
class SocialIQA:
    def load():
        # train_dataset = datasets.load_dataset("allenai/social_i_qa", split="train", trust_remote_code=True)
        val_dataset = datasets.load_dataset("allenai/social_i_qa", split="validation", trust_remote_code=True)
        # dataset = datasets.concatenate_datasets([train_dataset, val_dataset])
        return val_dataset

    def build(sample):

        system_prompt = (
            'You are a helpful assistant that answers multiple-choice science questions. Think step by step and then finish your answer with "answer is (X)" where X is the correct letter choice.'
        )

        question = sample["context"] + "\n" + sample["question"]

        input = f"{question}\n"
        input += f"A. {sample['answerA']}\n"
        input += f"B. {sample['answerB']}\n"
        input += f"C. {sample['answerC']}\n"
        truth = "A" if sample["label"] == "1" else "B" if sample["label"] == "2" else "C"
        user_prompt = input

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return {
            "input": question,
            "messages": messages,
            "truth": truth,
            "category": "socialiqa",
        }

    def extract(text):
        def extract_answer(text):
            pattern = r"answer is \(?([A-C])\)?"
            match = re.search(pattern, text)
            if match:
                return match.group(1)
            else:
                return extract_again(text)

        def extract_again(text):
            match = re.search(r".*[aA]nswer:\s*([A-C])", text)
            if match:
                return match.group(1)
            else:
                return extract_final(text)

        def extract_final(text):
            pattern = r"\b[A-C]\b(?!.*\b[A-C]\b)"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(0)
            else:
                return None

        return extract_answer(text)
