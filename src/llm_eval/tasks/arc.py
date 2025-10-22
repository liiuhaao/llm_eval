import re

import datasets

from llm_eval.task_factory import TaskFactory


@TaskFactory.register_task("ARC")
class ARC:
    def load():
        dataset_info = datasets.get_dataset_infos("allenai/ai2_arc")
        bbh_subsets = list(dataset_info.keys())

        all_datasets = []
        for subset in bbh_subsets:
            train_dataset = datasets.load_dataset("allenai/ai2_arc", subset, split="train")
            val_dataset = datasets.load_dataset("allenai/ai2_arc", subset, split="validation")
            test_dataset = datasets.load_dataset("allenai/ai2_arc", subset, split="test")
            subset_dataset = datasets.concatenate_datasets([train_dataset, val_dataset, test_dataset])
            subset_dataset = subset_dataset.add_column("category", ["arc_" + subset] * len(subset_dataset))
            all_datasets.append(subset_dataset)

        dataset = datasets.concatenate_datasets(all_datasets)
        return dataset

    def build(sample):
        system_prompt = (
            'You are a helpful assistant that answers multiple-choice science questions. Think step by step and then finish your answer with "answer is (X)" where X is the correct letter choice.'
        )

        question = sample["question"]
        options = sample["choices"]
        truth = sample["answerKey"]
        category = sample["category"]

        input = f"{question}\n"
        for i, t in enumerate(options):
            input += f"{chr(65+i)}. {t}\n"
        input += "\n"
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
