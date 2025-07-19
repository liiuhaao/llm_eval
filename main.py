import os
import json
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from tqdm import tqdm

from llm_eval.task_factory import TaskFactory


def arg_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--local_dir", type=str, default="/home/liuhao/huggingface")
    parser.add_argument("--output_file", type=str, default="data/data.jsonl")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=1)
    return parser.parse_args()


def load_model_list(csv_path="models.csv"):
    import pandas as pd

    df = pd.read_csv(csv_path)
    model_list = df["model_name"].tolist()
    return model_list


def get_processed_keys(output_file):
    processed = set()

    if not os.path.exists(output_file):
        return processed

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                key = (data["model"], data["dataset"], data["dataset_index"])
                processed.add(key)
            except Exception as e:
                continue

    return processed


def load_model(args, model):
    model_name = model.split("/")[-1]
    model_path = os.path.join(args.local_dir, model_name)
    os.makedirs(model_path, exist_ok=True)

    print(f">>> Model {model} downloading...")
    snapshot_download(repo_id=model, local_dir=model_path)

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampling_params = SamplingParams(
        max_tokens=8192,
        temperature=args.temperature,
        n=args.n,
    )
    print(f">>> Model {model} loaded!!!")
    return llm, sampling_params


def main():
    args = arg_parser()

    # 加载模型和数据集列表
    model_list = load_model_list() if not args.model else [args.model]
    TaskFactory.auto_discover_tasks()
    task_list = TaskFactory.list_tasks()
    print(f">>> Available tasks: {task_list}")

    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # 读取已处理的样本
    if args.overwrite:
        processed_samples = set()
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write("")
    else:
        processed_samples = get_processed_keys(args.output_file) if not args.overwrite else set()

    # 遍历模型和任务
    for model_idx, model in enumerate(model_list):
        print(f">>> Processing model {model_idx + 1}/{len(model_list)}: {model}")
        model_name = model.split("/")[-1]
        llm, sampling_params = None, None

        for task_name in task_list:
            if args.task and task_name != args.task:
                continue
            print(f">>> Task {task_name} loading...")
            task = TaskFactory.get_task(task_name)
            dataset = task.load()

            # 收集需要处理的索引
            remaining_indices = []
            for idx in range(len(dataset)):
                key = (model_name, task_name, idx)
                if key not in processed_samples:
                    remaining_indices.append(idx)

            # 如果不存在需要处理的样本，跳过
            if not remaining_indices:
                print(f">>> Skipping {model_name} on {task_name}, already completed.")
                continue

            # 如果有剩余样本，加载模型
            if llm is None:
                llm, sampling_params = load_model(args, model)

            bar = tqdm(total=len(remaining_indices), desc=f"{model}[{model_idx+1}/{len(model_list)}] {task_name}", dynamic_ncols=True)

            accs = []
            for start in range(0, len(remaining_indices), args.batch_size):
                batch_indices = remaining_indices[start : start + args.batch_size]

                batch_messages = []
                batch_samples = []

                for idx in batch_indices:
                    sample = dataset[idx]
                    sample_dict = task.build(sample)
                    sample_dict["dataset_index"] = idx
                    batch_samples.append(sample_dict)
                    batch_messages.append(sample_dict["messages"])

                outputs = llm.chat(
                    messages=batch_messages,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )

                for k, output in enumerate(outputs):
                    idx = batch_indices[k]
                    sample = batch_samples[k]

                    generations = []
                    correct_list = []

                    for _, choice in enumerate(output.outputs):
                        answer_text = choice.text
                        answer = task.extract(answer_text)
                        correct = answer == sample["truth"]

                        generations.append({"output": answer_text, "answer": answer, "correct": correct})
                        correct_list.append(correct)

                    accuracy = sum(correct_list) / len(correct_list) if correct_list else 0.0

                    data = {
                        "model": model_name,
                        "dataset": task_name,
                        "dataset_index": idx,
                        "category": sample.get("category", task_name),
                        "input": sample["input"],
                        "truth": sample["truth"],
                        "messages": batch_messages[k],
                        "generations": generations,
                        "accuracy": accuracy,
                    }

                    # 写入输出文件
                    with open(args.output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")

                    accs.append(accuracy)
                    bar.update(1)
                    bar.set_postfix({"accuracy": f"{sum(accs) / len(accs):.4f}"})

            bar.close()
            print(f">>> Finished {model_name} on {task_name}")


if __name__ == "__main__":
    main()
