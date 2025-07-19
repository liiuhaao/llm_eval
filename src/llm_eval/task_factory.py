# task_factory.py
import importlib
import os
from pathlib import Path


class TaskFactory:
    _registry = {}

    @classmethod
    def register_task(cls, name):
        def decorator(task_cls):
            if name in cls._registry:
                raise ValueError(f"Task {name} already registered")
            cls._registry[name] = task_cls
            return task_cls

        return decorator

    @classmethod
    def get_task(cls, task_name):
        task_class = cls._registry.get(task_name)
        if not task_class:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown task: {task_name}. Available tasks: {available}")
        return task_class

    @classmethod
    def list_tasks(cls):
        return list(cls._registry.keys())

    @classmethod
    def auto_discover_tasks(cls):
        tasks_dir = Path(__file__).parent / "tasks"
        for file in os.listdir(tasks_dir):
            if file.endswith(".py") and not file.startswith("__"):
                module_name = f"llm_eval.tasks.{file[:-3]}"
                importlib.import_module(module_name)
