# task_factory.py
import importlib
import os
from functools import wraps
from pathlib import Path

from llm_eval.cache import get_cache_dir_for, has_cache_for, load_task_cache, save_task_cache


class TaskFactory:
    _registry = {}

    @classmethod
    def register_task(cls, name):
        def decorator(task_cls):
            if name in cls._registry:
                raise ValueError(f"Task {name} already registered")

            load_fn = getattr(task_cls, "load", None)
            if callable(load_fn):

                @wraps(load_fn)
                def wrapped_load(*args, cache_dir: str = None, reload: bool = False, **kwargs):
                    key = f"task::{name}"
                    cache_path = get_cache_dir_for(key, cache_dir)

                    if not reload and has_cache_for(key, cache_dir):
                        try:
                            return load_task_cache(cache_path)
                        except FileNotFoundError:
                            pass

                    result = load_fn(*args, **kwargs)

                    try:
                        save_task_cache(cache_path, result)
                    except Exception:
                        pass

                    return result

                setattr(task_cls, "load", staticmethod(wrapped_load))

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
