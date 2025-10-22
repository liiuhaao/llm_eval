import hashlib
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union

from loguru import logger

DEFAULT_CACHE_DIR = Path(os.environ.get("LLMEVAL_CACHE_DIR", Path.home() / ".cache" / "llm_eval"))


def _key_hash(key: str):
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def get_cache_dir_for(key, cache_dir):
    base = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    safe = key.replace("/", "_")
    return base / f"{safe}_{_key_hash(safe)}"


def has_cache_for(key, cache_dir):
    path = get_cache_dir_for(key, cache_dir)
    pkl = path.with_suffix(".pkl")
    return path.exists() or pkl.exists()


def load_task_cache(path):
    pkl = path.with_suffix(".pkl")
    if pkl.exists():
        try:
            with open(pkl, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to unpickle cache {pkl}: {e}")

    raise FileNotFoundError(f"No usable cache found at {path} or {pkl}")


def save_task_cache(path, obj):
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    pkl = path.with_suffix(".pkl")
    tmp = pkl.with_suffix(".tmp")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
        tmp.replace(pkl)
        logger.info(f"Saved pickle cache to {pkl}")
    except Exception as e:
        logger.warning(f"Failed to pickle cache to {pkl}: {e}")
