import inspect
import os
import pickle
from functools import wraps

from httpx import AsyncClient

func_caches = {}
## On local disk, not data/ (a shared storage box): each process rewrites whole files, so machines sharing one would clobber each other
CACHE_DIR = ".cache"


def pickle_cache(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        cache_path = f"{CACHE_DIR}/{func.__name__}.pickle"

        sig = inspect.signature(func)
        relevant_args = {}
        for (name, param), arg in zip(sig.parameters.items(), args):
            if param.annotation != AsyncClient:
                relevant_args[name] = arg
        relevant_args = frozenset(relevant_args.items())

        if func.__name__ not in func_caches:
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    func_caches[func.__name__] = pickle.load(f)
            else:
                func_caches[func.__name__] = {}

        if relevant_args not in func_caches[func.__name__]:
            result = await func(*args, **kwargs)
            ## Don't persist failed API responses, they would stick until the cache file is deleted
            if isinstance(result, dict) and result.get("statusCode", 200) != 200:
                return result
            func_caches[func.__name__][relevant_args] = result
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(func_caches[func.__name__], f)

        return func_caches[func.__name__][relevant_args]

    return wrapper
